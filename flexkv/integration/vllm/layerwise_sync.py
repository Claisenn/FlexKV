"""Worker-side synchronization primitives for vLLM layer-wise KV loading.

The FlexKV data plane signals completion of each layer through Linux eventfds.
This module intentionally has no torch/vLLM/c_ext dependency so its counter
state machine and Unix-domain-socket handshake can be tested on CPU-only hosts.
"""

from __future__ import annotations

import ctypes
import errno
import os
import select
import socket
import struct
import threading
import time
from dataclasses import dataclass
from typing import Callable, Sequence


_EFD_SEMAPHORE = 0x1


def _linux_eventfd(initval: int = 0, flags: int = _EFD_SEMAPHORE) -> int:
    eventfd_fn = getattr(os, "eventfd", None)
    if eventfd_fn is not None:
        return int(eventfd_fn(initval, flags))

    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    libc.eventfd.argtypes = [ctypes.c_uint, ctypes.c_int]
    libc.eventfd.restype = ctypes.c_int
    fd = libc.eventfd(ctypes.c_uint(initval), ctypes.c_int(flags))
    if fd < 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))
    return int(fd)


def _settle_eventfd(fd: int, timeout_s: float) -> bool:
    """Consume one pending unit from fd, waiting at most timeout_s.

    Returns True if a unit was consumed. Unlike _read_eventfd this never raises
    on timeout: not finding a unit is an expected outcome when the transfer that
    owed it was cancelled.
    """
    try:
        ready, _, _ = select.select([fd], [], [], timeout_s)
        if not ready:
            return False
        os.read(fd, 8)
        return True
    except OSError:
        # Closed/invalid fd during shutdown: nothing to reclaim, and nothing
        # left that could satisfy a future wait.
        return True


def _read_eventfd(fd: int) -> int:
    timeout_s = float(os.getenv("FLEXKV_LAYERWISE_WAIT_TIMEOUT_S", "60"))
    ready, _, _ = select.select([fd], [], [], timeout_s)
    if not ready:
        raise TimeoutError(
            f"timed out after {timeout_s}s waiting for layer-wise KV load")
    payload = os.read(fd, 8)
    if len(payload) != 8:
        raise OSError(errno.EIO, f"short eventfd read: {len(payload)} bytes")
    return int(struct.unpack("Q", payload)[0])


def _send_fds(sock: socket.socket, fds: Sequence[int], counter_id: int) -> None:
    packed_fds = struct.pack(f"{len(fds)}i", *fds)
    sock.sendmsg(
        [struct.pack("i", counter_id)],
        [(socket.SOL_SOCKET, socket.SCM_RIGHTS, packed_fds)],
    )


@dataclass(frozen=True)
class LayerwiseLoadMetadata:
    """Serializable scheduler-to-worker metadata for one model step."""

    enabled: bool = False
    counter_id: int = -1
    has_load: bool = False


class LayerwiseStepCoordinator:
    """Scheduler-side deterministic counter assignment."""

    def __init__(self, enabled: bool, num_counters: int = 3) -> None:
        if num_counters <= 0:
            raise ValueError("num_counters must be positive")
        self.enabled = enabled
        self.num_counters = num_counters
        self._next_counter = 0

    def build_metadata(self, has_load: bool) -> LayerwiseLoadMetadata:
        if not self.enabled or not has_load:
            return LayerwiseLoadMetadata(enabled=self.enabled)
        counter_id = self._next_counter
        self._next_counter = (self._next_counter + 1) % self.num_counters
        return LayerwiseLoadMetadata(
            enabled=True,
            counter_id=counter_id,
            has_load=True,
        )

    def external_match_is_async(self, needs_load: bool) -> bool:
        """Whether vLLM should enter WAITING_FOR_REMOTE_KVS.

        Layer-wise loads must run in the same model step so attention-layer
        hooks can wait on their eventfds. Non-layer-wise loads retain the
        existing whole-transfer async behavior.
        """
        return bool(needs_load and not self.enabled)

    def launch_kwargs(self, metadata: LayerwiseLoadMetadata) -> dict[str, object]:
        # A no-load step carries counter_id=-1. Clamping that to 0 would make
        # the data plane signal counter 0 -- which a concurrent real load may
        # own -- so its layers could see units that belong to no transfer.
        # Only a metadata that actually has a load may drive a layer-wise
        # launch; otherwise fall back to the non-layer-wise path.
        if not metadata.enabled or not metadata.has_load:
            return {
                "as_batch": metadata.enabled,
                "layerwise_transfer": False,
                "counter_id": 0,
            }
        self._validate_launch_counter(metadata.counter_id)
        return {
            "as_batch": metadata.enabled,
            "layerwise_transfer": metadata.enabled,
            "counter_id": metadata.counter_id,
        }

    def _validate_launch_counter(self, counter_id: int) -> None:
        if counter_id < 0 or counter_id >= self.num_counters:
            raise ValueError(
                f"layer-wise launch counter_id={counter_id} outside "
                f"[0, {self.num_counters})"
            )


class LayerwiseCounterPool:
    """Triple-buffered per-layer eventfd synchronization.

    The scheduler chooses a counter id for each layer-wise batch and sends it
    through connector metadata. The worker binds that id at forward start;
    each attention layer then consumes exactly one semaphore unit from its
    eventfd. A counter can only be reused after its final layer was consumed.
    """

    def __init__(
        self,
        layer_names: Sequence[str],
        num_counters: int = 3,
        fd_factory: Callable[[], int] = _linux_eventfd,
        fd_reader: Callable[[int], int] = _read_eventfd,
        fd_closer: Callable[[int], None] = os.close,
        fd_settler: Callable[[int, float], bool] = _settle_eventfd,
    ) -> None:
        if not layer_names:
            raise ValueError("layer_names must not be empty")
        if len(set(layer_names)) != len(layer_names):
            raise ValueError("layer_names must be unique")
        if num_counters <= 0:
            raise ValueError("num_counters must be positive")

        self.layer_names = tuple(layer_names)
        self.layer_to_index = {
            layer_name: index for index, layer_name in enumerate(self.layer_names)
        }
        self.num_layers = len(self.layer_names)
        self.num_counters = num_counters
        self._fd_reader = fd_reader
        self._fd_closer = fd_closer
        self._fd_settler = fd_settler
        self._fds = [
            [fd_factory() for _ in range(self.num_layers)]
            for _ in range(num_counters)
        ]
        self._waited = [[False] * self.num_layers for _ in range(num_counters)]
        self._active_counter = -1
        self._lock = threading.Lock()
        self._closed = False
        self._failed_error: BaseException | None = None
        # Layers of a counter whose batch was launched but whose completion
        # signal has not been accounted for yet. See _reclaim_counter().
        self._owed = [[False] * self.num_layers for _ in range(num_counters)]
        # Layers already given their one blocking chance to settle, so a
        # transfer that never signals costs a single bounded wait rather than
        # one on every reuse.
        self._settle_attempted = [
            [False] * self.num_layers for _ in range(num_counters)
        ]
        self._settle_timeout_s = float(
            os.getenv("FLEXKV_LAYERWISE_SETTLE_TIMEOUT_S", "1.0"))

    @property
    def fds(self) -> tuple[tuple[int, ...], ...]:
        return tuple(tuple(counter_fds) for counter_fds in self._fds)

    def _reclaim_counter(self, counter_id: int) -> None:
        """Reclaim a counter whose forward pass ended without waiting all layers.

        This CANNOT be a non-blocking drain of what happens to be readable now.
        The data plane launches the batch with sync=false and signals each layer
        later, from cudaLaunchHostFunc or the polling thread, so a signal
        belonging to the abandoned batch can still arrive long after the forward
        gave up. A best-effort drain would miss it, and that leftover unit would
        later satisfy a wait() on this counter after it rotates back -- letting
        attention run against KV that has not landed.

        What saves us is that the outstanding signals are *countable*: the data
        plane emits exactly one unit per layer per launched batch. So instead of
        guessing, record the layers still owed a signal and settle the debt
        before the counter is used again: _await_owed() blocks for exactly the
        missing units. Reclaiming is deferred to the point of reuse so an
        aborted forward does not pay for a transfer nobody is waiting on.
        """
        with self._lock:
            if counter_id < 0 or counter_id >= self.num_counters:
                return
            for index, done in enumerate(self._waited[counter_id]):
                if not done:
                    self._owed[counter_id][index] = True
            self._waited[counter_id] = [False] * self.num_layers
            if self._active_counter == counter_id:
                self._active_counter = -1

    def _await_owed(self, counter_id: int) -> None:
        """Consume the signals still owed to counter_id before it is reused.

        Waits per layer, because an owed unit may legitimately still be in
        flight -- that is exactly the signal that must not survive into the next
        batch. But the wait is bounded and non-fatal: a cancelled or failed
        transfer may never signal at all, and blocking a fresh step forever (or
        failing it) to collect a unit that is not coming would be worse than the
        wedge this whole path exists to avoid.

        A layer that does not settle within the window stays marked, and every
        later reuse re-checks it without blocking. So the blocking cost is paid
        at most once per abandoned batch, while a unit that shows up much later
        is still caught before it can be mistaken for a real completion.
        """
        with self._lock:
            owed = [
                index for index, is_owed in enumerate(self._owed[counter_id])
                if is_owed
            ]
            waited_once = list(self._settle_attempted[counter_id])
        if not owed:
            return
        deadline = time.monotonic() + self._settle_timeout_s
        for index in owed:
            fd = self._fds[counter_id][index]
            # Block only on the first attempt for this layer. After that the
            # transfer is presumed cancelled or dead, so just sweep whatever
            # may have trickled in since.
            remaining = (max(0.0, deadline - time.monotonic())
                         if not waited_once[index] else 0.0)
            settled = self._fd_settler(fd, remaining)
            with self._lock:
                # Settled: the debt is closed and the layer starts fresh next
                # time. Unsettled: remember that it already had its blocking
                # chance, so later reuses only sweep.
                self._settle_attempted[counter_id][index] = not settled
                self._owed[counter_id][index] = not settled

    def release(self, counter_id: int) -> None:
        self._validate_counter(counter_id)
        with self._lock:
            self._waited[counter_id] = [False] * self.num_layers
            # Every layer was waited on, so every signal of this batch has been
            # consumed and the counter owes nothing.
            self._owed[counter_id] = [False] * self.num_layers
            if self._active_counter == counter_id:
                self._active_counter = -1

    def bind(self, metadata: LayerwiseLoadMetadata) -> None:
        """Bind the counter used by the current vLLM forward step."""
        # NOTE: a prior wait() failure is recorded but is NOT fatal here.
        # _read_eventfd raises TimeoutError after
        # FLEXKV_LAYERWISE_WAIT_TIMEOUT_S (default 60s), which a merely slow
        # load can hit. Poisoning the pool forever would turn one slow transfer
        # into a dead engine for the rest of the process's life.
        #
        # Caveat, so nobody builds on a guarantee that is not there: vLLM calls
        # wait_for_layer_load() without a try/except (see
        # model_executor/layers/attention/kv_transfer_utils.py), so an exception
        # propagates out of the model forward and its
        # kv_load_failure_policy -- which only covers invalid_block_ids reported
        # through connector output -- does not apply. A timeout may well take
        # the engine down. Not poisoning the pool is about not turning a
        # *recovered* step into a permanent brick; it does not by itself make a
        # failed load recoverable.
        stale = self._active_counter
        if stale >= 0:
            # A forward pass does not always consume every layer: vLLM can
            # abort, preempt, or raise between attention layers, and then no
            # further wait() calls arrive for that counter. Treating this as
            # fatal wedges the engine permanently on the next step -- the
            # counter can never be released because release only happens once
            # ALL layers have been waited on. Reclaim it instead.
            self._reclaim_counter(stale)
        self._failed_error = None
        if not metadata.enabled or not metadata.has_load:
            self._active_counter = -1
            return
        self._validate_counter(metadata.counter_id)
        # Settle any signal still owed to this counter from an earlier batch
        # BEFORE binding it. Those units are indistinguishable from this step's
        # own, so consuming them here is what stops a stale completion from
        # satisfying one of the waits below.
        self._await_owed(metadata.counter_id)
        with self._lock:
            self._waited[metadata.counter_id] = [False] * self.num_layers
            # This batch owes one signal per layer. Assign rather than OR: a
            # layer left unsettled above is already True and stays True, and one
            # that settled is genuinely starting a fresh debt.
            self._owed[metadata.counter_id] = [True] * self.num_layers
            self._active_counter = metadata.counter_id

    def wait(self, layer_name: str) -> None:
        counter_id = self._active_counter
        if counter_id < 0:
            return
        try:
            layer_index = self.layer_to_index[layer_name]
        except KeyError as exc:
            raise KeyError(
                f"unknown layer_name={layer_name!r}; registered layers="
                f"{self.layer_names!r}"
            ) from exc

        with self._lock:
            if self._waited[counter_id][layer_index]:
                return
        try:
            self._fd_reader(self._fds[counter_id][layer_index])
        except BaseException as exc:
            self._failed_error = exc
            # Hand the counter to _reclaim_counter() rather than just clearing
            # _active_counter: a timed-out layer is precisely the case where the
            # signal is still in flight, and dropping the identity here would
            # make the next bind() see no stale counter and skip settling it --
            # the leftover unit then satisfies a later wait() on this counter.
            self._reclaim_counter(counter_id)
            raise
        with self._lock:
            self._waited[counter_id][layer_index] = True
            self._owed[counter_id][layer_index] = False
            finished = all(self._waited[counter_id])
        if finished:
            self.release(counter_id)

    def send_to_worker(
        self,
        socket_path: str,
        tp_rank_per_node: int,
        tp_size_per_node: int,
        timeout_s: float = 360.0,
        retry_interval_s: float = 0.05,
        cancel_event: "threading.Event | None" = None,
    ) -> None:
        """Send all counter eventfds to the LayerwiseTransferWorker.

        This method is intended to run in a background thread before GPU-cache
        registration, because registration may start a worker that blocks while
        waiting for this handshake.

        ``cancel_event`` lets a caller abandon the retry loop early. Shutdown
        needs it: otherwise this thread keeps retrying for the full timeout_s
        while holding eventfds that the caller wants to close.
        """
        deadline = time.monotonic() + timeout_s
        last_error: BaseException | None = None
        while time.monotonic() < deadline:
            if cancel_event is not None and cancel_event.is_set():
                return
            try:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                    sock.connect(socket_path)
                    sock.sendall(
                        struct.pack(
                            "iiii",
                            tp_rank_per_node,
                            tp_size_per_node,
                            self.num_layers,
                            self.num_counters,
                        )
                    )
                    for counter_id, fds in enumerate(self._fds):
                        _send_fds(sock, fds, counter_id)
                    sock.settimeout(min(30.0, max(1.0, timeout_s)))
                    ack = sock.recv(1)
                    if ack != b"\x01":
                        raise RuntimeError(
                            f"LayerwiseTransferWorker rejected eventfds: {ack!r}"
                        )
                    return
            except (OSError, RuntimeError, TimeoutError) as exc:
                last_error = exc
                if cancel_event is not None:
                    # Interruptible sleep: a plain time.sleep() would ignore a
                    # cancel that arrives during the backoff.
                    if cancel_event.wait(retry_interval_s):
                        return
                else:
                    time.sleep(retry_interval_s)
        raise RuntimeError(
            f"timed out sending layer-wise eventfds to {socket_path}: {last_error}"
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for counter_fds in self._fds:
            for fd in counter_fds:
                self._fd_closer(fd)
        self._fds.clear()

    def _validate_counter(self, counter_id: int) -> None:
        if counter_id < 0 or counter_id >= self.num_counters:
            raise ValueError(
                f"counter_id={counter_id} outside [0, {self.num_counters})"
            )

    def __del__(self) -> None:
        self.close()
