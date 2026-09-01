import array
import os
import socket
import struct
import tempfile
import threading
import time

import pytest

from flexkv.integration.vllm.layerwise_sync import (
    LayerwiseCounterPool,
    LayerwiseLoadMetadata,
    LayerwiseStepCoordinator,
)


pytestmark = pytest.mark.unit


def _short_socket_path(prefix):
    fd, path = tempfile.mkstemp(prefix=prefix, suffix=".sock", dir="/tmp")
    os.close(fd)
    os.unlink(path)
    return path


class _FakeFDs:
    """Stand-in for the eventfd layer.

    ``pending`` models units the data plane has signalled but nobody consumed
    yet, so settle() can distinguish "a stale unit was reclaimed" from "there
    was nothing to reclaim" without touching real descriptors.
    """

    def __init__(self):
        self.next_fd = 10
        self.reads = []
        self.closed = []
        self.settled = []
        self.pending = set()

    def create(self):
        fd = self.next_fd
        self.next_fd += 1
        return fd

    def read(self, fd):
        self.reads.append(fd)
        self.pending.discard(fd)
        return 1

    def settle(self, fd, _timeout_s):
        self.settled.append(fd)
        if fd in self.pending:
            self.pending.discard(fd)
            return True
        return False

    def close(self, fd):
        self.closed.append(fd)


def _pool(layer_names=("layer.0", "layer.1", "layer.2")):
    fake = _FakeFDs()
    pool = LayerwiseCounterPool(
        layer_names,
        fd_factory=fake.create,
        fd_reader=fake.read,
        fd_closer=fake.close,
        fd_settler=fake.settle,
    )
    return pool, fake


def test_step_coordinator_round_robins_only_load_steps():
    coordinator = LayerwiseStepCoordinator(enabled=True, num_counters=3)
    assert coordinator.build_metadata(False) == LayerwiseLoadMetadata(enabled=True)
    assert [coordinator.build_metadata(True).counter_id for _ in range(5)] == [
        0, 1, 2, 0, 1]

    disabled = LayerwiseStepCoordinator(enabled=False)
    assert disabled.build_metadata(True) == LayerwiseLoadMetadata(enabled=False)
    assert coordinator.external_match_is_async(True) is False
    assert disabled.external_match_is_async(True) is True
    assert disabled.external_match_is_async(False) is False
    assert coordinator.launch_kwargs(
        LayerwiseLoadMetadata(enabled=True, counter_id=2, has_load=True)
    ) == {
        "as_batch": True,
        "layerwise_transfer": True,
        "counter_id": 2,
    }


def test_counter_pool_waits_once_per_layer_and_releases_on_last_layer():
    pool, fake = _pool()
    pool.bind(LayerwiseLoadMetadata(enabled=True, counter_id=1, has_load=True))

    pool.wait("layer.0")
    pool.wait("layer.0")
    pool.wait("layer.1")
    assert fake.reads == [13, 14]

    pool.wait("layer.2")
    assert fake.reads == [13, 14, 15]

    # Last-layer completion releases the active counter; later hooks are no-op.
    pool.wait("layer.0")
    assert fake.reads == [13, 14, 15]


def test_counter_pool_disabled_or_empty_metadata_is_noop():
    pool, fake = _pool()
    pool.bind(LayerwiseLoadMetadata(enabled=False))
    pool.wait("layer.0")
    pool.bind(LayerwiseLoadMetadata(enabled=True, counter_id=-1, has_load=False))
    pool.wait("layer.1")
    assert fake.reads == []


def test_counter_pool_rejects_unknown_layer_and_counter():
    pool, _ = _pool()
    with pytest.raises(ValueError, match="counter_id"):
        pool.bind(LayerwiseLoadMetadata(enabled=True, counter_id=9, has_load=True))

    pool.bind(LayerwiseLoadMetadata(enabled=True, counter_id=0, has_load=True))
    with pytest.raises(KeyError, match="unknown layer_name"):
        pool.wait("layer.99")


def test_failed_wait_recovers_on_next_bind():
    """A per-layer wait failure must be transient, not terminal.

    _read_eventfd raises TimeoutError after FLEXKV_LAYERWISE_WAIT_TIMEOUT_S
    (60s by default), which a merely slow transfer can hit. Poisoning the pool
    permanently would turn one slow load into a dead engine for the rest of the
    process, so the next bind() reclaims and reuses the counter instead.
    """
    attempts = []

    def flaky_read(fd):
        attempts.append(fd)
        if len(attempts) == 1:
            raise TimeoutError("simulated")
        return 1

    fake = _FakeFDs()
    pool = LayerwiseCounterPool(
        ("layer.0",),
        fd_factory=fake.create,
        fd_reader=flaky_read,
        fd_closer=fake.close,
        fd_settler=fake.settle,
    )
    pool.bind(LayerwiseLoadMetadata(enabled=True, counter_id=0, has_load=True))
    with pytest.raises(TimeoutError):
        pool.wait("layer.0")
    # Recovers: the counter is reclaimed and usable again. Rebinding settles
    # the debt left by the timed-out batch first, so the wait below consumes
    # this batch's own completion rather than a leftover unit.
    pool.bind(
        LayerwiseLoadMetadata(enabled=True, counter_id=0, has_load=True))
    assert fake.settled == [10]
    assert pool.wait("layer.0") is None
    assert attempts == [10, 10]


def test_bind_reclaims_counter_when_forward_skips_layers():
    """A forward pass that ends early must not wedge the pool.

    vLLM can abort, preempt, or raise between attention layers, leaving some
    layers of the bound counter unwaited. release() only fires once ALL layers
    were waited on, so rejecting the next bind() would deadlock the engine
    permanently. bind() reclaims the abandoned counter and continues.
    """
    pool, fake = _pool(layer_names=("a", "b"))
    metadata = LayerwiseLoadMetadata(enabled=True, counter_id=0, has_load=True)
    pool.bind(metadata)
    pool.wait("a")
    # "b" never waited -- forward aborted. Next step must still bind, and it
    # settles the signal still owed to "b" (fd 11) before rebinding.
    pool.bind(metadata)
    pool.wait("a")
    pool.wait("b")
    assert fake.settled == [11]
    assert fake.reads == [10, 10, 11]


def _signal(pool, counter_id, layer_index, count=1):
    """Emulate the data plane signalling one layer's completion."""
    os.write(pool.fds[counter_id][layer_index], struct.pack("Q", count))


class _Waiter:
    """A single background wait() whose completion can be polled twice.

    Deliberately ONE thread: two threads waiting on the same layer would race
    for the same semaphore unit, and whichever loses blocks forever on a unit
    that is never re-sent. vLLM calls wait_for_layer_load() synchronously from
    the forward, so a second concurrent waiter is a test artifact, not a
    scenario worth modelling.
    """

    def __init__(self, pool, layer_name):
        self._done = threading.Event()
        self._error = None

        def run():
            try:
                pool.wait(layer_name)
            except BaseException as exc:  # noqa: BLE001 - surfaced in finished()
                self._error = exc
            finally:
                self._done.set()

        threading.Thread(target=run, daemon=True).start()

    def finished(self, seconds):
        """True if the wait has returned, i.e. it consumed a unit."""
        completed = self._done.wait(seconds)
        if completed and self._error is not None:
            raise self._error
        return completed


def test_late_signal_from_abandoned_batch_cannot_satisfy_a_later_load():
    """A completion that lands after the counter was abandoned must not leak.

    The data plane launches with sync=false and signals each layer later, from
    cudaLaunchHostFunc or the polling thread, so a signal belonging to an
    abandoned batch can arrive after the forward gave up on it. If that unit
    survives, it satisfies a wait() once the counter rotates back and attention
    runs against KV that has not landed.

    Real eventfds on purpose: a fake reader cannot express "a unit arrived
    later", which is the whole failure mode.
    """
    layers = ("l0", "l1")
    pool = LayerwiseCounterPool(layers, num_counters=2)
    md = lambda cid: LayerwiseLoadMetadata(  # noqa: E731
        enabled=True, counter_id=cid, has_load=True)
    try:
        pool.bind(md(0))
        _signal(pool, 0, 0)
        pool.wait("l0")
        # Forward aborts here: "l1" is never waited on.

        pool.bind(md(1))
        # The abandoned batch's l1 completion arrives only now, after counter 0
        # was already handed back.
        _signal(pool, 0, 1)
        _signal(pool, 1, 0)
        _signal(pool, 1, 1)
        pool.wait("l0")
        pool.wait("l1")

        # Counter 0 comes back around. The data plane signals l0 only; the
        # stale unit must NOT be able to stand in for l1's real completion.
        pool.bind(md(0))
        _signal(pool, 0, 0)
        pool.wait("l0")
        waiter = _Waiter(pool, "l1")
        assert not waiter.finished(1.0), (
            "wait() was satisfied by a stale signal from the abandoned batch")

        # The same waiter unblocks only once the real completion arrives.
        _signal(pool, 0, 1)
        assert waiter.finished(5.0)
    finally:
        pool.close()


def test_timed_out_counter_settles_its_late_signal_before_reuse():
    """The timeout path must not drop the counter identity.

    wait() records the failure and hands the counter back for reclaim. If it
    merely cleared the active counter, the next bind() would see nothing stale,
    skip settling, and leave the in-flight unit to satisfy a future wait().
    """
    layers = ("l0", "l1")
    pool = LayerwiseCounterPool(layers, num_counters=2)
    md = lambda cid: LayerwiseLoadMetadata(  # noqa: E731
        enabled=True, counter_id=cid, has_load=True)
    previous = os.environ.get("FLEXKV_LAYERWISE_WAIT_TIMEOUT_S")
    os.environ["FLEXKV_LAYERWISE_WAIT_TIMEOUT_S"] = "0.2"
    try:
        pool.bind(md(0))
        with pytest.raises(TimeoutError):
            pool.wait("l0")
        # Both layers of the timed-out batch complete late.
        _signal(pool, 0, 0)
        _signal(pool, 0, 1)

        os.environ["FLEXKV_LAYERWISE_WAIT_TIMEOUT_S"] = "5"
        # Reusing counter 0 must consume both late units up front, so the waits
        # below block on this batch's own signals.
        pool.bind(md(0))
        waiter = _Waiter(pool, "l0")
        assert not waiter.finished(1.0), (
            "wait() consumed a leftover unit from the timed-out batch")
        _signal(pool, 0, 0)
        assert waiter.finished(5.0)
    finally:
        if previous is None:
            os.environ.pop("FLEXKV_LAYERWISE_WAIT_TIMEOUT_S", None)
        else:
            os.environ["FLEXKV_LAYERWISE_WAIT_TIMEOUT_S"] = previous
        pool.close()


def test_counter_reuse_survives_a_transfer_that_never_signals():
    """A cancelled transfer must not block or fail the step that reuses it.

    Settling an abandoned batch waits for signals still in flight, but a
    cancelled or dead transfer may never signal at all. Blocking a fresh step
    indefinitely -- or failing it -- to collect a unit that is not coming would
    be worse than the wedge this path exists to prevent, so the wait is bounded
    and taken at most once per layer.
    """
    layers = ("l0", "l1")
    md = lambda cid: LayerwiseLoadMetadata(  # noqa: E731
        enabled=True, counter_id=cid, has_load=True)
    previous = os.environ.get("FLEXKV_LAYERWISE_SETTLE_TIMEOUT_S")
    os.environ["FLEXKV_LAYERWISE_SETTLE_TIMEOUT_S"] = "0.3"
    # Constructed after the env var: the settle window is read at __init__.
    pool = LayerwiseCounterPool(layers, num_counters=2)
    try:
        pool.bind(md(0))
        _signal(pool, 0, 0)
        pool.wait("l0")
        # Forward aborts and the transfer is cancelled: l1 is never signalled.
        pool.bind(md(1))

        started = time.monotonic()
        pool.bind(md(0))  # must not raise, must not hang
        first_reuse = time.monotonic() - started
        assert first_reuse < 5.0

        # Complete this batch cleanly so the only debt left on counter 0 is the
        # l1 unit that is never coming.
        _signal(pool, 0, 0)
        _signal(pool, 0, 1)
        pool.wait("l0")
        pool.wait("l1")

        started = time.monotonic()
        pool.bind(md(0))  # the never-signalled layer must not be waited on again
        second_reuse = time.monotonic() - started
        assert second_reuse < 0.2, (
            f"blocking settle repeated on every reuse ({second_reuse:.2f}s)")

        # Still correct: this batch's own signal is what unblocks the wait.
        _signal(pool, 0, 0)
        pool.wait("l0")
    finally:
        if previous is None:
            os.environ.pop("FLEXKV_LAYERWISE_SETTLE_TIMEOUT_S", None)
        else:
            os.environ["FLEXKV_LAYERWISE_SETTLE_TIMEOUT_S"] = previous
        pool.close()


def test_send_to_worker_returns_promptly_when_cancelled():
    """Shutdown must be able to stop the handshake before its own deadline.

    Nothing is listening on this path, so send_to_worker() would otherwise retry
    for the full timeout_s while holding fds the caller wants to close.
    """
    pool, _ = _pool(layer_names=("a",))
    cancel = threading.Event()
    finished = threading.Event()

    def run():
        pool.send_to_worker(
            _short_socket_path("flexkv-cancel-"),
            tp_rank_per_node=0,
            tp_size_per_node=1,
            timeout_s=300.0,
            retry_interval_s=0.05,
            cancel_event=cancel,
        )
        finished.set()

    threading.Thread(target=run, daemon=True).start()
    time.sleep(0.2)
    assert not finished.is_set()
    cancel.set()
    assert finished.wait(5.0), "send_to_worker ignored the cancel event"


def test_counter_pool_close_is_idempotent():
    pool, fake = _pool(layer_names=("a", "b"))
    expected_fds = [fd for counter in pool.fds for fd in counter]
    pool.close()
    pool.close()
    assert fake.closed == expected_fds


def test_send_to_worker_matches_layerwise_worker_wire_contract():
    socket_path = _short_socket_path("flexkv-lw-")
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(socket_path)
    server.listen(1)

    received = {}

    def receive():
        conn, _ = server.accept()
        with conn:
            metadata = conn.recv(16)
            received["metadata"] = struct.unpack("iiii", metadata)
            counters = {}
            for _ in range(3):
                msg, ancdata, _flags, _addr = conn.recvmsg(
                    4, socket.CMSG_SPACE(2 * struct.calcsize("i")))
                counter_id = struct.unpack("i", msg)[0]
                for level, kind, data in ancdata:
                    if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
                        fds = array.array("i")
                        fds.frombytes(data[:2 * fds.itemsize])
                        counters[counter_id] = list(fds)
            received["counters"] = counters
            conn.sendall(b"\x01")

    server_thread = threading.Thread(target=receive)
    server_thread.start()

    owned_write_fds = []

    def pipe_fd():
        read_fd, write_fd = os.pipe()
        owned_write_fds.append(write_fd)
        return read_fd

    pool = LayerwiseCounterPool(
        ("layer.0", "layer.1"),
        fd_factory=pipe_fd,
        fd_reader=lambda _fd: 1,
    )
    try:
        pool.send_to_worker(
            socket_path,
            tp_rank_per_node=1,
            tp_size_per_node=2,
            timeout_s=2,
            retry_interval_s=0.01,
        )
    finally:
        server_thread.join(timeout=2)
        server.close()
        if os.path.exists(socket_path):
            os.unlink(socket_path)
        for fds in received.get("counters", {}).values():
            for fd in fds:
                os.close(fd)
        pool.close()
        for fd in owned_write_fds:
            os.close(fd)

    assert received["metadata"] == (1, 2, 2, 3)
    assert set(received["counters"]) == {0, 1, 2}
    assert all(len(fds) == 2 for fds in received["counters"].values())


def test_send_to_worker_retries_after_nack():
    socket_path = _short_socket_path("flexkv-retry-")
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(socket_path)
    server.listen(2)
    attempts = []

    def receive():
        for ack in (b"\x00", b"\x01"):
            conn, _ = server.accept()
            with conn:
                conn.recv(16)
                for _ in range(3):
                    _msg, ancdata, _flags, _addr = conn.recvmsg(
                        4, socket.CMSG_SPACE(struct.calcsize("i")))
                    for level, kind, data in ancdata:
                        if (level == socket.SOL_SOCKET
                                and kind == socket.SCM_RIGHTS):
                            fds = array.array("i")
                            fds.frombytes(data[:fds.itemsize])
                            for fd in fds:
                                os.close(fd)
                attempts.append(ack)
                conn.sendall(ack)

    thread = threading.Thread(target=receive)
    thread.start()
    write_fds = []

    def pipe_fd():
        read_fd, write_fd = os.pipe()
        write_fds.append(write_fd)
        return read_fd

    pool = LayerwiseCounterPool(
        ("layer.0",), fd_factory=pipe_fd, fd_reader=lambda _fd: 1)
    try:
        pool.send_to_worker(
            socket_path,
            tp_rank_per_node=0,
            tp_size_per_node=1,
            timeout_s=2,
            retry_interval_s=0.01,
        )
    finally:
        thread.join(timeout=2)
        server.close()
        if os.path.exists(socket_path):
            os.unlink(socket_path)
        pool.close()
        for fd in write_fds:
            os.close(fd)

    assert attempts == [b"\x00", b"\x01"]
