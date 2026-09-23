from unirobosim_isaaclab.config import IsaacLabAdapterConfig
from unirobosim_isaaclab.runtime_pool import RuntimePool


class Runtime:
    reusable = True
    terminated = False

    def __init__(self, config):
        self.config = config
        self.closed = 0

    def close(self):
        self.closed += 1
        self.terminated = True


def test_idle_lease_reuses_worker_but_active_leases_are_distinct():
    pool = RuntimePool()
    config = IsaacLabAdapterConfig()
    one = pool.acquire(config, Runtime)
    other = pool.acquire(config, Runtime)
    assert one._runtime is not other._runtime
    one.close()
    again = pool.acquire(config, Runtime)
    assert again is not one and again._runtime is one._runtime
    again.close()
    other.close()
    assert other._runtime.closed == 1
    pool.close()
    assert one._runtime.closed == 1


def test_invalid_failed_lease_is_not_reused():
    pool = RuntimePool()
    config = IsaacLabAdapterConfig()
    first = pool.acquire(config, Runtime)
    first.invalidate()
    first.close()
    assert first._runtime.closed == 1
    second = pool.acquire(config, Runtime)
    assert second._runtime is not first._runtime
    pool.close()


def test_render_configuration_change_retires_incompatible_worker():
    pool = RuntimePool()
    first = pool.acquire(IsaacLabAdapterConfig(enable_cameras=False), Runtime)
    first.close()
    second = pool.acquire(IsaacLabAdapterConfig(enable_cameras=True), Runtime)
    assert first._runtime.closed == 1
    assert second._runtime is not first._runtime
    pool.close()


def test_unhealthy_idle_worker_is_retired():
    pool = RuntimePool()
    config = IsaacLabAdapterConfig()
    first = pool.acquire(config, Runtime)
    first.close()
    first._runtime.reusable = False
    second = pool.acquire(config, Runtime)
    assert first._runtime.closed == 1
    assert second._runtime is not first._runtime
    pool.close()
