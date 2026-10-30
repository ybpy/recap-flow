from concurrent.futures import ThreadPoolExecutor
import threading

import yaml

from openpi_client.e0.preflight import configure_libero


def test_parallel_config_writers_never_expose_partial_yaml(tmp_path, monkeypatch):
    monkeypatch.setenv("LIBERO_CONFIG_PATH", str(tmp_path))
    expected = configure_libero()
    barrier = threading.Barrier(9)

    def writer():
        barrier.wait()
        for _ in range(100):
            configure_libero()

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(writer) for _ in range(8)]
        barrier.wait()
        reads = 0
        while not all(future.done() for future in futures) or reads < 100:
            assert yaml.safe_load((tmp_path / "config.yaml").read_text()) == expected
            reads += 1
        for future in futures:
            future.result()
    assert not list(tmp_path.glob(".config-*.yaml"))
