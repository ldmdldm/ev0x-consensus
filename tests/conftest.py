import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "asyncio: mark test as an async test"
    )


@pytest.fixture(scope="session")
def event_loop():
    import asyncio
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
