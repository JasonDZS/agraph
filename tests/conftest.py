import pytest


@pytest.fixture(scope="session")
def setup_database():
    pass


@pytest.fixture(autouse=True)
def run_around_tests():
    pass
