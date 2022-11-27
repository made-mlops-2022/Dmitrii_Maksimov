import pytest
from app import load_model


@pytest.fixture(scope="session", autouse=True)
def load_model_for_tests():
    load_model()
