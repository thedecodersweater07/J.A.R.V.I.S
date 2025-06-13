"""Configuration file for pytest."""
import os
import sys
from pathlib import Path
from typing import Generator, Any, Dict

import pytest

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Fixtures

@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Return the path to the test data directory."""
    return Path(__file__).parent / "test_data"

@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set up the test environment."""
    # Set environment variables for testing
    monkeypatch.setenv("ENV", "test")
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).parent.parent))

@pytest.fixture
def mock_model() -> Dict[str, Any]:
    """Return a mock model configuration for testing."""
    return {
        "model_type": "test",
        "model_name": "test-model",
        "config": {
            "hidden_size": 128,
            "num_attention_heads": 4,
            "num_hidden_layers": 2,
        },
    }

# Skip tests that require network access
skip_network = pytest.mark.skipif(
    not os.getenv("TEST_NETWORK"),
    reason="Network tests disabled. Set TEST_NETWORK=1 to enable.",
)

# Skip tests that require specific hardware
skip_if_no_gpu = pytest.mark.skipif(
    not os.getenv("TEST_GPU"),
    reason="GPU tests disabled. Set TEST_GPU=1 to enable.",
)

# Add custom markers
def pytest_configure(config: Any) -> None:
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with '-m "
        "not slow')",
    )
    config.addinivalue_line(
        "markers",
        "integration: marks integration tests (deselect with '-m "
        "not integration')",
    )
    config.addinivalue_line(
        "markers",
        "unit: marks unit tests (deselect with '-m not unit')",
    )

# Add command line options
def pytest_addoption(parser: Any) -> None:
    """Add custom command line options."""
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="run slow tests",
    )

# Configure test collection
def pytest_collection_modifyitems(config: Any, items: Any) -> None:
    """Modify test collection based on command line options."""
    if not config.getoption("--runslow"):
        skip_slow = pytest.mark.skip(reason="need --runslow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
