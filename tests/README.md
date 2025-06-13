# J.A.R.V.I.S. Model Tests

This directory contains integration tests for the J.A.R.V.I.S. AI models.

## Running Tests

### Prerequisites

1. Python 3.8 or higher
2. pip (Python package manager)

### Setup

1. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install test dependencies:
   ```bash
   pip install -r requirements-test.txt
   ```

3. (Optional) For full test coverage, install the core dependencies by uncommenting them in `requirements-test.txt` and running:
   ```bash
   pip install -r requirements-test.txt
   ```

### Running Tests

Run all tests:
```bash
pytest tests/
```

Run tests with coverage report:
```bash
pytest --cov=models tests/
```

Run a specific test file:
```bash
pytest tests/test_jarvis_models.py -v
```

Run a specific test case:
```bash
pytest tests/test_jarvis_models.py::TestJarvisModel -v
```

## Test Structure

- `test_jarvis_models.py`: Contains integration tests for all J.A.R.V.I.S. model classes
  - `TestJarvisModel`: Tests for the base model class
  - `TestJarvisLanguageModel`: Tests for language model functionality
  - `TestJarvisNLPModel`: Tests for NLP model functionality
  - `TestJarvisMLModel`: Tests for ML model functionality
  - `TestJarvisModelManager`: Tests for model management
  - `TestCreateJarvisModel`: Tests for the model factory function

## Writing Tests

When adding new functionality, please add corresponding tests. Follow these guidelines:

1. Test one piece of functionality per test method
2. Use descriptive test method names (e.g., `test_model_initialization`)
3. Use mocks to isolate tests from external dependencies
4. Include both positive and negative test cases
5. Keep tests focused and fast

## Troubleshooting

- **Import errors**: Make sure all dependencies are installed and your Python path is set correctly
- **Skipped tests**: Some tests may be skipped if required dependencies are not installed
- **Test failures**: Check the test output for details and ensure your environment matches the expected configuration
