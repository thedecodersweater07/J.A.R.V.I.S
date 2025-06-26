from setuptools import setup, find_packages

setup(
    name="bluetooth_health",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "bleak>=0.20.2",
        "fastapi>=0.95.0",
        "uvicorn>=0.21.1",
        "python-dotenv>=1.0.0",
    ],
    entry_points={
        "console_scripts": [
            "bluetooth-health=bluetooth_health.cli:main",
        ],
    },
)
