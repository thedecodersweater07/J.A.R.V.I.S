from setuptools import setup, find_packages

setup(
    name="jarvis",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'torch>=2.0.0',
        'transformers>=4.30.0',
        'numpy>=1.24.0',
        'pandas>=2.0.0',
        'PyYAML>=6.0',
        'tqdm>=4.65.0',
        'psutil>=5.9.0',
    ],
    python_requires='>=3.8',
    package_data={
        '': ['*.yaml', '*.json', '*.txt'],
    },
    include_package_data=True,
)
