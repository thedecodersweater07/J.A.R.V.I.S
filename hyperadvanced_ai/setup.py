"""
Setup script for the hyperadvanced_ai package.
"""

from setuptools import setup, find_packages

# Read requirements from requirements.txt
with open('requirements.txt', 'r', encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip()]

# Read long description from README.md
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="hyperadvanced_ai",
    version="0.1.0",
    description="Hyperadvanced AI components for JARVIS",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Nova Industrie AI Team",
    author_email="ai@novaindustrie.com",
    url="https://github.com/novaindustrie/hyperadvanced_ai",
    packages=find_packages(include=['hyperadvanced_ai', 'hyperadvanced_ai.*']),
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.0',
            'mypy>=0.910',
            'flake8>=3.9',
            'black>=21.7b0',
            'isort>=5.9.0',
        ],
        'nlp': [
            'nltk>=3.8.1',
            'spacy>=3.5.0',
            'transformers>=4.25.0',
            'torch>=1.9.0',
        ],
        'vision': [
            'opencv-python>=4.5.0',
            'Pillow>=8.0.0',
            'numpy>=1.20.0',
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='ai nlp machine-learning jarvis',
    project_urls={
        'Bug Reports': 'https://github.com/novaindustrie/hyperadvanced_ai/issues',
        'Source': 'https://github.com/novaindustrie/hyperadvanced_ai',
    },
    include_package_data=True,
    zip_safe=False,
)
