from setuptools import setup, find_packages
import os

def get_requirements():
    with open('requirements.txt') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="jarvis",
    version="0.1.0",
    packages=find_packages(include=['llm', 'llm.processors', 'llm.processors.*']),
    install_requires=get_requirements(),
    python_requires='>=3.8',
    package_data={
        '': ['*.yaml', '*.json', '*.txt', '*.md'],
    },
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'jarvis=jarvis.cli:main',
        ],
    },
    # Download NLTK data during installation
    setup_requires=[
        'nltk>=3.6.0',
    ],
    # Add any additional metadata
    author="Your Name",
    author_email="your.email@example.com",
    description="JARVIS AI Assistant - A sophisticated AI assistant",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/yourusername/jarvis",
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
