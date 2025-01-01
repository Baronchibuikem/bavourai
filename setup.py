from setuptools import setup, find_packages

setup(
    name="bavour-ai",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "langchain",
        "qdrant-client",
        "numpy",
        "openai",
        "requests",
        "chromadb",
    ],
    description="AI package for managing prompts with multiple embedding providers.",
    author="Baron Chibuike",
    author_email="baronchibuike@gmail.com",
)
