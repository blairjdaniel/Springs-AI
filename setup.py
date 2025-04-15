from setuptools import setup, find_packages

setup(
    name="AI-Assistant-Springs",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "transformers==4.29.0",
        "torch==2.0.0",
        "datasets==2.10.0",
        "fastapi==0.95.1",
        "uvicorn==0.22.0",
        "pyyaml==6.0",
        "python-dotenv==1.0.0",
        "pandas==1.5.3",
        "numpy==1.23.5",
    ],
    entry_points={
        "console_scripts": [
            "start-ai-assistant=llm.main:main",
        ],
    },
    author="Your Name",
    description="An AI Assistant project using multiple LLM providers",
)