from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="chromalab",
    description="Library for cone fundamentals and printer math.",
    version="0.3",
    packages=find_packages(),
    package_data={
        "chromalab": ["cones/*.csv"],
    },
    include_package_data=True,
    install_requires=requirements,
)
