from setuptools import setup, find_packages

setup(
    name="chromalab",
	version="0.1",
    packages=find_packages(),
	package_data={
		"chromalab": ["cones/*.csv"],
	},
	include_package_data=True,
)

