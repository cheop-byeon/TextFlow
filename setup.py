from setuptools import setup, find_packages

with open("README.md") as readme_file:
    readme = readme_file.read()

with open("requirements.txt") as reqs_file:
    requirements = reqs_file.read().split("\n")

setup(
    name="ProtoSpec",
    python_requires='>=3.7',
    description="A framework for the finetuning, generation evaluation of TextFlow.",
    long_description=readme,
    license="Apache 2.0",
    packages=find_packages() ,
    install_requires=requirements,
)
