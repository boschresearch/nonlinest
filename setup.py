from setuptools import find_packages, setup

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="nonlinear_estimation_toolbox",
    version="0.0.0",
    packages=find_packages(),
    url="https://sourcecode.socialcoding.bosch.com/users/dog2rng/repos/nonlinear_estimation_toolbox",
    description="Python implementation of a nonlinear estimation toolbox inspired by the nonlinear estimation toolbox by Jannik Steinbring",
    install_requires=requirements,
    python_requires=">=3.10.0",
    package_data={},
)
