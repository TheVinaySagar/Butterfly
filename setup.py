from setuptools import setup, find_packages
import os

# Directory where setup.py is located
here = os.path.abspath(os.path.dirname(__file__))

# Load dependencies from requirements.txt
try:
    with open(os.path.join(here, "requirements.txt"), encoding="utf-8") as f:
        REQUIRED = f.read().splitlines()
except:
    REQUIRED = []

setup(
    name='Butterfly_Classification',
    version='0.1.0',
    description='Library for butterfly image classification',
    author='Vinay Sagar',
    author_email='vinaysagar4445@gmail.com',
    url='https://github.com/TheVinaySagar/Butterfly_Classification',
    license='MIT',
    install_requires=REQUIRED,
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],
    packages=find_packages(exclude=("examples", "app", "data", "docker", "tests")),
)
