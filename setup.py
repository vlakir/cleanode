from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = ["numpy>=1.21.2", "funnydeco>=0.1.5", "matplotlib>=3.4.3", "scipy>=1.7.1", "PyQt5>=5.15.4"]

setup(
    name="cleanode",
    version="0.1.2",
    author="Vladimir Kirievskiy",
    author_email="vlakir1234@gmail.com",
    description="Example using an embedded solver",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/vlakir/cleanode.git",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Topic :: Scientific/Engineering"
    ],
)
