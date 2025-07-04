from setuptools import setup, find_packages
from setuptools.command.develop import develop
from setuptools.command.install import install
from subprocess import check_call

def install_playwright_deps():
    check_call(["playwright", "install"])

class PostDevelopCommand(develop):
    def run(self):
        develop.run(self)
        install_playwright_deps()

class PostInstallCommand(install):
    def run(self):
        install.run(self)
        install_playwright_deps()

# Read requirements with error handling
def read_requirements():
    try:
        with open("requirements.txt", "r") as f:
            return [
                line.strip()
                for line in f
                if line.strip() and not line.startswith("#")
            ]
    except FileNotFoundError:
        return []

setup(
    name="paperpal",
    version="0.1.0",
    packages=find_packages(),
    install_requires=read_requirements(),
    cmdclass={
        'develop': PostDevelopCommand,
        'install': PostInstallCommand,
    },
)