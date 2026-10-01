from setuptools import setup, find_packages

setup(
    name="sonon",
    version="0.1.0",
    packages=find_packages(),
    package_data={
        "sonon": ["*.so", "*.dylib", "*.dll"],
    },
    include_package_data=True,
    zip_safe=False,
)
