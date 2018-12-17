import os
from io import open
from setuptools import setup, find_packages


# Load package info, without importing the package
basedir = os.path.dirname(os.path.abspath(__file__))
package_info_path = os.path.join(basedir, "dwave", "cloud", "package_info.py")
package_info = {}
try:
    with open(package_info_path, encoding='utf-8') as f:
        exec(f.read(), package_info)
except SyntaxError:
    execfile(package_info_path, package_info)


# Package requirements, minimal pinning
install_requires = ['requests[socks]>=2.18', 'six>=1.10', 'homebase>=1.0',
                    'click>=6.7', 'python-dateutil>=2.7']

# Package extras requirements
extras_require = {
    'test': ['requests_mock', 'mock', 'numpy', 'coverage', 'plucky'],

    # python2 backports
    ':python_version == "2.7"': ['futures', 'configparser']
}

# Packages provided. Only include packages under the 'dwave' namespace.
# Do not include tests, benchmarks, etc.
packages = [package for package in find_packages() if package.startswith('dwave')]

setup(
    name=package_info['__packagename__'],
    version=package_info['__version__'],
    author=package_info['__author__'],
    description=package_info['__description__'],
    long_description=open('README.rst', encoding='utf-8').read(),
    url=package_info['__url__'],
    license=package_info['__license__'],
    packages=packages,
    install_requires=install_requires,
    extras_require=extras_require,
    zip_safe=False,
    entry_points={
        'console_scripts': [
            'dwave = dwave.cloud.cli:cli'
        ]
    }
)
