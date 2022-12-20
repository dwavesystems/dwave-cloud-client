import os
from io import open
from setuptools import setup, find_packages


# Load package info, without importing the package
basedir = os.path.dirname(os.path.abspath(__file__))
package_info_path = os.path.join(basedir, "dwave", "cloud", "package_info.py")
package_info = {}
with open(package_info_path, encoding='utf-8') as f:
    exec(f.read(), package_info)

python_requires = '>=3.7'

# Package requirements, minimal pinning
install_requires = ['requests[socks]>=2.18', 'pydantic>=1.7.3', 'homebase>=1.0',
                    'click>=7.0', 'python-dateutil>=2.7', 'plucky>=0.4.3',
                    'diskcache>=5.2.1']

# Package extras requirements
extras_require = {
    'test': ['requests_mock', 'mock', 'numpy', 'coverage'],

    # bqm support
    'bqm': ['dimod>=0.10.5,<0.13,!=0.11.4', 'numpy>=1.17.3'],
    'bqm-legacy': ['dimod~=0.9.0', 'numpy>=1.16,<1.24'],

    # dqm support
    'dqm': ['dimod>=0.10.5,<0.13,!=0.11.4', 'numpy>=1.17.3'],
    'dqm-legacy': ['dimod~=0.9.7,<0.10', 'numpy>=1.16,<1.24'],

    # cqm support
    'cqm': ['dimod>=0.10.5,<0.13,!=0.11.4', 'numpy>=1.17.3'],

    # testing mocks
    'mocks': ['dwave-networkx>=0.8.9'],
}

# Packages provided. Only include packages under the 'dwave' namespace.
# Do not include tests, benchmarks, etc.
packages = [package for package in find_packages() if package.startswith('dwave')]

classifiers = [
    'License :: OSI Approved :: Apache Software License',
    'Operating System :: OS Independent',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
    'Programming Language :: Python :: 3.11',
]

setup(
    name=package_info['__packagename__'],
    version=package_info['__version__'],
    author=package_info['__author__'],
    description=package_info['__description__'],
    long_description=open('README.rst', encoding='utf-8').read(),
    url=package_info['__url__'],
    license=package_info['__license__'],
    license_files=["LICENSE"],
    packages=packages,
    python_requires=python_requires,
    install_requires=install_requires,
    extras_require=extras_require,
    classifiers=classifiers,
    zip_safe=False,
    entry_points={
        'console_scripts': [
            'dwave = dwave.cloud.cli:cli'
        ]
    }
)
