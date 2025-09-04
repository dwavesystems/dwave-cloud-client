import os
from io import open
from setuptools import setup, find_namespace_packages


# Load package info, without importing the package
basedir = os.path.dirname(os.path.abspath(__file__))
package_info_path = os.path.join(basedir, "dwave", "cloud", "package_info.py")
package_info = {}
with open(package_info_path, encoding='utf-8') as f:
    exec(f.read(), package_info)

python_requires = '>=3.9'

# Package requirements, minimal pinning
install_requires = ['requests[socks]>=2.25,<3', 'urllib3>=1.26,<3',
                    'pydantic>=2,<3', 'homebase>=1.0,<2',
                    'click>=7.0,<9', 'python-dateutil>=2.7,<3', 'plucky>=0.4.3,<0.5',
                    'diskcache>=5.2.1,<6', 'packaging>=19', 'werkzeug>=2.2,<4',
                    'typing-extensions>=4.5.0,<5', 'authlib>=1.2,<2',
                    'importlib_metadata>=5.0.0',    # can be dropped in py312+
                    'orjson>=3.10', 'http-sf>=1.0.4',
                    ]

# Package extras requirements
extras_require = {
    'test': ['requests_mock', 'mock', 'numpy', 'coverage'],

    # bqm support
    'bqm': ['dimod>=0.10.5,<0.13,!=0.11.4', 'numpy>=1.17.3'],

    # dqm support
    'dqm': ['dimod>=0.10.5,<0.13,!=0.11.4', 'numpy>=1.17.3'],

    # cqm support
    'cqm': ['dimod>=0.10.5,<0.13,!=0.11.4', 'numpy>=1.17.3'],

    # nlm support
    'nlm': ['dwave-optimization>=0.1.0,<0.5', 'numpy>=1.20.0'],

    # testing mocks
    'mocks': ['dwave-networkx>=0.8.10'],
}

# Packages provided. Only include packages under the 'dwave' namespace.
# Do not include tests, benchmarks, etc.
packages = find_namespace_packages(include=['dwave.*'])

classifiers = [
    'License :: OSI Approved :: Apache Software License',
    'Operating System :: OS Independent',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
    'Programming Language :: Python :: 3.11',
    'Programming Language :: Python :: 3.12',
    'Programming Language :: Python :: 3.13',
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
