from setuptools import setup, find_packages

install_requires = ['requests>=2.18', 'six>=1.10', 'homebase>=1.0', 'click>=6.7']

extras_require = {
    'test': ['requests_mock', 'mock', 'numpy'],
    ':python_version == "2.7"': ['futures', 'configparser']
}

# Only include packages under the 'dwave' namespace. Do not include tests,
# benchmarks, etc.
packages = [package for package in find_packages() if package.startswith('dwave')]

setup(
    name='dwave-cloud-client',
    version='0.3.2',
    author='D-Wave Systems Inc.',
    description='A minimal client for interacting with D-Wave cloud resources',
    url='https://github.com/dwavesystems/dwave-cloud-client',
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
