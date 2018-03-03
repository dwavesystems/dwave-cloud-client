from setuptools import setup, find_packages

install_requires = ['requests>=2.18', 'six>=1.10']

extras_require = {
    'test': ['requests_mock', 'mock', 'numpy'],
    ':python_version == "2.7"': ['futures']
}

# Only include packages under the 'dwave' namespace. Do not include tests,
# benchmarks, etc.
packages = [package for package in find_packages() if package.startswith('dwave')]

setup(
    name='dwave-cloud-client',
    version='0.3.0.dev1',
    description='A minimal client for interacting with SAPI servers.',
    url='https://github.com/dwavesystems/dwave-cloud-client',
    packages=packages,
    install_requires=install_requires,
    extras_require=extras_require,
    zip_safe=False
)
