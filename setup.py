from setuptools import setup

install_requires = ['requests>=2.18', 'six>=1.10']
extras_require = {'test': ['requests_mock', 'mock', 'numpy']}

setup(
    name='dwave_micro_client',
    py_modules=['dwave_micro_client'],
    version='0.2.2',
    description='A minimal client for interacting with SAPI servers.',
    url='https://github.com/dwavesystems/dwave_micro_client',
    install_requires=install_requires,
    extras_require=extras_require
)
