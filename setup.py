from setuptools import setup

setup(
    name='dwave_micro_client',
    py_modules=['dwave_micro_client'],
    version='0.1',
    description='A minimal client for interacting with SAPI servers.',
    url='https://github.com/dwavesystems/dwave_micro_client',
    install_requires=['requests>=2.18', 'six>=1.10']
)
