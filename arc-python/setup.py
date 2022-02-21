from setuptools import find_packages, setup

setup(
    name='arclib',
    packages=find_packages(include=['arclib', 'cloudpickle']),
    version='0.1.0',
    description='Library for data streaming',
    author='Klas Segeljakt, Frej Drejhammar',
    license='MIT',
    install_requires=['cloudpickle'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests',
)
