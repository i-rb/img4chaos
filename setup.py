from setuptools import setup

setup(
    name='img4chaos',
    url='https://github.com/i-rb/img4chaos',
    author='Ivan Rendo Barreiro',
    author_email='irendo@yahoo.es',
    packages=['img4chaos'],
    install_requires=['numpy','pandas'], #more
    version='0.1',
    license='MIT',
    description='A package for detecting chaos',
    long_description=open('README.md').read(),
)
