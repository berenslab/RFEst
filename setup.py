from setuptools import setup

with open('requirements.txt') as f:
    required = f.read().splitlines()

exec(open('rfest/version.py').read())

setup(
      name='RFEst',
      version=__version__,
      description='Python 3 toolbox for receptive field estimation',
      author='Ziwei Huang, Jonathan Oesterle',
      author_email='huang-ziwei@outlook.com',
      install_requires=required,
)
