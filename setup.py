from setuptools import setup

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
      name='RFEst',
      version='2.1.0',
      description='Python 3 toolbox for receptive field estimation',
      author='Ziwei Huang, Jonathan Oesterle',
      author_email='huang-ziwei@outlook.com',
      install_requires=required,
)
