from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(name='RFEst',
      version='2.0.0',
      description='Python 3 toolbox for receptive field estimation',
      author='Ziwei Huang',
      author_email='huang-ziwei@outlook.com',
      install_requires=required,
     )
