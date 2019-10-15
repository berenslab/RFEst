from setuptools import setup, find_packages

setup(name='RFEst',
      version='0.0.2',
      description='Python 3 tool for receptive field estimation',
      author='Ziwei Huang',
      author_email='huang-ziwei@outlook.com',
      packages=find_packages(),
      install_requires=[
        "numpy>=1.13.1",
        "scipy",
        "matplotlib",
        "sklearn",
        "jax",
        "jaxlib"
      ],
     )
