from setuptools import setup


setup(name = 'mmwchanmod',
      version = '0.0',
      description = 'Package for modeling and simulation of mmWave wireless channel',
      author = 'GAMP Team',
      install_requires = ['numpy','scipy','matplotlib','tensorflow','scikit-learn',],
      author_email = 'srangan@nyu.edu',
      license = 'MIT',
      packages = ['mmwchanmod'],
      zip_safe = False)