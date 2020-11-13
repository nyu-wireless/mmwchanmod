from setuptools import setup, find_packages


setup(name = 'mmwchanmod',
      version = '0.0',
      description = 'Package for modeling and simulation of mmWave wireless channels',
      author = 'Sundeep Rangan, William Xia, NYU',
      install_requires = ['numpy','scipy','matplotlib','tensorflow','scikit-learn',],
      author_email = 'srangan@nyu.edu',
      license = 'MIT',
      packages=find_packages(),      
      zip_safe = False)