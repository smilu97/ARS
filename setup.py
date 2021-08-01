#!/usr/bin/env python

from distutils.core import setup

setup(name='ars',
      version='0.1',
      description='ARS Implementation',
      author='Kim YoungJin',
      author_email='smilup2244@gmail.com',
      url='https://github.com/smilu97/ars',
      install_requires=[
        'numpy',
        'gym',
        'ray'
      ],
      packages=['ars'],
     )