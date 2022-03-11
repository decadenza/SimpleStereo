#!/usr/bin/env python3
from setuptools import setup, Extension

class get_numpy_include(object):
    """Defer numpy.get_include() until after numpy is installed."""

    def __str__(self):
        import numpy
        return numpy.get_include()

if __name__ == '__main__':
         
    setup(packages = ['simplestereo'],
          setup_requires = [
                            'numpy>=1.19',
                            ],
          install_requires = [
                             'numpy>=1.19',
                             'opencv-contrib-python>=4.5, <=4.5.3 ',
                             'scipy>=1.4',
                             'matplotlib>=3',
                             ],
          ext_modules = [
                        Extension(
                                'simplestereo._passive', 
                                ['simplestereo/_passive.cpp'],
                                include_dirs=[get_numpy_include()]
                                ),
                        Extension(
                                 'simplestereo._unwrapping',
                                 ['simplestereo/_unwrapping.cpp'],
                                 include_dirs=[get_numpy_include()]
                                 ),
                        ],
          )
