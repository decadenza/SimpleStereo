#!/usr/bin/env python3
from setuptools import setup, Extension

class get_numpy_include(object):
    """ Defer numpy.get_include() until after numpy is installed. """

    def __str__(self):
        import numpy
        return numpy.get_include()

if __name__ == '__main__':
         
    setup(
        name = "SimpleStereo",
        version = "1.0.9",
        author = "decadenza",
        description = "Stereo vision made simple",
        long_description = open("README.md").read(),
        long_description_content_type = "text/markdown",
        url = "https://github.com/decadenza/SimpleStereo",
        classifiers = [
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
            "Operating System :: OS Independent",
        ],
        setup_requires = [
                        'numpy>=1.19, <2',
        ],
        install_requires = [
                             'numpy>=1.19, <2',
                             'opencv-contrib-python>=4.5',
                             'scipy>=1.4',
                             'matplotlib>=3',
        ],
        ext_modules = [
                        Extension(
                                'simplestereo._passive', 
                                ['simplestereo/_passive.cpp'],
                                include_dirs=[get_numpy_include()],
                                extra_compile_args=["-std=c++11"]
                                ),
                        Extension(
                                    'simplestereo._unwrapping',
                                    ['simplestereo/_unwrapping.cpp'],
                                    include_dirs=[get_numpy_include()],
                                    extra_compile_args=["-std=c++11"]
                                    ),
        ],
    )
