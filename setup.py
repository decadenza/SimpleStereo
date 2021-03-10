#!/usr/bin/env python3

from setuptools import setup, Extension


if __name__ == '__main__':
         
    setup(name='SimpleStereo',
          version = "0.9.0",
          description = "Stereo vision made simple",
          author = "Pasquale Lafiosca",
          author_email = "decadenza@protonmail.com",
          url = "http://github.com/decadenza",
          packages = ['simplestereo'],
          ext_modules = [Extension('simplestereo._passive', ['./simplestereo/_passive.cpp']),
                        Extension('simplestereo._unwrapping', ['./simplestereo/_unwrapping.cpp'])],
          install_requires = [
                             'numpy>=1.19',
                             'opencv-contrib-python>=4.5',
                             'scipy>=1.4',
                             'matplotlib>=3',
                             ],
         )
