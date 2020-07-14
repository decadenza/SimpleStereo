#!/usr/bin/env python3


from setuptools import setup, Extension

if __name__ == '__main__':
         
    setup(name='SimpleStereo',
          version='0.9',
          description='Stereo vision made simple',
          author='Pasquale Lafiosca',
          author_email='decadenza@protonmail.com',
          url='',
          packages=['simplestereo'],
          ext_modules=[Extension('simplestereo.passiveExt', ['./simplestereo/passiveExt.cpp'])],
          install_requires=[
                            'numpy>=1.18',
                            'opencv-python>=4.2, <4.3',
                            'scipy>=1.4',
                           ],
         )
