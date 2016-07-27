#!/usr/bin/env python

import os
from setuptools import setup, find_packages

setup(
    name='taoconvert',
    version='0.1',
    author='Luke Hodkinson',
    author_email='furious.luke@gmail.com',
    maintainer='Luke Hodkinson',
    maintainer_email='furious.luke@gmail.com',
    url='http://github.com/CAS-eResearch/TAO-ScienceModules',
    description='Convert semi-analytic datasets to TAO format.',
    long_description=open(os.path.join(os.path.dirname(__file__), 'README.md')).read(),
    classifiers = [
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
    ],
    license='BSD',

    packages=find_packages(),
    scripts=['tao/scripts/taoconvert'],
    include_package_data=True,
    install_requires=['setuptools','progressbar2'],
    zip_safe=False,
)
