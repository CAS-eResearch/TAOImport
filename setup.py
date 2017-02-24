#!/usr/bin/env python

import os
from setuptools import setup, find_packages

setup(
    name='taoconvert',
    version='0.2',
    author='Luke Hodkinson',
    author_email='furious.luke@gmail.com',
    maintainer='Manodeep Sinha',
    maintainer_email='manodeep@gmail.com',
    url='http://github.com/CAS-eResearch/TAO-ScienceModules',
    description='Convert semi-analytic datasets to TAO format.',
    long_description=open(os.path.join(os.path.dirname(__file__), 'README.md')).read(),
    classifiers = [
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
    ],
    license='BSD',

    packages=find_packages(),
    scripts=['tao/scripts/taoconvert'],
    include_package_data=True,
    install_requires=['setuptools', 'tqdm', 'h5py','mpi4py'],
    zip_safe=False,
)
