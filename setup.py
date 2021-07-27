#!/usr/bin/env python
import os
import subprocess
import time
from setuptools import find_packages, setup


def long_description():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content

if __name__ == "__main__":
    setup(
        name='QuanTransformer',
        setup_requires=['setuptools_scm'],
         use_scm_version={
        "root": "..",
        "relative_to": __file__,
    },
        description='quant operators of different quantization methods. ',
        long_description=long_description(),
        packages=find_packages(include=(('QuanTransformer'))),
        include_package_data=True,
        classifiers=[
            'Development Status :: 3 - Alpha',
            'License :: OSI Approved :: Apache Software License',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 3',
        ],
        license='Apache License 2.0',
        zip_safe=False)