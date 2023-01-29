#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['Click>=7.0', ]

test_requirements = [ ]

setup(
    author="Giuseppe Attanasio",
    author_email='giuseppeattanasio6@gmail.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Simple evaluation of classification confidence intervals",
    entry_points={
        'console_scripts': [
            'confidence_intervals=confidence_intervals.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='confidence_intervals',
    name='confidence_intervals',
    packages=find_packages(include=['confidence_intervals', 'confidence_intervals.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/g8a9/confidence_intervals',
    version='0.1.0',
    zip_safe=False,
)
