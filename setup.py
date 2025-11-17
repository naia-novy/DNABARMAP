from setuptools import setup, find_packages

def parse_requirements(filename='requirements.txt'):
    """Read requirements.txt and return a list of dependencies."""
    with open(filename, 'r') as f:
        lines = f.readlines()
    # Filter out empty lines and comments
    requirements = [line.strip() for line in lines if line.strip() and not line.startswith('#')]
    return requirements

setup(
    # Basic package info
    name='dnabarmap',
    version='0.1.0',
    description='Pipeline to determine barcode-variant mappings of degenerate barcodes from noisy sequencing data',
    long_description=open('README.md').read(),
    author='Naia Novy',
    url='https://github.com/naia-novy/DNABARMAP',

    # Package contents
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.10,<3.11",

    # Dependencies
    install_requires=parse_requirements(),
    entry_points={
        'console_scripts': [
            'dnabarmap = dnabarmap.run:cli',
            'align = dnabarmap.array_align:cli',
            'clustermap = dnabarmap.cluster_map:cli',
            'generate_barcode_template = dnabarmap.generate_barcode_template:cli',
            'generate_syndata = dnabarmap.generate_syndata:cli',
        ],
    },

    # Additional metadata
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries',
    ],

    # License info (make sure LICENSE file is included in your repo)
    license='MIT',
)
