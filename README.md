DNABARMAP
=========

DNABARMAP is a pipeline to determine barcode--variant mappings of degenerate barcodes from noisy sequencing data.

* * * * *

Installation
------------

### Option A --- Clone with Git (recommended)

bash

CopyEdit

`git clone https://github.com/naia-novy/DNABARMAP.git
cd DNABARMAP
make   # environment setup, initialization, and (optional) downloads
conda activate dnabarmap
dnabarmap --help   # confirm installation`

### Option B --- No Git (download ZIP)

1.  Download the ZIP from: `https://github.com/naia-novy/DNABARMAP` → **Code → Download ZIP**

2.  Extract and open a terminal in the extracted folder, then run:

bash

CopyEdit

`make
conda activate dnabarmap
dnabarmap --help`

**Notes**

-   `make.sh` should create a conda environment named `dnabarmap` (or instruct users how to create one) and run `pip install -e .` to install the package in editable mode.

-   Consider including `environment.yml` so users can run `conda env create -f environment.yml` as an alternative.

* * * * *

Quick usage examples
--------------------

### Generate a barcode template

Generate a barcode template (length 60, max homopolymer length 3):

bash

CopyEdit

`generate_barcode_template --barcode_len 60 --max_homopolymer_len 3`

Example output:

nginx

CopyEdit

`HVWBWRHSRBWRKARHBWSSYKVYMKYRMDSHGBVMRKRYWSSWMWYYSRDWKSYMRYVW`

### Generate synthetic data (validation)

Create synthetic reads using a barcode template:

bash

CopyEdit

`generate_syndata\
  --barcode_template HVWBWRHSRBWRKARHBWSSYKVYMKYRMDSHGBVMRKRYWSSWMWYYSRDWKSYMRYVW\
  --fn test\
  --duplication_rate 50\
  --barcodes_per_variant 20\
  --num_variants 100`

> Note: `generate_syndata` also accepts motif-style templates (e.g., `"2323232"`) for specifying allowed nucleotides per position.

### Run DNABARMAP (map barcodes → variants)

Use either synthetic or real FASTQ input:

bash

CopyEdit

`dnabarmap\
  --fastq_fn test.fastq\
  --barcode_template HVWBWRHSRBWRKARHBWSSYKVYMKYRMDSHGBVMRKRYWSSWMWYYSRDWKSYMRYVW\
  --left_coding_flank CTGCTATCGT\
  --right_coding_flank TATCAGAGTC\
  --max_len 150\
  --buffer 30\
  --min_sequences 25\
  --save_intermediate_files\
  --synthetic_data_available`

* * * * *

CLI help
--------

For full options and flags:

bash

CopyEdit

`dnabarmap --help
generate_syndata --help
generate_barcode_template --help`

* * * * *

Data & large files
------------------

-   **Do not commit large data files** to the repository. Put large FASTQ / PKL files in a separate data store (Cloud, S3, Zenodo, institutional storage) and provide a download script or instructions.

-   Add large-data paths to `.gitignore` (example entries: `syndata/`, `DNABARMAP_outputs/`).

-   If you need versioned large-file storage in the repo, consider [Git LFS](https://git-lfs.github.com/) and document that requirement in the README.

* * * * *

Development
-----------

-   Install in editable / dev mode (after cloning):

bash

CopyEdit

`pip install -e .`

-   To recreate environment from YAML (if provided):

bash

CopyEdit

`conda env create -f environment.yml
conda activate dnabarmap
pip install -e .`

-   Tests (if provided) can be run with:

bash

CopyEdit

`pytest`

* * * * *

Troubleshooting
---------------

-   If `bash make.sh` fails, ensure Conda/Miniconda is installed and on your `PATH`.

-   If `dnabarmap` command is not found after installation, make sure the active environment is `dnabarmap` and `pip install -e .` completed without errors.

-   If installation fails due to non-Python system tools (e.g., `minimap2`, `racon`), install those via `conda` (bioconda/conda-forge) or system package manager, or document them in `make.sh`.

* * * * *

Citation
--------

If you use DNABARMAP in published work, please cite:

> *Paper to be published.*

Also acknowledge the following third-party tools used by DNABARMAP:

-   `vsearch`

-   `minimap2`

-   `racon`

-   `pbsim3`

(Replace with exact citation/DOI lines once available.)

* * * * *

License
-------

Add your license here (e.g., MIT, Apache-2.0). Include a `LICENSE` file in the repo.

* * * * *

Contact & Contributions
-----------------------

-   Contributions: open issues and pull requests on the GitHub repository. Please follow repository `CONTRIBUTING` guidelines if present.

-   For questions, open an issue or contact the maintainer (add preferred contact/email).