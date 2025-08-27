DNABARMAP
=========

DNABARMAP is a pipeline to determine barcode--variant mappings of degenerate barcodes from high error sequencing data, such as nanopore sequencing.

* * * * *

Installation
------------

### Option A --- Clone with Git (recommended)



`git clone https://github.com/naia-novy/DNABARMAP.git`

`cd DNABARMAP`

`make`# environment setup, initialization, and (optional) downloads

`conda activate dnabarmap`

`dnabarmap --help`   # confirm installation`



### Option B --- No Git (download ZIP)

1.  Download the ZIP from: `https://github.com/naia-novy/DNABARMAP` → **Code → Download ZIP**

2.  Extract and open a terminal in the extracted folder, then run:


`make`

`conda activate dnabarmap`

`dnabarmap --help`


Quick usage examples
--------------------

### Generate a barcode template

Generate a barcode template (length 60, max homopolymer length 3):



`generate_barcode_template --barcode_len 60 --max_homopolymer_len 3`

Example output:

`HVWBWRHSRBWRKARHBWSSYKVYMKYRMDSHGBVMRKRYWSSWMWYYSRDWKSYMRYVW`

### Generate synthetic data (validation)

Create synthetic reads using a barcode template:

`generate_syndata
  --barcode_template HVWBWRHSRBWRKARHBWSSYKVYMKYRMDSHGBVMRKRYWSSWMWYYSRDWKSYMRYVW
  --fn test
  --duplication_rate 50
  --barcodes_per_variant 20
  --num_variants 10`

### Run DNABARMAP (map barcodes → variants)

Use either synthetic or real FASTQ input:

`dnabarmap
  --fastq_fn test.fastq
  --barcode_template HVWBWRHSRBWRKARHBWSSYKVYMKYRMDSHGBVMRKRYWSSWMWYYSRDWKSYMRYVW\
  --left_coding_flank GCTATCGT 
  --right_coding_flank TATCAGAG
  --buffer 30
  --min_sequences 25
  --save_intermediate_files
  --synthetic_data_available`

Flanks must be adapted according to your construct. These flanks correspond to the synthetically generated constructs.
* * * * *

CLI help
--------

For full options and flags:

bash

CopyEdit

`dnabarmap --help
generate_syndata --help
generate_barcode_template --help`


Important flags:

--barcode_template

Not all degenerate barcode sequences work well. We suggest using one of our validated barcodes or 
                    generating and validating a new barcode with generate_barcode_template and synthetic data. Barcode
                    alignment improves if 4-8 bp constant regions flanking the degenerate sequence are included.

--buffer

How many bases inside the read you expect the barcode to start

--min_sequences

Ignore clusters with less than this number of sequences



Citation
--------
If you use DNABARMAP in published work, please cite:

`Paper to be published`

Also acknowledge the following third-party tools used by DNABARMAP:

vsearch:
Rognes T, Flouri T, Nichols B, Quince C, Mahé F. (2016) VSEARCH: a versatile open source tool for metagenomics. PeerJ 4:e2584. doi: 10.7717/peerj.2584

minimap2:
Li, H. (2018). Minimap2: pairwise alignment for nucleotide sequences. Bioinformatics, 34:3094-3100. doi:10.1093/bioinformatics/bty191

racon:
Vaser R, Sović I, Nagarajan N, Šikić M. Fast and accurate de novo genome assembly from long uncorrected reads. Genome Res. 2017 May;27(5):737-746. doi: 10.1101/gr.214270.116. Epub 2017 Jan 18. PMID: 28100585; PMCID: PMC5411768.

apboa:
Yan Gao, Yongzhuang Liu, Yanmei Ma, Bo Liu, Yadong Wang, Yi Xing, abPOA: an SIMD-based C library for fast partial order alignment using adaptive band, Bioinformatics, Volume 37, Issue 15, August 2021, Pages 2209–2211, https://doi.org/10.1093/bioinformatics/btaa963

pbsim3:
Yukiteru Ono, Michiaki Hamada, Kiyoshi Asai, PBSIM3: a simulator for all types of PacBio and ONT long reads, NAR Genomics and Bioinformatics, Volume 4, Issue 4, December 2022, lqac092, https://doi.org/10.1093/nargab/lqac092