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
Barcode length greater than 50 is reccomended for better barcode clustering.

`generate_barcode_template --barcode_len 60 --max_homopolymer_len 3`

Example output:

`HVVKWRSHRVYVRSMWSRDMSRHMBRDMBRNYVRBMRYRVBMKVDMSRSMSRMSMRVYDV`

### Generate synthetic data (validation)

Create synthetic reads using a barcode template:

`generate_syndata
  --barcode_template HVVKWRSHRVYVRSMWSRDMSRHMBRDMBRNYVRBMRYRVBMKVDMSRSMSRMSMRVYDV
  --fn syndata/syndata
--left_coding_flank GCTATCGT 
--right_coding_flank TATCAGAG 
  --duplication_rate 50
  --barcodes_per_variant 5
  --num_variants 5`

### Run DNABARMAP (map barcodes → variants)

Use either synthetic or real FASTQ input:

` dnabarmap 
--input_fn syndata/syndata.pkl 
--mapping_fn mapped_barcodes.tsv 
--barcode_template HVVKWRSHRVYVRSMWSRDMSRHMBRDMBRNYVRBMRYRVBMKVDMSRSMSRMSMRVYDV 
--left_coding_flank GCTATCGT 
--right_coding_flank TATCAGAG 
--min_sequences 20 
--save_intermediate_files 
--synthetic_data_available`

Flanks must be adapted according to your construct. These flanks correspond to the synthetically generated constructs.
* * * * *

dnabarmap can also be run all at once or step-wise using the commands align, cluster, consensus, and map. Reccomended usage for full control and speed is to use commands sequentially

ex all at once:
dnabarmap

ex sequentially:


CLI help
--------

For full options and flags:

dnabarmap --help

align --help

cluster --help

consensus --help

map --help

generate_syndata --help

generate_barcode_template --help


Citation
--------
### If you use DNABARMAP in published work, please cite:

Paper to be published

### Also acknowledge the following tools used by DNABARMAP:

mmseqs2:
Steinegger, M., Söding, J. MMseqs2 enables sensitive protein sequence searching for the analysis of massive data sets. Nat Biotechnol 35, 1026–1028 (2017). https://doi.org/10.1038/nbt.3988

minimap2:
Li, H. (2018). Minimap2: pairwise alignment for nucleotide sequences. Bioinformatics, 34:3094-3100. doi:10.1093/bioinformatics/bty191

pbsim3:
Yukiteru Ono, Michiaki Hamada, Kiyoshi Asai, PBSIM3: a simulator for all types of PacBio and ONT long reads, NAR Genomics and Bioinformatics, Volume 4, Issue 4, December 2022, lqac092, https://doi.org/10.1093/nargab/lqac092

medaka:
https://github.com/nanoporetech/medaka