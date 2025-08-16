#!/bin/bash
conda init
echo "Creating dnabarmap environment..."
conda create -n dnabarmap python=3.10 -y
conda activate dnabarmap
conda install -n dnabarmap -y -c bioconda pbsim3 vsearch pyabpoa
conda install -n dnabarmap -y -c nanoporetech medaka
conda run -n dnabarmap pip install -e .
conda run -n dnabarmap pip install cuda-cuda12x && exit 0