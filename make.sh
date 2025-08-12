#!/bin/bash
conda init
echo "Creating dnabarmap environment..."
conda create -n dnabarmap python=3.11 -y
pip install cupy-cuda12x && exit 0
conda activate dnabarmap
make