.PHONY: env clean

# Create and setup the conda environment
env:
	@echo "Creating dnabarmap environment..."
	conda create -n dnabarmap python=3.10 -y
	conda run -n dnabarmap pip install -e .
	conda install -n dnabarmap -y -c conda-forge libgcc-ng=12
	conda install -n dnabarmap -y -c conda-forge -c bioconda pbsim3 vsearch abpoa racon minimap2
    conda install -n dnabarmap -y -c conda-forge cupy cudatoolkit=12.2

# Remove the environment
clean:
	conda env remove -n dnabarmap -y