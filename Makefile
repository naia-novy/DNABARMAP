.PHONY: env clean

# Create and setup the conda environment
env:
	@echo "Creating dnabarmap environment..."
	conda create -n dnabarmap python=3.10 -y
	conda install -n dnabarmap -y -c bioconda pbsim3 vsearch abpoa racon minimap2
	conda run -n dnabarmap pip install -e .
	conda run -n dnabarmap pip install cuda-cuda12x || true

# Remove the environment
clean:
	conda env remove -n dnabarmap -y