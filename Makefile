.PHONY: all deps install

all: deps install

# Install system/bioinformatics tools (example with conda)
deps:
	@echo "Installing system dependencies..."
	conda install -n dnabarmap -y -c bioconda minimap2 racon pbsim3 vsearch

# Run your python package installation
install:
	@echo "Installing Python package..."
	conda run -n dnabarmap pip install -e .