.PHONY: all deps install

all: deps install

# Run your python package installation
install:
	@echo "Installing Python package..."
	conda run -n dnabarmap pip install -e .