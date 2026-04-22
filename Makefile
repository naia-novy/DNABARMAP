.PHONY: env env-core check-medaka clean

CONDA ?= $(shell if command -v mamba >/dev/null 2>&1; then echo mamba; elif command -v micromamba >/dev/null 2>&1; then echo micromamba; else echo conda; fi)
ENV_NAME ?= dnabarmap
COMMON_CHANNELS = --override-channels --strict-channel-priority -c conda-forge -c nanoporetech -c bioconda
CORE_PACKAGES = python=3.10 pip pbsim3 mmseqs2 minimap2 samtools racon
PY_PACKAGES = biopython matplotlib=3.8.0 numpy=1.25 pandas=2.3.1 regex=2023.10.3 scipy=1.15 seaborn=0.13.2 cffi=1.15.0 setuptools=80
MEDAKA_PACKAGES = medaka pyspoa
PIP_INSTALL = pip install -e . --no-deps --no-build-isolation

# Create and setup the conda environment
env:
	@echo "Creating $(ENV_NAME) environment with $(CONDA)..."
	$(CONDA) create -n $(ENV_NAME) -y $(COMMON_CHANNELS) $(CORE_PACKAGES) $(PY_PACKAGES) $(MEDAKA_PACKAGES)
	$(CONDA) run -n $(ENV_NAME) $(PIP_INSTALL)
	$(MAKE) check-medaka ENV_NAME=$(ENV_NAME) CONDA=$(CONDA)

# Faster setup that skips Medaka entirely.
env-core:
	@echo "Creating $(ENV_NAME) core environment with $(CONDA)..."
	$(CONDA) create -n $(ENV_NAME) -y $(COMMON_CHANNELS) $(CORE_PACKAGES) $(PY_PACKAGES)
	$(CONDA) run -n $(ENV_NAME) $(PIP_INSTALL)
#	$(CONDA) install -n $(ENV_NAME) -y -c conda-forge libgcc-ng=12
#	$(CONDA) install -n $(ENV_NAME) -y -c conda-forge cupy cudatoolkit

check-medaka:
	@echo "Checking Medaka/spoa inside $(ENV_NAME)..."
	$(CONDA) run -n $(ENV_NAME) python -c "import sys, spoa; print(sys.executable); print(spoa.__file__)"
	$(CONDA) run -n $(ENV_NAME) medaka_consensus --help >/dev/null

# Remove the environment
clean:
	$(CONDA) env remove -n $(ENV_NAME) -y
