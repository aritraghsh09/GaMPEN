.PHONY: clean lint requirements reset-mlflow

PROJECT_NAME = ggt
PY = python

# Install dependencies with pip
requirements: clean
	$(PY) -m pip install -U pip setuptools wheel
	$(PY) -m pip install -r requirements.txt

# Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf $(PROJECT_NAME).egg-info/

# Lint using flake8
lint:
	flake8 --ignore=E203,W503 $(PROJECT_NAME)

# Automatically fix code style
style:
	black . --line-length=79

# Run tests
check: lint
	pytest

# Make SDSS data directory structure
sdss:
	mkdir -p data/sdss/cutouts
	curl http://amritrau.github.io/assets/data/info.csv > data/sdss/info.csv

hsc_demo:
	mkdir -p $(demodir)/cutouts
	curl https://epyc.astro.washington.edu/~aritrag/pub/hsc_morph/gampen_tutorial_files/for_train/info.csv > $(demodir)/info.csv
	wget -r -nd -A fits https://epyc.astro.washington.edu/~aritrag/pub/hsc_morph/gampen_tutorial_files/for_train/cutouts/ -P $(demodir)/cutouts/


# Clear all MLFlow logs (use with care!)
reset-mlflow: clean
	rm -rf mlruns/
