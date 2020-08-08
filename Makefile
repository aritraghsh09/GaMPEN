.PHONY: clean data lint requirements

PROJECT_NAME = ggt
PY = python

# Install dependencies with pip
requirements:
	$(PY) -m pip install -U pip setuptools wheel
	$(PY) -m pip install -r requirements.txt

# Preprocess the dataset
data: requirements
	$(PY) src/data/make_dataset.py data/raw data/processed

# Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

# Lint using flake8
lint:
	flake8 src
