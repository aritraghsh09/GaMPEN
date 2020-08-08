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
	flake8 $(PROJECT_NAME)

# Clear all MLFlow logs. Does not clear artifacts
reset-mlflow: clean
	rm -rf mlruns/
