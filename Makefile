# Makefile for pyBRIEF project

# Variables
PYTHON = ./env/bin/python
PYTEST = $(PYTHON) -m pytest
PIP = ./env/bin/pip

# Default target
.PHONY: help
help:
	@echo "Available commands:"
	@echo "  make test          - Run all tests"
	@echo "  make test-verbose  - Run all tests with verbose output"
	@echo "  make test-file     - Run tests for a specific file (usage: make test-file FILE=test_utils.py)"
	@echo "  make test-coverage - Run tests with coverage report"
	@echo "  make test-fast     - Run tests without verbose output"
	@echo "  make clean         - Clean cache files"
	@echo "  make install       - Install dependencies"
	@echo "  make lint          - Check code style (if flake8 is installed)"
	@echo "  make setup         - Set up the development environment"

# Test targets
.PHONY: test
test:
	$(PYTEST) -v

.PHONY: test-verbose
test-verbose:
	$(PYTEST) -v -s

.PHONY: test-fast
test-fast:
	$(PYTEST)

.PHONY: test-file
test-file:
	@if [ -z "$(FILE)" ]; then \
		echo "Usage: make test-file FILE=test_utils.py"; \
		exit 1; \
	fi
	$(PYTEST) tests/$(FILE) -v

.PHONY: test-coverage
test-coverage:
	$(PYTEST) --cov=pybrief --cov-report=html --cov-report=term-missing

.PHONY: test-specific
test-specific:
	@if [ -z "$(TEST)" ]; then \
		echo "Usage: make test-specific TEST=test_function_name"; \
		exit 1; \
	fi
	$(PYTEST) -k "$(TEST)" -v

# Development targets
.PHONY: install
install:
	$(PIP) install -r requirements.txt

.PHONY: install-dev
install-dev:
	$(PIP) install pytest pytest-cov flake8 black

.PHONY: lint
lint:
	@if [ -f "./env/bin/flake8" ]; then \
		./env/bin/flake8 pybrief tests; \
	else \
		echo "flake8 not installed. Run 'make install-dev' first."; \
	fi

.PHONY: format
format:
	@if [ -f "./env/bin/black" ]; then \
		./env/bin/black pybrief tests; \
	else \
		echo "black not installed. Run 'make install-dev' first."; \
	fi

# Utility targets
.PHONY: clean
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -name ".coverage" -delete

.PHONY: setup
setup:
	@echo "Setting up development environment..."
	python -m venv env
	$(PIP) install --upgrade pip
	$(MAKE) install
	$(MAKE) install-dev
	@echo "Setup complete! You can now run 'make test' to run tests."

.PHONY: info
info:
	@echo "Python: $(PYTHON)"
	@echo "PyTest: $(PYTEST)"
	@$(PYTHON) --version
	@$(PYTEST) --version

# Quick shortcuts
.PHONY: t
t: test

.PHONY: tv
tv: test-verbose

.PHONY: tf
tf: test-fast

# Test specific modules
.PHONY: test-utils
test-utils:
	$(PYTEST) tests/test_utils.py -v

.PHONY: test-filters
test-filters:
	$(PYTEST) tests/test_filters.py -v

.PHONY: test-corners
test-corners:
	$(PYTEST) tests/test_corners.py -v

.PHONY: test-nms
test-nms:
	$(PYTEST) tests/test_nms.py -v

.PHONY: test-brief
test-brief:
	$(PYTEST) tests/test_brief_descriptor.py tests/test_brief_pattern.py -v

.PHONY: test-matching
test-matching:
	$(PYTEST) tests/test_matching.py -v