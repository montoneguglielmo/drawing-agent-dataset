# Python virtual environment path
VENV_PATH := venv
PYTHON := $(VENV_PATH)/bin/python
PIP := $(VENV_PATH)/bin/pip

# Default target
all: venv generate_videos process_mnist

# Create and setup virtual environment
venv:
	python3 -m venv $(VENV_PATH)
	$(PIP) install -r requirements.txt

# Generate drawing videos
generate_videos:
	$(PYTHON) generate_videos.py

# Process MNIST dataset
process_mnist:
	$(PYTHON) process_mnist.py

# Clean generated files
clean:
	rm -rf processed_mnist/
	rm -rf dataset/

# Clean everything including virtual environment
clean-all: clean
	rm -rf $(VENV_PATH)

.PHONY: all venv generate_videos process_mnist clean clean-all 