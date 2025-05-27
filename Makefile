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
generate_backgrounds:
	$(PYTHON) generate_backgrounds.py

# Generate drawing videos
generate_videos: generate_backgrounds
	$(PYTHON) generate_videos.py

# Process MNIST dataset
generate_mnist:
	$(PYTHON) process_mnist.py

# Compare video frames with MNIST samples
compare_samples:
	$(PYTHON) compare_samples.py

# Generate complete dataset
generate_dataset: generate_backgrounds generate_mnist generate_videos compare_samples

# Clean generated files
clean:
	$(PYTHON) -c "import yaml; config = yaml.safe_load(open('config.yaml')); \
		import shutil; \
		shutil.rmtree(config['output']['mnist'], ignore_errors=True); \
		shutil.rmtree(config['output']['videos'], ignore_errors=True); \
		shutil.rmtree('comparisons', ignore_errors=True); \
		import os; \
		backgrounds_path = os.path.join(config['output']['backgrounds'], 'backgrounds.npy'); \
		os.remove(backgrounds_path) if os.path.exists(backgrounds_path) else None"

# Clean everything including virtual environment
clean-all: clean
	rm -rf $(VENV_PATH)

.PHONY: all venv generate_backgrounds generate_videos generate_mnist compare_samples generate_dataset clean clean-all 