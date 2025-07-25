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

# Generate masked videos
generate_masked_videos: generate_videos
	$(PYTHON) generate_masked_videos.py

# Process MNIST dataset
generate_mnist:
	$(PYTHON) process_mnist.py

# Generate curve lines dataset
generate_curve_lines: generate_backgrounds
	$(PYTHON) generate_curve_lines.py

# Compare video frames with MNIST samples
compare_samples:
	$(PYTHON) compare_samples.py

# Copy config to dataset directory
copy_config:
	$(PYTHON) -c "import yaml; config = yaml.safe_load(open('config.yaml')); \
		import shutil; \
		import os; \
		base_dir = config['output']['base_dir']; \
		os.makedirs(base_dir, exist_ok=True); \
		shutil.copy('config.yaml', os.path.join(base_dir, 'config.yaml'))"

# Generate complete dataset
generate_dataset: copy_config generate_backgrounds generate_mnist generate_curve_lines generate_videos generate_masked_videos compare_samples create_video_index

# Create video index
create_video_index:
	$(PYTHON) create_video_index.py

# Clean generated files
clean:
	$(PYTHON) clean.py

# Clean everything including virtual environment
clean-all: clean
	rm -rf $(VENV_PATH)

.PHONY: all venv generate_backgrounds generate_videos generate_masked_videos generate_mnist generate_curve_lines compare_samples generate_dataset create_video_index copy_config clean clean-all 