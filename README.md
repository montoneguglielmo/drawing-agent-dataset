# Drawing Agent Dataset Generator

This project generates synthetic drawing videos, a MNIST dataset and a dataset of images of curves and streight lines. 
It creates a 3 datasets that includes:
- Perlin noise backgrounds
- Drawing videos with compass indicators
- Masked versions of drawing videos (to visualize VJepa masking strategy)
- MNIST digit samples with backgrounds
- Comparison samples between videos and MNIST digits (to make sure drawing in videos are close in style to MNIST digits)
- Curve and straight line image classification dataset

## Prerequisites

- Python 3.x
- Virtual environment (recommended)

## Installation

1. Clone the repository
2. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Unix/macOS
# or
.\venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

The project uses a `config.yaml` file to configure various parameters:

```yaml
# Image dimensions
image:
  width: 100
  height: 100

# Background generation parameters
background:
  num_backgrounds: 1000  # Number of backgrounds to generate
  scale:
    min: 2.0
    max: 20.0
  octaves:
    min: 2
    max: 10
  persistence:
    min: 0.1
    max: 0.9
  lacunarity:
    min: 1.0
    max: 3.0

# Output directory
output:
  base_dir: "../datasets/drawing-agent-test"

# Generation parameters
generation:
  num_videos: 15

# Mask parameters for masked videos
mask:
  aspect_ratio:
    - 0.75
    - 1.5
  num_blocks: 8
  spatial_scale:
    - 0.15
    - 0.15
  temporal_scale:
    - 1.0
    - 1.0

# Curve lines dataset parameters
curve_lines_dataset:
  num_samples_per_class: 1000  # Number of samples per class (straight lines and curves)
  num_translations_per_shape: 8  # Number of translated versions per base shape
```

## Usage

The project provides several Makefile targets for different operations:

### Generate Complete Dataset
To generate the entire dataset (backgrounds, MNIST samples, videos, masked videos, and comparisons):
```bash
make generate_dataset
```

This will:
1. Copy the config file to the dataset directory
2. Generate backgrounds
3. Process MNIST data
4. Generate curve and straight line dataset
5. Generate videos
6. Generate masked videos
7. Create comparisons
8. Create the video index

### Individual Components
You can also generate individual components:

- Generate backgrounds:
```bash
make generate_backgrounds
```

- Generate MNIST samples:
```bash
make generate_mnist
```

- Generate drawing videos:
```bash
make generate_videos
```

- Generate masked videos:
```bash
make generate_masked_videos
```

- Generate curve and straight line dataset:
```bash
make generate_curve_lines
```

- Generate comparison samples:
```bash
make compare_samples
```

- Create video index:
```bash
make create_video_index
```

- Copy config to dataset directory:
```bash
make copy_config
```

### Cleaning Up
To clean generated files:
```bash
make clean
```

To clean everything including the virtual environment:
```bash
make clean-all
```

## Project Structure

- `generate_backgrounds.py`: Generates Perlin noise backgrounds
- `generate_videos.py`: Creates drawing videos with compass indicators
- `generate_masked_videos.py`: Creates masked versions of drawing videos
- `generate_curve_lines.py`: Creates curve and straight line classification dataset
- `process_mnist.py`: Processes MNIST dataset and combines with backgrounds
- `compare_samples.py`: Creates comparison samples between videos and MNIST digits
- `create_drawing_agent_index.py`: Creates an index file for the generated videos
- `config.yaml`: Configuration file for all parameters
- `Makefile`: Build automation and task management

## Output Structure

The generated dataset will be organized in the following structure:
```
datasets/drawing-agent-test/
├── config.yaml
├── backgrounds/
│   └── backgrounds.npy
├── mnist/
│   ├── train/
│   │   ├── class0/
│   │   ├── class1/
│   │   └── ...
│   ├── val/
│   │   ├── class0/
│   │   ├── class1/
│   │   └── ...
│   ├── test/
│   │   ├── class0/
│   │   ├── class1/
│   │   └── ...
│   ├── train_images.npy
│   ├── train_labels.npy
│   ├── val_images.npy
│   ├── val_labels.npy
│   ├── test_images.npy
│   └── test_labels.npy
├── videos/
│   └── drawing_agent_index.csv
├── masked_videos_examples/
│   └── masked_*.mp4
├── curve_lines_dataset/
│   ├── train/
│   │   ├── class0/  # Straight lines
│   │   └── class1/  # Curves
│   ├── val/
│   │   ├── class0/  # Straight lines
│   │   └── class1/  # Curves
│   └── test/
│       ├── class0/  # Straight lines
│       └── class1/  # Curves
└── comparisons/
    └── comparison_*.png
```