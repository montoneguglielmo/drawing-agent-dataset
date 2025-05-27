# Drawing Agent Dataset Generator

This project generates synthetic drawing videos and MNIST dataset samples for training drawing agents. It creates a dataset that includes:
- Perlin noise backgrounds
- Drawing videos with compass indicators
- MNIST digit samples with backgrounds
- Comparison samples between videos and MNIST digits

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

# Output directories
output:
  videos: "../datasets/drawing-agent-test/video"
  mnist: "../datasets/drawing-agent-test/mnist"
  backgrounds: "../datasets/drawing-agent-test/backgrounds"
  comparisons: "../datasets/drawing-agent-test/comparisons"

# Generation parameters
generation:
  num_videos: 15
```

## Usage

The project provides several Makefile targets for different operations:

### Generate Complete Dataset
To generate the entire dataset (backgrounds, MNIST samples, videos, and comparisons):
```bash
make generate_dataset
```

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

- Generate comparison samples:
```bash
make compare_samples
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
- `process_mnist.py`: Processes MNIST dataset and combines with backgrounds
- `compare_samples.py`: Creates comparison samples between videos and MNIST digits
- `config.yaml`: Configuration file for all parameters
- `Makefile`: Build automation and task management

## Output Structure

The generated dataset will be organized in the following structure:
```
datasets/drawing-agent-test/
├── backgrounds/
│   └── backgrounds.npy
├── mnist/
│   ├── samples/
│   ├── train_images.npy
│   ├── train_labels.npy
│   ├── test_images.npy
│   └── test_labels.npy
├── video/
│   └── drawing_*.mp4
└── comparisons/
    └── comparison_*.png
```

## Features

- Perlin noise background generation
- Drawing videos with compass indicators showing direction
- MNIST digit processing with background integration
- Comparison samples for visual analysis
- Configurable parameters through YAML
- Automated build process with Makefile 