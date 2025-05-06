# Drawing Agent Dataset Generator

This project generates synthetic videos of drawing actions using VJepa. The videos simulate a white paper in a drawing software with mouse movements and line drawings.

## Features

- Generate synthetic drawing videos
- Simulate mouse movements and drawing actions
- Create a dataset suitable for VJepa training

## Setup

1. Clone this repository:
```bash
git clone https://github.com/yourusername/drawing-agent-dataset.git
cd drawing-agent-dataset
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

To generate synthetic drawing videos:

```bash
python generate_videos.py --num_videos 100 --output_dir ./dataset
```

## Project Structure

- `generate_videos.py`: Main script for generating synthetic videos
- `requirements.txt`: Project dependencies
- `dataset/`: Directory containing generated videos (created when running the script)

## License

MIT License 