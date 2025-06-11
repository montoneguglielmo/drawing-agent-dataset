import os
import csv
import random
import yaml
from pathlib import Path

def create_video_index(dataset_path, output_file):
    # Get all video files
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    video_files = []
    
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(video_extensions):
                video_files.append(os.path.join(root, file))
    
    # Sort files for reproducibility
    video_files.sort()
    
    # Create CSV file
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=' ')
        # Write each video with a random label
        for video_path in video_files:
            # Use absolute path
            abs_path = os.path.abspath(video_path)
            # Assign random label (0-999)
            label = random.randint(0, 999)
            writer.writerow([abs_path, label])
    print(f"Created video index with {len(video_files)} videos at {output_file}")

def load_config(config_path):
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found at {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing config file: {e}")

if __name__ == "__main__":
    # Load config
    config = load_config("config.yaml")
    
    # Get base directory from config
    base_dir = config['output']['base_dir']
    if not base_dir:
        raise ValueError("base_dir not found in config file")
    
    # Create videos directory if it doesn't exist
    videos_dir = os.path.join(base_dir, "videos")
    os.makedirs(videos_dir, exist_ok=True)
    
    # Create output file in the videos directory
    output_file = os.path.join(videos_dir, "videos_index.csv")
    
    create_video_index(videos_dir, output_file) 