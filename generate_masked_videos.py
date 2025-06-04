import cv2
import numpy as np
import os
import random
from tqdm import tqdm
import argparse
import yaml
import glob
from pathlib import Path

class MaskedVideoGenerator:
    def __init__(self, config_path='config.yaml'):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.width = self.config['image']['width']
        self.height = self.config['image']['height']
        self.mask_config = self.config['mask']
        
    def generate_mask(self, frame_shape):
        """Generate a random mask for a single frame."""
        mask = np.ones(frame_shape, dtype=np.uint8) * 255

        # Generate random blocks
        for _ in range(self.mask_config['num_blocks']):
            aspect_ratio = random.uniform(*self.mask_config['aspect_ratio'])
            spatial_scale = random.uniform(*self.mask_config['spatial_scale'])
        
            block_width = int(self.width * spatial_scale)
            block_height = int(block_width * aspect_ratio)
            
            x = random.randint(0, self.width - block_width)
            y = random.randint(0, self.height - block_height)
            
            # Set the block region to zero
            mask[y:y+block_height, x:x+block_width] = 0
        
        return mask
    
    def process_video(self, input_path, output_path):
        """Process a single grayscale video by applying masks."""
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Error opening video {input_path}")
            return False
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height), isColor=False)
        if not out.isOpened():
            print(f"Error creating output video {output_path}")
            cap.release()
            return False
        
        # Read the first frame and ensure it's grayscale
        ret, first_frame = cap.read()
        if not ret:
            print(f"Error reading first frame from {input_path}")
            cap.release()
            out.release()
            return False

        # Convert to grayscale if needed
        if len(first_frame.shape) == 3:
            first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

        # Generate mask
        mask = self.generate_mask(first_frame.shape)

        # Reset video
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            masked_frame = cv2.bitwise_and(frame, mask)
            out.write(masked_frame)
            frame_count += 1
        
        cap.release()
        out.release()

        if frame_count != total_frames:
            print(f"Warning: Processed {frame_count} frames out of {total_frames}")
            return False

        return True

def main():
    parser = argparse.ArgumentParser(description='Generate masked grayscale videos')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration file')
    parser.add_argument('--num_videos', type=int, default=10, help='Number of videos to process')
    parser.add_argument('--input_dir', type=str, help='Directory containing input videos')
    parser.add_argument('--output_dir', type=str, help='Directory for output masked videos')

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    base_dir = config['output']['base_dir']
    input_dir = args.input_dir or os.path.join(base_dir, 'videos')
    output_dir = args.output_dir or os.path.join(base_dir, 'masked_videos_examples')
    os.makedirs(output_dir, exist_ok=True)

    video_files = glob.glob(os.path.join(input_dir, '*.mp4'))
    if not video_files:
        print(f"No video files found in {input_dir}")
        return

    selected_videos = random.sample(video_files, min(args.num_videos, len(video_files)))
    generator = MaskedVideoGenerator(config_path=args.config)

    print(f"Processing {len(selected_videos)} videos...")
    for video_path in tqdm(selected_videos):
        video_name = os.path.basename(video_path)
        output_path = os.path.join(output_dir, f"masked_{video_name}")
        success = generator.process_video(video_path, output_path)
        if not success:
            print(f"Warning: Failed to process {video_name}")

    print(f"Masked videos saved in {output_dir}")

if __name__ == "__main__":
    main()