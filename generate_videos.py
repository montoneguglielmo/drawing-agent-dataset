import cv2
import numpy as np
import os
import random
from tqdm import tqdm
import argparse
from datetime import datetime

class DrawingVideoGenerator:
    def __init__(self, width=200, height=200, fps=30, duration=3):
        self.width = width
        self.height = height
        self.fps = fps
        self.duration = duration
        self.total_frames = fps * duration
        
    def create_white_background(self):
        """Create a white background image."""
        return np.ones((self.height, self.width, 3), dtype=np.uint8) * 255
    
    def generate_random_path(self, num_points=10):
        """Generate a random path for the drawing."""
        points = []
        current_x = random.randint(50, self.width - 50)
        current_y = random.randint(50, self.height - 50)
        
        for _ in range(num_points):
            # Generate next point with some randomness
            next_x = current_x + random.randint(-50, 50)
            next_y = current_y + random.randint(-50, 50)
            
            # Keep points within bounds
            next_x = max(50, min(self.width - 50, next_x))
            next_y = max(50, min(self.height - 50, next_y))
            
            points.append((next_x, next_y))
            current_x, current_y = next_x, next_y
            
        return points
    
    def interpolate_points(self, points, num_frames):
        """Interpolate between points to create smooth movement."""
        interpolated = []
        frames_per_segment = num_frames // (len(points) - 1)
        
        for i in range(len(points) - 1):
            start = points[i]
            end = points[i + 1]
            for t in np.linspace(0, 1, frames_per_segment):
                x = int(start[0] * (1 - t) + end[0] * t)
                y = int(start[1] * (1 - t) + end[1] * t)
                interpolated.append((x, y))
        
        # Add the last point to ensure we reach the end
        interpolated.append(points[-1])
        
        # If we have fewer points than frames, pad with the last point
        while len(interpolated) < num_frames:
            interpolated.append(points[-1])
            
        return interpolated[:num_frames]  # Ensure we return exactly num_frames points
    
    def generate_video(self, output_path):
        """Generate a single drawing video."""
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
        
        # Generate random path
        points = self.generate_random_path()
        interpolated_points = self.interpolate_points(points, self.total_frames)
        
        # Create frames
        for i in range(self.total_frames):
            frame = self.create_white_background()
            
            # Draw all lines up to current point
            if i > 0:
                for j in range(1, i + 1):
                    cv2.line(frame, 
                            interpolated_points[j-1],
                            interpolated_points[j],
                            (0, 0, 0), 2)
            
            # Draw current cursor position
            cv2.circle(frame, interpolated_points[i], 5, (255, 0, 0), -1)
            
            out.write(frame)
        
        out.release()

def main():
    parser = argparse.ArgumentParser(description='Generate synthetic drawing videos')
    parser.add_argument('--num_videos', type=int, default=100, help='Number of videos to generate')
    parser.add_argument('--output_dir', type=str, default='./dataset', help='Output directory for videos')
    parser.add_argument('--width', type=int, default=200, help='Video width')
    parser.add_argument('--height', type=int, default=200, help='Video height')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second')
    parser.add_argument('--duration', type=int, default=3, help='Video duration in seconds')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize generator
    generator = DrawingVideoGenerator(
        width=args.width,
        height=args.height,
        fps=args.fps,
        duration=args.duration
    )
    
    # Generate videos
    print(f"Generating {args.num_videos} videos...")
    for i in tqdm(range(args.num_videos)):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(args.output_dir, f"drawing_{timestamp}_{i:04d}.mp4")
        generator.generate_video(output_path)
    
    print(f"Generated {args.num_videos} videos in {args.output_dir}")

if __name__ == "__main__":
    main() 