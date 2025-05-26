import cv2
import numpy as np
import os
import random
from tqdm import tqdm
import argparse
from datetime import datetime
import math
from scipy.interpolate import interp1d
from noise import pnoise2  # Add this import for Perlin noise
import yaml  # Add yaml import

class DrawingVideoGenerator:
    def __init__(self, config_path='config.yaml', fps=30, duration=3, show_compass=True):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.width = self.config['image']['width']
        self.height = self.config['image']['height']
        self.fps = fps
        self.duration = duration
        self.total_frames = fps * duration
        self.compass_size = 40  # Size of the compass in pixels
        self.compass_margin = 10  # Margin from the edge
        self.look_ahead_frames = 5  # Number of frames to look ahead for direction
        self.min_pause_frames = int(fps * 0.5)  # Minimum pause duration (0.5 seconds)
        self.max_pause_frames = int(fps * 1.5)  # Maximum pause duration (1.5 seconds)
        self.show_compass = show_compass  # Whether to show the compass
        
    def create_smooth_grey_background(self):
        """Create a smooth grey background using Perlin noise."""
        # Create a grid of coordinates
        x = np.linspace(0, 1, self.width)
        y = np.linspace(0, 1, self.height)
        X, Y = np.meshgrid(x, y)
        
        # Get parameters from config
        bg_config = self.config['background']
        scale = random.uniform(bg_config['scale']['min'], bg_config['scale']['max'])
        octaves = random.randint(bg_config['octaves']['min'], bg_config['octaves']['max'])
        persistence = random.uniform(bg_config['persistence']['min'], bg_config['persistence']['max'])
        lacunarity = random.uniform(bg_config['lacunarity']['min'], bg_config['lacunarity']['max'])
        
        # Generate noise for each pixel
        noise = np.zeros((self.height, self.width))
        for i in range(self.height):
            for j in range(self.width):
                noise[i, j] = pnoise2(X[i, j] * scale, 
                                    Y[i, j] * scale, 
                                    octaves=octaves, 
                                    persistence=persistence, 
                                    lacunarity=lacunarity)
        
        # Map noise values directly to 0-255 range
        # Perlin noise typically ranges from -1 to 1, so we map it to 0-255
        grey = ((noise + 1) * 127.5).astype(np.uint8)
        
        # Create a 3-channel image with the same grey value in all channels
        background = np.stack([grey, grey, grey], axis=-1)
        
        return background

    def create_white_background(self):
        """Create a white background image."""
        return np.ones((self.height, self.width, 3), dtype=np.uint8) * 255
    
    def draw_compass(self, frame, direction, is_drawing=True):
        """Draw a compass in the top right corner showing the current drawing direction."""
        # Calculate compass position
        x = self.width - self.compass_size - self.compass_margin
        y = self.compass_margin
        
        # Draw status indicator box
        status_box_size = 15
        status_x = x - status_box_size - 5  # 5 pixels margin from compass
        status_y = y
        status_color = (0, 0, 255) if is_drawing else (128, 128, 128)  # Red for drawing, Grey for pause
        cv2.rectangle(frame, 
                     (status_x, status_y),
                     (status_x + status_box_size, status_y + status_box_size),
                     status_color, -1)  # -1 for filled rectangle
        
        # Draw cardinal points
        center = (x + self.compass_size//2, y + self.compass_size//2)
        radius = self.compass_size//2 - 5
        
        # Draw N, S, E, W markers
        cv2.putText(frame, "N", (center[0]-5, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(frame, "S", (center[0]-5, y+self.compass_size-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(frame, "E", (x+self.compass_size-10, center[1]+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(frame, "W", (x+5, center[1]+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Draw direction arrow
        if direction is not None:
            angle = math.radians(direction)
            end_x = center[0] + int(radius * math.sin(angle))
            end_y = center[1] - int(radius * math.cos(angle))
            cv2.arrowedLine(frame, center, (end_x, end_y), (0, 0, 255), 2)
    
    def calculate_direction(self, point1, point2):
        """Calculate the direction between two points in degrees (0 is North, 90 is East)."""
        if point1 is None or point2 is None:
            return None
        dx = point2[0] - point1[0]
        dy = point2[1] - point1[1]
        angle = math.degrees(math.atan2(dx, -dy))
        return angle if angle >= 0 else angle + 360
    
    def generate_random_path(self, num_points=10):
        """Generate a random path for the drawing."""
        points = []
        # Calculate margins as 10% of dimensions
        margin_x = int(self.width * 0.1)
        margin_y = int(self.height * 0.1)
        
        # Initial point within margins
        current_x = random.randint(margin_x, self.width - margin_x)
        current_y = random.randint(margin_y, self.height - margin_y)
        
        for _ in range(num_points):
            # Generate next point with some randomness
            # Use 20% of dimensions for maximum step size
            max_step_x = int(self.width * 0.2)
            max_step_y = int(self.height * 0.2)
            next_x = current_x + random.randint(-max_step_x, max_step_x)
            next_y = current_y + random.randint(-max_step_y, max_step_y)
            
            # Keep points within margins
            next_x = max(margin_x, min(self.width - margin_x, next_x))
            next_y = max(margin_y, min(self.height - margin_y, next_y))
            
            points.append((next_x, next_y))
            current_x, current_y = next_x, next_y
            
        return points
    
    def generate_drawing_mask(self, num_frames):
        """Generate a mask indicating when to draw (1) and when to pause (0)."""
        # Initialize mask with all ones (drawing)
        mask = np.ones(num_frames, dtype=int)
        
        # 50% chance to have a pause
        if random.random() < 0.5:
            # Randomly select start of pause
            pause_start = random.randint(0, num_frames - self.min_pause_frames)
            
            # Randomly select pause duration
            pause_duration = random.randint(self.min_pause_frames, self.max_pause_frames)
            
            # Ensure pause doesn't exceed video length
            pause_end = min(pause_start + pause_duration, num_frames)
            
            # Set pause frames to 0
            mask[pause_start:pause_end] = 0
        
        return mask
    
    def interpolate_points(self, points, num_frames):
        """Interpolate between points using scipy's interp1d for smooth movement."""
        if len(points) < 2:
            return points * num_frames
            
        # Convert points to numpy arrays for x and y coordinates
        points = np.array(points)
        x_coords = points[:, 0]
        y_coords = points[:, 1]
        
        # Create interpolation functions for x and y coordinates
        t = np.linspace(0, 1, len(points))
        fx = interp1d(t, x_coords, kind='cubic', bounds_error=False, fill_value=(x_coords[0], x_coords[-1]))
        fy = interp1d(t, y_coords, kind='cubic', bounds_error=False, fill_value=(y_coords[0], y_coords[-1]))
        
        # Generate interpolated points
        t_new = np.linspace(0, 1, num_frames)
        x_new = fx(t_new)
        y_new = fy(t_new)
        
        # Round to integers and create list of tuples
        interpolated = list(zip(np.round(x_new).astype(int), np.round(y_new).astype(int)))
        
        return interpolated
    
    def generate_video(self, output_path):
        """Generate a single drawing video."""
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
        
        # Generate random path
        points = self.generate_random_path()
        interpolated_points = self.interpolate_points(points, self.total_frames)
        
        # Generate drawing mask
        drawing_mask = self.generate_drawing_mask(self.total_frames)
        
        # Create frames
        for i in range(self.total_frames):
            frame = self.create_smooth_grey_background()
            
            # Draw all lines up to current point, respecting the drawing mask
            if i > 0:
                for j in range(1, i + 1):
                    if drawing_mask[j-1] and drawing_mask[j]:  # Only draw if both points are in drawing mode
                        cv2.line(frame, 
                                interpolated_points[j-1],
                                interpolated_points[j],
                                (0, 0, 0), 2)
            
            # Draw current cursor position
            cv2.circle(frame, interpolated_points[i], 5, (255, 0, 0), -1)
            
            # Calculate and draw direction with look-ahead
            look_ahead_idx = min(i + self.look_ahead_frames, self.total_frames - 1)
            if look_ahead_idx < self.total_frames - 1:
                direction = self.calculate_direction(
                    interpolated_points[look_ahead_idx],
                    interpolated_points[look_ahead_idx + 1]
                )
                # Use look-ahead for drawing state
                is_drawing = bool(drawing_mask[look_ahead_idx])
            else:
                # For the last frame, use the previous direction
                direction = self.calculate_direction(
                    interpolated_points[look_ahead_idx - 1],
                    interpolated_points[look_ahead_idx]
                )
                # Use current state for last frames
                is_drawing = bool(drawing_mask[i])
            
            if self.show_compass:
                self.draw_compass(frame, direction, is_drawing)
            
            out.write(frame)
        
        out.release()

def main():
    parser = argparse.ArgumentParser(description='Generate synthetic drawing videos')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration file')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second')
    parser.add_argument('--duration', type=int, default=3, help='Video duration in seconds')
    parser.add_argument('--no_compass', action='store_true', help='Disable the compass in the video')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get output directory and number of videos from config
    output_dir = config['output']['videos']
    num_videos = config['generation']['num_videos']
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize generator
    generator = DrawingVideoGenerator(
        config_path=args.config,
        fps=args.fps,
        duration=args.duration,
        show_compass=not args.no_compass
    )
    
    # Generate videos
    print(f"Generating {num_videos} videos...")
    for i in tqdm(range(num_videos)):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"drawing_{timestamp}_{i:04d}.mp4")
        generator.generate_video(output_path)
    
    print(f"Generated {num_videos} videos in {output_dir}")

if __name__ == "__main__":
    main() 