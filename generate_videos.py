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
    def __init__(self, config_path='config.yaml', fps=30, duration=3):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.width = self.config['image']['width']
        self.height = self.config['image']['height']
        self.fps = fps
        self.duration = duration
        self.total_frames = fps * duration
        self.compass_size = min(self.width, self.height) // 5  # 5 windows across the width
        self.compass_margin = min(self.width, self.height) // 20  # 1/20th of the smaller dimension
        self.show_compass_percentage = self.config.get('video', {}).get('show_compass', 1.0)  # Get from config, default to 1.0 (100%)
        self.fixed_background = self.config.get('video', {}).get('fixed_background', False)  # Get from config, default to False
        self.compass_margin = self.config.get('video', {}).get('compass_margin', 10)  # Get from config, default to 10
        
        # Load drawing mask configuration from video section
        self.pause_probability = self.config.get('video', {}).get('pause_probability', 0.5)
        
        # Fixed pause duration values (0.5 to 1.5 seconds)
        self.min_pause_seconds = 0.5
        self.max_pause_seconds = 1.5
        
        # Calculate pause frames from seconds
        self.min_pause_frames = int(fps * self.min_pause_seconds)
        self.max_pause_frames = int(fps * self.max_pause_seconds)
        
        # Load pre-generated backgrounds
        base_dir = self.config['output']['base_dir']
        backgrounds_path = os.path.join(base_dir, 'backgrounds', 'backgrounds.npy')
        if not os.path.exists(backgrounds_path):
            raise FileNotFoundError(f"Backgrounds file not found at {backgrounds_path}. Please run generate_backgrounds.py first.")
        self.backgrounds = np.load(backgrounds_path)
    
    def should_show_compass_for_video(self):
        """Decide whether this video should show the compass based on the percentage."""
        return random.random() < self.show_compass_percentage

    def get_random_background(self):
        """Get a random background from the pre-generated ones."""
        idx = random.randint(0, len(self.backgrounds) - 1)
        background = self.backgrounds[idx]
        
        # Scale to 0-255 range and keep as 2D array
        background = (background * 255).astype(np.uint8)
        return background

    def create_white_background(self):
        """Create a white background image."""
        return np.ones((self.height, self.width), dtype=np.uint8) * 255
    
    def draw_compass(self, frame, directions, is_drawing=True):
        """Draw 4 compasses and a status box across the upper edge of the image.
        
        Args:
            frame: The frame to draw on
            directions: List of 4 directions in degrees (0 is North, 90 is East)
            is_drawing: Boolean indicating if currently drawing
        """
        # Calculate window sizes and spacing using compass attributes
        total_width = self.width
        window_width = self.compass_size
        window_height = self.compass_size

        start_x = self.compass_size//2
        
        # Draw 4 compasses
        for i in range(4):
            if i < len(directions) and directions[i] is not None:
                # Calculate position for this compass
                x = start_x + i * window_width
                y = self.compass_size//2  # Use compass_margin for top margin
                
                radius = self.compass_size//2
                
                # Draw direction arrow
                angle = math.radians(directions[i])
                end_x = x+ int(radius * math.sin(angle))
                end_y = y - int(radius * math.cos(angle))
                cv2.arrowedLine(frame, (x, y), (end_x, end_y), 255, 2)
                
        
        # Draw status box (5th window)
        status_x = start_x + 4 * window_width
        status_y = self.compass_margin  # Use compass_margin for top margin
        status_color = 255 if is_drawing else 128  # White for drawing, Grey for pause
        
        # Draw status box
        cv2.rectangle(frame, 
                     (status_x, status_y),
                     (status_x + window_width, status_y + window_height),
                     status_color, -1)  # -1 for filled rectangle
        
        # Add status text
        status_text = "DRAW" if is_drawing else "PAUSE"
        text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x = status_x + (window_width - text_size[0]) // 2
        text_y = status_y + (window_height + text_size[1]) // 2
        cv2.putText(frame, status_text, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 2)  # Black text
    
    def calculate_direction(self, point1, point2):
        """Calculate the direction between two points in degrees (0 is North, 90 is East)."""
        if point1 is None or point2 is None:
            return None
        dx = point2[0] - point1[0]
        dy = point2[1] - point1[1]
        angle = math.degrees(math.atan2(dx, -dy))
        return angle if angle >= 0 else angle + 360
    
    def calculate_multiple_directions(self, interpolated_points, current_frame, drawing_mask):
        """Calculate 4 different directions based on current and future path points."""
        directions = []
        
        # Look-ahead frames for different compasses
        look_ahead_frames = [4, 6, 8, 10]  # Different look-ahead distances
        
        for look_ahead in look_ahead_frames:
            look_ahead_idx = min(current_frame + look_ahead, self.total_frames)
            direction = self.calculate_direction(
                interpolated_points[current_frame],
                interpolated_points[look_ahead_idx]
                )
            directions.append(direction)
        
        return directions
    
    def generate_random_path(self, num_points=None):
        """Generate a random path for the drawing."""
        points = []
        
        # Randomly select number of points from the specified list
        if num_points is None:
            n_points = [4, 5, 6]
            num_points = random.choice(n_points)
        
        # Calculate margins as 10% of dimensions
        margin_x = int(self.width * 0.1)
        margin_y = int(self.height * 0.1)
        
        compass_boundary_bottom = self.compass_size
        
        # Generate all points randomly, avoiding compass boundary
        x_coords = [random.randint(margin_x, self.width - margin_x) for _ in range(num_points)]
        y_coords = [random.randint(compass_boundary_bottom, self.height - margin_y) for _ in range(num_points)]
        points = list(zip(x_coords, y_coords))
        
        return points
    
    def generate_drawing_mask(self, num_frames):
        """Generate a mask indicating when to draw (1) and when to pause (0)."""
        # Initialize mask with all ones (drawing)
        mask = np.ones(num_frames, dtype=int)
        
        # Check if this video should have a pause based on configuration
        if random.random() < self.pause_probability:
            # Randomly select start of pause
            pause_start = random.randint(0, num_frames - self.max_pause_frames)
            
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
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height), isColor=False)
        
        # Generate random path
        points = self.generate_random_path()
        interpolated_points = self.interpolate_points(points, self.total_frames)
        
        # Generate drawing mask
        drawing_mask = self.generate_drawing_mask(self.total_frames)
        
        # Decide whether this video should show the compass (once per video)
        show_compass_for_this_video = self.should_show_compass_for_video()
        
        # If fixed background is enabled, select one background at initialization
        if self.fixed_background:
            idx = random.randint(0, len(self.backgrounds) - 1)
            self.current_background = (self.backgrounds[idx] * 255).astype(np.uint8)
        
        # Create frames
        for i in range(self.total_frames):
            if self.fixed_background:
                frame = self.current_background.copy()
            else:
                frame = self.get_random_background()
            
            # Draw all lines up to current point, respecting the drawing mask
            if i > 0:
                for j in range(1, i + 1):
                    if drawing_mask[j-1] and drawing_mask[j]:  # Only draw if both points are in drawing mode
                        cv2.line(frame, interpolated_points[j-1], interpolated_points[j], 255, thickness=3)
            
            # Determine drawing state for current frame
            is_drawing = bool(drawing_mask[i])
            
            if show_compass_for_this_video:
                # Calculate multiple directions for the 4 compasses
                multiple_directions = self.calculate_multiple_directions(interpolated_points, i, drawing_mask)
                self.draw_compass(frame, multiple_directions, is_drawing)
            
            out.write(frame)
        
        out.release()

def main():
    parser = argparse.ArgumentParser(description='Generate synthetic drawing videos')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration file')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second')
    parser.add_argument('--duration', type=int, default=3, help='Video duration in seconds')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get output directory from config
    base_dir = config['output']['base_dir']
    output_dir = os.path.join(base_dir, 'videos')
    os.makedirs(output_dir, exist_ok=True)
    
    # Get background directory from config
    background_dir = os.path.join(base_dir, 'backgrounds')
    
    # Initialize generator
    generator = DrawingVideoGenerator(
        config_path=args.config,
        fps=args.fps,
        duration=args.duration
    )
    
    # Generate videos
    print(f"Generating {config['generation']['num_videos']} videos...")
    for i in tqdm(range(config['generation']['num_videos'])):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"drawing_{timestamp}_{i:04d}.mp4")
        generator.generate_video(output_path)
    
    print(f"Generated {config['generation']['num_videos']} videos in {output_dir}")

if __name__ == "__main__":
    main() 