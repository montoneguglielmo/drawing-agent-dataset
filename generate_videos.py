import cv2
import numpy as np
import os
import random
from tqdm import tqdm
import argparse
from datetime import datetime
import math
from scipy.interpolate import interp1d

class DrawingVideoGenerator:
    def __init__(self, width=200, height=200, fps=30, duration=3):
        self.width = width
        self.height = height
        self.fps = fps
        self.duration = duration
        self.total_frames = fps * duration
        self.compass_size = 40  # Size of the compass in pixels
        self.compass_margin = 10  # Margin from the edge
        self.look_ahead_frames = 5  # Number of frames to look ahead for direction
        
    def create_white_background(self):
        """Create a white background image."""
        return np.ones((self.height, self.width, 3), dtype=np.uint8) * 255
    
    def draw_compass(self, frame, direction):
        """Draw a compass in the top right corner showing the current drawing direction."""
        # Calculate compass position
        x = self.width - self.compass_size - self.compass_margin
        y = self.compass_margin
        
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
            
            # Calculate and draw direction with look-ahead
            look_ahead_idx = min(i + self.look_ahead_frames, self.total_frames - 1)
            if look_ahead_idx < self.total_frames - 1:
                direction = self.calculate_direction(
                    interpolated_points[look_ahead_idx],
                    interpolated_points[look_ahead_idx + 1]
                )
            else:
                # For the last frame, use the previous direction
                direction = self.calculate_direction(
                    interpolated_points[look_ahead_idx - 1],
                    interpolated_points[look_ahead_idx]
                )
            self.draw_compass(frame, direction)
            
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