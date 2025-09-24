import cv2
import numpy as np
import os
import random
from tqdm import tqdm
import argparse
from datetime import datetime
import math
from scipy.interpolate import interp1d
import yaml

class CurveLineGenerator:
    def __init__(self, config_path='config.yaml', curve_config=None):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.width = self.config['image']['width']
        self.height = self.config['image']['height']
        
        # Use provided curve_config or fall back to first curve config in config file
        if curve_config is None:
            curve_config = self.config.get('curve_lines_dataset', [{}])[0] if isinstance(self.config.get('curve_lines_dataset'), list) else self.config.get('curve_lines_dataset', {})
        
        self.curve_config = curve_config
        
        # Load pre-generated backgrounds
        base_dir = self.config['output']['base_dir']
        backgrounds_path = os.path.join(base_dir, 'backgrounds', 'backgrounds.npy')
        if not os.path.exists(backgrounds_path):
            raise FileNotFoundError(f"Backgrounds file not found at {backgrounds_path}. Please run generate_backgrounds.py first.")
        self.backgrounds = np.load(backgrounds_path)
    
    def get_random_background(self):
        """Get a random background from the pre-generated ones."""
        idx = random.randint(0, len(self.backgrounds) - 1)
        background = self.backgrounds[idx]
        
        # Scale to 0-255 range and keep as 2D array
        background = (background * 255).astype(np.uint8)
        return background
    
    def generate_straight_line_points(self):
        """Generate 3 points that form a straight line."""
        # Calculate margins as 10% of dimensions
        margin_x = int(self.width * 0.1)
        margin_y = int(self.height * 0.1)
        
        # Generate two random points within margins
        x1 = random.randint(margin_x, self.width - margin_x)
        y1 = random.randint(margin_y, self.height - margin_y)
        x2 = random.randint(margin_x, self.width - margin_x)
        y2 = random.randint(margin_y, self.height - margin_y)
        
        # Calculate midpoint for the third point
        x3 = (x1 + x2) // 2
        y3 = (y1 + y2) // 2
        
        # Add small random perturbation to make it slightly curved (but still mostly straight)
        # This ensures the classifier has to learn to distinguish between truly straight and curved lines
        perturbation = random.randint(-2, 2)
        x3 += perturbation
        y3 += perturbation
        
        # Ensure points stay within bounds
        x3 = max(margin_x, min(self.width - margin_x, x3))
        y3 = max(margin_y, min(self.height - margin_y, y3))
        
        return [(x1, y1), (x2, y2), (x3, y3)]
    
    def generate_curve_points(self):
        """Generate 3 points that form a curve."""
        # Calculate margins as 10% of dimensions
        margin_x = int(self.width * 0.1)
        margin_y = int(self.height * 0.1)
        
        # Generate three points that will form a curve
        # Start and end points
        x1 = random.randint(margin_x, self.width - margin_x)
        y1 = random.randint(margin_y, self.height - margin_y)
        x3 = random.randint(margin_x, self.width - margin_x)
        y3 = random.randint(margin_y, self.height - margin_y)
        
        # Middle point that creates a curve
        # Position it away from the straight line between x1,y1 and x3,y3
        mid_x = (x1 + x3) // 2
        mid_y = (y1 + y3) // 2
        
        # Add significant offset to create a curve
        offset_x = random.randint(-int(self.width * 0.15), int(self.width * 0.15))
        offset_y = random.randint(-int(self.height * 0.15), int(self.height * 0.15))
        
        x2 = mid_x + offset_x
        y2 = mid_y + offset_y
        
        # Ensure points stay within bounds
        x2 = max(margin_x, min(self.width - margin_x, x2))
        y2 = max(margin_y, min(self.height - margin_y, y2))
        
        return [(x1, y1), (x2, y2), (x3, y3)]
    
    def interpolate_points(self, points, num_points=50):
        """Interpolate between points using scipy's interp1d for smooth curves."""
        if len(points) < 2:
            return points
            
        # Convert points to numpy arrays for x and y coordinates
        points = np.array(points)
        x_coords = points[:, 0]
        y_coords = points[:, 1]
        
        # Create interpolation functions for x and y coordinates
        t = np.linspace(0, 1, len(points))
        
        # Use linear interpolation for 3 points, cubic for more points
        if len(points) == 3:
            fx = interp1d(t, x_coords, kind='linear', bounds_error=False, fill_value=(x_coords[0], x_coords[-1]))
            fy = interp1d(t, y_coords, kind='linear', bounds_error=False, fill_value=(y_coords[0], y_coords[-1]))
        else:
            fx = interp1d(t, x_coords, kind='cubic', bounds_error=False, fill_value=(x_coords[0], x_coords[-1]))
            fy = interp1d(t, y_coords, kind='cubic', bounds_error=False, fill_value=(y_coords[0], y_coords[-1]))
        
        # Generate interpolated points
        t_new = np.linspace(0, 1, num_points)
        x_new = fx(t_new)
        y_new = fy(t_new)
        
        # Round to integers and create list of tuples
        interpolated = list(zip(np.round(x_new).astype(int), np.round(y_new).astype(int)))
        
        return interpolated
    
    def draw_shape(self, points, background):
        """Draw the shape on the background using the same style as videos."""
        frame = background.copy()
        
        # Interpolate points for smooth drawing
        interpolated_points = self.interpolate_points(points)
        
        # Draw lines connecting all points
        for i in range(1, len(interpolated_points)):
            cv2.line(frame, interpolated_points[i-1], interpolated_points[i], 255, thickness=3)
        
        return frame
    
    def generate_translated_versions(self, points, num_translations):
        """Generate translated versions of the same shape."""
        translated_versions = []
        
        # Calculate margins as 10% of dimensions
        margin_x = int(self.width * 0.1)
        margin_y = int(self.height * 0.1)
        
        # Calculate the bounding box of the original shape
        points_array = np.array(points)
        min_x = np.min(points_array[:, 0])
        max_x = np.max(points_array[:, 0])
        min_y = np.min(points_array[:, 1])
        max_y = np.max(points_array[:, 1])
        
        # Calculate shape dimensions
        shape_width = max_x - min_x
        shape_height = max_y - min_y
        
        # Calculate maximum translation range to keep shape within bounds
        max_translate_x = self.width - margin_x - max_x
        min_translate_x = margin_x - min_x
        max_translate_y = self.height - margin_y - max_y
        min_translate_y = margin_y - min_y
        
        # Generate translations
        for i in range(num_translations):
            # Generate random translation within bounds
            translate_x = random.randint(min_translate_x, max_translate_x)
            translate_y = random.randint(min_translate_y, max_translate_y)
            
            # Apply translation to all points
            translated_points = []
            for point in points:
                x, y = point
                new_x = x + translate_x
                new_y = y + translate_y
                translated_points.append((new_x, new_y))
            
            translated_versions.append(translated_points)
        
        return translated_versions
    
    def generate_dataset(self):
        """Generate the complete dataset with train/val/test splits."""
        # Get configuration parameters from the specific curve config
        num_samples_per_class = self.curve_config.get('num_samples_per_class', 1000)
        num_translations_per_shape = self.curve_config.get('num_translations_per_shape', 8)
        folder_name = self.curve_config.get('folder_name', 'curve_lines_dataset')
        
        # Define split ratios
        train_ratio = 0.7
        val_ratio = 0.2
        test_ratio = 0.1
        
        # Calculate number of samples for each split
        train_samples = int(num_samples_per_class * train_ratio)
        val_samples = int(num_samples_per_class * val_ratio)
        test_samples = num_samples_per_class - train_samples - val_samples
        
        # Get output directory from config
        base_dir = self.config['output']['base_dir']
        dataset_dir = os.path.join(base_dir, folder_name)
        
        # Create directory structure
        splits = ['train', 'val', 'test']
        classes = ['class0', 'class1']  # class0: straight lines, class1: curves
        
        for split in splits:
            for class_name in classes:
                os.makedirs(os.path.join(dataset_dir, split, class_name), exist_ok=True)
        
        # Generate samples for each class
        for class_idx, class_name in enumerate(classes):
            print(f"Generating {class_name} samples...")
            
            if class_idx == 0:  # Straight lines
                generate_points_func = self.generate_straight_line_points
            else:  # Curves
                generate_points_func = self.generate_curve_points
            
            # Generate base shapes
            base_shapes = []
            for _ in range(num_samples_per_class // num_translations_per_shape):  # num_translations_per_shape translations per shape
                base_shapes.append(generate_points_func())
            
            # Generate all samples with translations
            all_samples = []
            for base_shape in base_shapes:
                translated_shapes = self.generate_translated_versions(base_shape, num_translations_per_shape)
                all_samples.extend(translated_shapes)
            
            # Shuffle samples
            random.shuffle(all_samples)
            
            # Split into train/val/test
            train_samples_list = all_samples[:train_samples]
            val_samples_list = all_samples[train_samples:train_samples + val_samples]
            test_samples_list = all_samples[train_samples + val_samples:]
            
            # Generate images for each split
            for split_name, samples_list in [('train', train_samples_list), 
                                           ('val', val_samples_list), 
                                           ('test', test_samples_list)]:
                print(f"  Generating {split_name} split for {class_name}...")
                
                for i, points in enumerate(tqdm(samples_list, desc=f"{split_name}_{class_name}")):
                    # Get random background
                    background = self.get_random_background()
                    
                    # Draw shape
                    image = self.draw_shape(points, background)
                    
                    # Save image
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{class_name}_{timestamp}_{i:04d}.png"
                    output_path = os.path.join(dataset_dir, split_name, class_name, filename)
                    cv2.imwrite(output_path, image)
        
        print(f"Dataset generated successfully in {dataset_dir}")
        print(f"Train: {train_samples} samples per class")
        print(f"Val: {val_samples} samples per class")
        print(f"Test: {test_samples} samples per class")
        print(f"Number of translations per shape: {num_translations_per_shape}")

def main():
    parser = argparse.ArgumentParser(description='Generate curve/line classification dataset')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get curve_lines_dataset configurations
    curve_configs = config.get('curve_lines_dataset', [])
    if not isinstance(curve_configs, list):
        curve_configs = [curve_configs]
    
    # Generate datasets for each configuration
    total_datasets_generated = 0
    for curve_config in curve_configs:
        folder_name = curve_config.get('folder_name', 'curve_lines_dataset')
        num_samples_per_class = curve_config.get('num_samples_per_class', 1000)
        num_translations_per_shape = curve_config.get('num_translations_per_shape', 8)
        
        print(f"\nGenerating curve lines dataset for folder: {folder_name}")
        print(f"Configuration: num_samples_per_class={num_samples_per_class}, "
              f"num_translations_per_shape={num_translations_per_shape}")
        
        # Initialize generator with specific curve config
        generator = CurveLineGenerator(
            config_path=args.config,
            curve_config=curve_config
        )
        
        # Generate dataset for this configuration
        generator.generate_dataset()
        total_datasets_generated += 1
    
    print(f"\nTotal curve lines datasets generated: {total_datasets_generated}")

if __name__ == "__main__":
    main() 