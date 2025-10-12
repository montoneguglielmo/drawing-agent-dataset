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

class ShapeDatasetGenerator:
    def __init__(self, config_path='config.yaml', shape_config=None):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.width = self.config['image']['width']
        self.height = self.config['image']['height']
        self.margin_x = int(self.width * 0.15)
        self.margin_y = int(self.height * 0.15)
        self.compass_size = min(self.width, self.height) // 5  # Compass space
        
        # Use provided shape_config or fall back to first shape config in config file
        if shape_config is None:
            shape_config = self.config.get('shape_dataset', [{}])[0] if isinstance(self.config.get('shape_dataset'), list) else self.config.get('shape_dataset', {})
        
        self.shape_config = shape_config
        
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
    
    def generate_random_path(self, num_points=None):
        """Generate a random path for the drawing."""
        points = []
        
        # Randomly select number of points from the specified list
        if num_points is None:
            n_points = [2,3,4]
            num_points = random.choice(n_points)
        
        # Calculate margins as 15% of dimensions
        margin_x = self.margin_x
        margin_y = self.margin_y
        
        # Generate all points randomly, avoiding compass boundary
        x_coords = [random.randint(margin_x, self.width - margin_x) for _ in range(num_points)]
        y_coords = [random.randint(self.compass_size + margin_y, self.height - margin_y) for _ in range(num_points)]
        points = list(zip(x_coords, y_coords))
        
        return points
    
    
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
        
        # Use linear interpolation for simple shapes, cubic for more complex ones
        if len(points) <= 10:
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
        
        # Calculate margins as 15% of dimensions
        margin_x = self.margin_x
        margin_y = self.margin_y
        compass_size = self.compass_size
        
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
        # For x: use regular margins
        max_translate_x = self.width - margin_x - max_x
        min_translate_x = margin_x - min_x
        
        # For y: use compass space + regular margin as lower bound
        max_translate_y = self.height - margin_y - max_y
        min_translate_y = compass_size + margin_y - min_y
                
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
        # Get configuration parameters from the specific shape config
        num_samples_per_class = self.shape_config.get('num_samples_per_class', 1000)
        num_translations_per_shape = self.shape_config.get('num_translations_per_shape', 8)
        folder_name = self.shape_config.get('folder_name', 'shape_dataset')
        num_classes = self.shape_config.get('num_classes', 3)
        
        # Define split ratios
        train_ratio = 0.5
        val_ratio = 0.3
        test_ratio = 0.2
        
        # Calculate number of samples for each split
        train_samples = int(num_samples_per_class * train_ratio)
        val_samples = int(num_samples_per_class * val_ratio)
        test_samples = num_samples_per_class - train_samples - val_samples
        
        # Get output directory from config
        base_dir = self.config['output']['base_dir']
        dataset_dir = os.path.join(base_dir, folder_name)
        
        # Create directory structure
        splits = ['train', 'val', 'test']
        classes = [f'class{idx}' for idx in range(num_classes)]  # class0, class1, etc.
        
        for split in splits:
            for class_name in classes:
                os.makedirs(os.path.join(dataset_dir, split, class_name), exist_ok=True)
        
        # Generate samples for each shape class
        for class_idx in range(num_classes):
            class_name = f'class{class_idx}'
            print(f"Generating {class_name} samples for random shapes...")
            
            # Generate ONE base shape per class
            base_shape = self.generate_random_path()
            
            # Generate all samples with translations of the same base shape
            all_samples = self.generate_translated_versions(base_shape, num_samples_per_class)
            
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
        print(f"Number of classes: {num_classes}")
        print(f"Train: {train_samples} samples per class")
        print(f"Val: {val_samples} samples per class")
        print(f"Test: {test_samples} samples per class")
        print(f"Number of translations per shape: {num_translations_per_shape}")

def main():
    parser = argparse.ArgumentParser(description='Generate shape classification dataset')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get shape_dataset configurations
    shape_configs = config.get('shape_dataset', [])
    if not isinstance(shape_configs, list):
        shape_configs = [shape_configs]
    
    # Generate datasets for each configuration
    total_datasets_generated = 0
    for shape_config in shape_configs:
        folder_name = shape_config.get('folder_name', 'shape_dataset')
        num_samples_per_class = shape_config.get('num_samples_per_class', 1000)
        num_classes = shape_config.get('num_classes', 3)
        
        print(f"\nGenerating shape dataset for folder: {folder_name}")
        print(f"Configuration: num_samples_per_class={num_samples_per_class}, "
              f"num_classes={num_classes}")
        
        # Initialize generator with specific shape config
        generator = ShapeDatasetGenerator(
            config_path=args.config,
            shape_config=shape_config
        )
        
        # Generate dataset for this configuration
        generator.generate_dataset()
        total_datasets_generated += 1
    
    print(f"\nTotal shape datasets generated: {total_datasets_generated}")

if __name__ == "__main__":
    main()
