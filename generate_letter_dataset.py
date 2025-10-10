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

class LetterDatasetGenerator:
    def __init__(self, config_path='config.yaml', letter_config=None):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.width = self.config['image']['width']
        self.height = self.config['image']['height']
        self.margin_x = int(self.width * 0.15)
        self.margin_y = int(self.height * 0.15)
        self.upper_margin_y = min(self.width, self.height) // 5  # Compass space
        
        # Use provided letter_config or fall back to first letter config in config file
        if letter_config is None:
            letter_config = self.config.get('letter_dataset', [{}])[0] if isinstance(self.config.get('letter_dataset'), list) else self.config.get('letter_dataset', {})
        
        self.letter_config = letter_config
        
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
    
    def generate_letter_points(self, letter):
        """Generate points that form a specific letter."""
        # Use larger margins for initial generation to ensure translation space
        initial_margin_x = self.margin_x + 3
        initial_margin_y = self.margin_y + 3
        upper_margin_y = self.upper_margin_y
        
        # Define letter size as percentage of image dimensions
        letter_width = int(self.width * 0.4)  # 40% of image width
        letter_height = int(self.height * 0.4)  # 40% of image height
        
        # Center the letter in the available space
        center_x = self.width // 2
        center_y = (self.height + upper_margin_y) // 2
        
        # Generate letter-specific points
        if letter.lower() == 'a':
            return self._generate_a_points(center_x, center_y, letter_width, letter_height)
        elif letter.lower() == 'c':
            return self._generate_c_points(center_x, center_y, letter_width, letter_height)
        elif letter.lower() == 'h':
            return self._generate_h_points(center_x, center_y, letter_width, letter_height)
        elif letter.lower() == 'o':
            return self._generate_o_points(center_x, center_y, letter_width, letter_height)
        elif letter.lower() == 's':
            return self._generate_s_points(center_x, center_y, letter_width, letter_height)
        else:
            # Default to a simple cross for unknown letters
            return self._generate_cross_points(center_x, center_y, letter_width, letter_height)
    
    def _generate_a_points(self, center_x, center_y, width, height):
        """Generate points for letter A."""
        half_width = width // 2
        half_height = height // 2
        
        # A shape: two diagonal lines meeting at top, horizontal line in middle
        points = []
        
        # Left diagonal line
        points.extend([
            (center_x - half_width, center_y + half_height),  # Bottom left
            (center_x, center_y - half_height),  # Top center
        ])
        
        # Right diagonal line  
        points.extend([
            (center_x, center_y - half_height),  # Top center
            (center_x + half_width, center_y + half_height),  # Bottom right
        ])
        
        # Horizontal line
        points.extend([
            (center_x - half_width//2, center_y),  # Left of horizontal
            (center_x + half_width//2, center_y),  # Right of horizontal
        ])
        
        return points
    
    def _generate_c_points(self, center_x, center_y, width, height):
        """Generate points for letter C."""
        half_width = width // 2
        half_height = height // 2
        
        # C shape: curved line
        points = []
        num_points = 20
        
        for i in range(num_points + 1):
            angle = math.pi + (i * math.pi / num_points)  # From 180 to 360 degrees
            x = center_x + int(half_width * math.cos(angle))
            y = center_y + int(half_height * math.sin(angle))
            points.append((x, y))
        
        return points
    
    def _generate_h_points(self, center_x, center_y, width, height):
        """Generate points for letter H."""
        half_width = width // 2
        half_height = height // 2
        
        # H shape: two vertical lines connected by horizontal line
        points = []
        
        # Left vertical line
        points.extend([
            (center_x - half_width, center_y - half_height),  # Top left
            (center_x - half_width, center_y + half_height),  # Bottom left
        ])
        
        # Right vertical line
        points.extend([
            (center_x + half_width, center_y - half_height),  # Top right
            (center_x + half_width, center_y + half_height),  # Bottom right
        ])
        
        # Horizontal connecting line
        points.extend([
            (center_x - half_width, center_y),  # Left of horizontal
            (center_x + half_width, center_y),  # Right of horizontal
        ])
        
        return points
    
    def _generate_o_points(self, center_x, center_y, width, height):
        """Generate points for letter O."""
        half_width = width // 2
        half_height = height // 2
        
        # O shape: oval/circle
        points = []
        num_points = 30
        
        for i in range(num_points):
            angle = (i * 2 * math.pi / num_points)
            x = center_x + int(half_width * math.cos(angle))
            y = center_y + int(half_height * math.sin(angle))
            points.append((x, y))
        
        # Close the circle
        points.append(points[0])
        
        return points
    
    def _generate_s_points(self, center_x, center_y, width, height):
        """Generate points for letter S."""
        half_width = width // 2
        half_height = height // 2
        
        # S shape: curved line with two curves
        points = []
        num_points = 20
        
        for i in range(num_points + 1):
            t = i / num_points
            if t <= 0.5:
                # First half: upper curve
                angle = math.pi + (t * 2 * math.pi)  # From 180 to 360 degrees
                x = center_x + int(half_width * math.cos(angle))
                y = center_y - half_height + int(half_height * t)
            else:
                # Second half: lower curve
                angle = (t - 0.5) * 2 * math.pi  # From 0 to 180 degrees
                x = center_x + int(half_width * math.cos(angle))
                y = center_y + int(half_height * (t - 0.5))
            points.append((x, y))
        
        return points
    
    def _generate_cross_points(self, center_x, center_y, width, height):
        """Generate points for a simple cross (default for unknown letters)."""
        half_width = width // 2
        half_height = height // 2
        
        # Cross shape: vertical and horizontal lines
        points = []
        
        # Vertical line
        points.extend([
            (center_x, center_y - half_height),  # Top
            (center_x, center_y + half_height),  # Bottom
        ])
        
        # Horizontal line
        points.extend([
            (center_x - half_width, center_y),  # Left
            (center_x + half_width, center_y),  # Right
        ])
        
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
        upper_margin_y = self.upper_margin_y
        
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
        min_translate_y = upper_margin_y + margin_y - min_y
                
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
        # Get configuration parameters from the specific letter config
        num_samples_per_class = self.letter_config.get('num_samples_per_class', 1000)
        num_translations_per_shape = self.letter_config.get('num_translations_per_shape', 8)
        folder_name = self.letter_config.get('folder_name', 'letter_dataset')
        letters = self.letter_config.get('letters', ['a', 'c', 'h'])
        
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
        classes = [f'class{idx}' for idx in range(len(letters))]  # class0, class1, etc.
        
        for split in splits:
            for class_name in classes:
                os.makedirs(os.path.join(dataset_dir, split, class_name), exist_ok=True)
        
        # Generate samples for each letter class
        for class_idx, letter in enumerate(letters):
            class_name = f'class{class_idx}'
            print(f"Generating {class_name} samples for letter '{letter}'...")
            
            # Generate base shapes
            base_shapes = []
            for _ in range(num_samples_per_class // num_translations_per_shape):
                base_shapes.append(self.generate_letter_points(letter))
            
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
        print(f"Letters: {letters}")
        print(f"Train: {train_samples} samples per class")
        print(f"Val: {val_samples} samples per class")
        print(f"Test: {test_samples} samples per class")
        print(f"Number of translations per shape: {num_translations_per_shape}")

def main():
    parser = argparse.ArgumentParser(description='Generate letter classification dataset')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get letter_dataset configurations
    letter_configs = config.get('letter_dataset', [])
    if not isinstance(letter_configs, list):
        letter_configs = [letter_configs]
    
    # Generate datasets for each configuration
    total_datasets_generated = 0
    for letter_config in letter_configs:
        folder_name = letter_config.get('folder_name', 'letter_dataset')
        num_samples_per_class = letter_config.get('num_samples_per_class', 1000)
        num_translations_per_shape = letter_config.get('num_translations_per_shape', 8)
        letters = letter_config.get('letters', ['a', 'c', 'h'])
        
        print(f"\nGenerating letter dataset for folder: {folder_name}")
        print(f"Configuration: num_samples_per_class={num_samples_per_class}, "
              f"num_translations_per_shape={num_translations_per_shape}, letters={letters}")
        
        # Initialize generator with specific letter config
        generator = LetterDatasetGenerator(
            config_path=args.config,
            letter_config=letter_config
        )
        
        # Generate dataset for this configuration
        generator.generate_dataset()
        total_datasets_generated += 1
    
    print(f"\nTotal letter datasets generated: {total_datasets_generated}")

if __name__ == "__main__":
    main()
