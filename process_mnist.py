import numpy as np
import cv2
from noise import pnoise2
import yaml
import os
from tqdm import tqdm
import torch
from torchvision import datasets
import random

class MNISTProcessor:
    def __init__(self, config_path='config.yaml'):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.width = self.config['image']['width']
        self.height = self.config['image']['height']
        
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
        
        # Map noise values to 0-255 range
        grey = ((noise + 1) * 127.5).astype(np.uint8)
        
        # Create a 3-channel image with the same grey value in all channels
        background = np.stack([grey, grey, grey], axis=-1)
        
        return background

    def process_image(self, mnist_image):
        """Process a single MNIST image."""
        # Convert PIL Image to numpy array
        if isinstance(mnist_image, torch.Tensor):
            mnist_image = mnist_image.numpy()
        elif hasattr(mnist_image, 'convert'):  # PIL Image
            mnist_image = np.array(mnist_image)
        
        # Scale image to target size
        scaled_image = cv2.resize(mnist_image, (self.width, self.height), 
                                interpolation=cv2.INTER_LINEAR)
        
        # Create background
        background = self.create_smooth_grey_background()
        
        # Convert scaled image to 3 channels
        if len(scaled_image.shape) == 2:
            scaled_image = np.stack([scaled_image] * 3, axis=-1)
        
        # Add the two images together
        combined = scaled_image.astype(np.int32) + background.astype(np.int32)
        
        # Apply threshold (255 is the maximum value for uint8)
        threshold = 255
        result = np.clip(combined, 0, threshold).astype(np.uint8)
        
        return result

    def process_dataset(self, dataset, name):
        """Process an entire dataset and return images and labels."""
        print(f"Processing {name} set...")
        images = []
        labels = []
        
        # Create directory for sample images
        sample_dir = os.path.join('processed_mnist', 'samples')
        os.makedirs(sample_dir, exist_ok=True)
        
        for i, (image, label) in enumerate(tqdm(dataset, desc=f"Processing {name} images")):
            processed_image = self.process_image(image)
            images.append(processed_image)
            labels.append(label)
            
            # Save first 5 images as PNGs for inspection
            if i < 5:
                sample_path = os.path.join(sample_dir, f'{name}_sample_{i}_label_{label}.png')
                cv2.imwrite(sample_path, processed_image)
        
        # Convert to numpy arrays
        images = np.array(images)
        labels = np.array(labels)
        
        return images, labels

def main():
    # Create output directory
    output_dir = 'processed_mnist'
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize processor
    processor = MNISTProcessor()
    
    # Load MNIST datasets
    print("Loading MNIST datasets...")
    train_dataset = datasets.MNIST(root='./data', train=True, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True)
    
    # Process datasets
    train_images, train_labels = processor.process_dataset(train_dataset, "training")
    test_images, test_labels = processor.process_dataset(test_dataset, "test")
    
    # Save processed datasets
    print("Saving processed datasets...")
    np.save(os.path.join(output_dir, 'train_images.npy'), train_images)
    np.save(os.path.join(output_dir, 'train_labels.npy'), train_labels)
    np.save(os.path.join(output_dir, 'test_images.npy'), test_images)
    np.save(os.path.join(output_dir, 'test_labels.npy'), test_labels)
    
    print(f"Processed datasets saved to {output_dir}")
    print(f"Training set shape: {train_images.shape}")
    print(f"Test set shape: {test_images.shape}")

if __name__ == "__main__":
    main()