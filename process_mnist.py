import numpy as np
import cv2
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
        
        # Load pre-generated backgrounds
        backgrounds_path = os.path.join(self.config['output']['backgrounds'], 'backgrounds.npy')
        if not os.path.exists(backgrounds_path):
            raise FileNotFoundError(f"Backgrounds file not found at {backgrounds_path}. Please run generate_backgrounds.py first.")
        self.backgrounds = np.load(backgrounds_path)
        
    def get_random_background(self):
        """Get a random background from the pre-generated ones."""
        idx = random.randint(0, len(self.backgrounds) - 1)
        background = self.backgrounds[idx]
        
        # Convert to 3-channel image
        background = np.stack([background, background, background], axis=-1)
        # Scale to 0-255 range
        background = (background * 255).astype(np.uint8)
        
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
        
        # Get random background
        background = self.get_random_background()
        
        # Create a copy of the background to draw on
        result = background.copy()
        
        # Draw the MNIST digit on the background
        # We'll use a threshold to determine which pixels to draw
        threshold = 128
        mask = scaled_image > threshold
        result[mask] = 255
        return result

    def process_dataset(self, dataset, name):
        """Process an entire dataset and return images and labels."""
        print(f"Processing {name} set...")
        images = []
        labels = []
        
        # Create directory for sample images using the mnist output path from config
        sample_dir = os.path.join(self.config['output']['mnist'], 'samples')
        os.makedirs(sample_dir, exist_ok=True)
        
        for i, (image, label) in enumerate(tqdm(dataset, desc=f"Processing {name} images")):
            processed_image = self.process_image(image)
            images.append(processed_image)
            labels.append(label)
            
            # Save first 5 images as PNGs for inspection
            if i < 15:
                sample_path = os.path.join(sample_dir, f'{name}_sample_{i}_label_{label}.png')
                cv2.imwrite(sample_path, processed_image)
        
        # Convert to numpy arrays
        images = np.array(images)
        labels = np.array(labels)
        
        return images, labels

def main():
    # Initialize processor
    processor = MNISTProcessor()
    
    # Get output directory from config
    output_dir = processor.config['output']['mnist']
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
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