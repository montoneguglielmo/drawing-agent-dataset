import numpy as np
import cv2
import yaml
import os
from tqdm import tqdm
import torch
from torchvision import datasets
import random
from sklearn.model_selection import train_test_split

class MNISTProcessor:
    def __init__(self, config_path='config.yaml'):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.width = self.config['image']['width']
        self.height = self.config['image']['height']
        
        # Load pre-generated backgrounds
        backgrounds_path = os.path.join(self.config['output']['base_dir'], 'backgrounds', 'backgrounds.npy')
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

    def create_class_directories(self, base_dir):
        """Create directories for each class (0-9) within the given base directory."""
        for i in range(10):
            class_dir = os.path.join(base_dir, f'class{i}')
            os.makedirs(class_dir, exist_ok=True)

    def process_dataset(self, dataset, name):
        """Process an entire dataset and return images and labels."""
        print(f"Processing {name} set...")
        images = []
        labels = []
        
        # Create directories for images
        mnist_dir = os.path.join(self.config['output']['base_dir'], 'mnist')
        train_dir = os.path.join(mnist_dir, 'train')
        val_dir = os.path.join(mnist_dir, 'val')
        test_dir = os.path.join(mnist_dir, 'test')
        
        # Create base directories
        if name == "training":
            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(val_dir, exist_ok=True)
            self.create_class_directories(train_dir)
            self.create_class_directories(val_dir)
        else:  # test
            os.makedirs(test_dir, exist_ok=True)
            self.create_class_directories(test_dir)
        
        for i, (image, label) in enumerate(tqdm(dataset, desc=f"Processing {name} images")):
            processed_image = self.process_image(image)
            images.append(processed_image)
            labels.append(label)
            
            # Determine output directory based on the dataset type
            if name == "training":
                # For training set, we'll split into train/val later
                continue
            else:  # test
                output_dir = os.path.join(test_dir, f'class{label}')
                image_path = os.path.join(output_dir, f'image_{i:05d}.png')
                cv2.imwrite(image_path, processed_image)
        
        # Convert to numpy arrays
        images = np.array(images)
        labels = np.array(labels)
        
        # Split training set into train and validation
        if name == "training":
            train_images, val_images, train_labels, val_labels = train_test_split(
                images, labels, test_size=0.2, random_state=42, stratify=labels
            )
            
            # Save train images
            for i, (image, label) in enumerate(zip(train_images, train_labels)):
                output_dir = os.path.join(train_dir, f'class{label}')
                image_path = os.path.join(output_dir, f'image_{i:05d}.png')
                cv2.imwrite(image_path, image)
            
            # Save validation images
            for i, (image, label) in enumerate(zip(val_images, val_labels)):
                output_dir = os.path.join(val_dir, f'class{label}')
                image_path = os.path.join(output_dir, f'image_{i:05d}.png')
                cv2.imwrite(image_path, image)
            
            return train_images, train_labels, val_images, val_labels
        
        # For test set, return the same format but with None for validation data
        return images, labels, None, None

def main():
    # Initialize processor
    processor = MNISTProcessor()
    
    # Get output directory from config
    output_dir = os.path.join(processor.config['output']['base_dir'], 'mnist')
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load MNIST datasets
    print("Loading MNIST datasets...")
    train_dataset = datasets.MNIST(root='./data', train=True, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True)
    
    # Process datasets
    train_images, train_labels, val_images, val_labels = processor.process_dataset(train_dataset, "training")
    test_images, test_labels, _, _ = processor.process_dataset(test_dataset, "test")
    
    # Save processed datasets
    print("Saving processed datasets...")
    np.save(os.path.join(output_dir, 'train_images.npy'), train_images)
    np.save(os.path.join(output_dir, 'train_labels.npy'), train_labels)
    np.save(os.path.join(output_dir, 'val_images.npy'), val_images)
    np.save(os.path.join(output_dir, 'val_labels.npy'), val_labels)
    np.save(os.path.join(output_dir, 'test_images.npy'), test_images)
    np.save(os.path.join(output_dir, 'test_labels.npy'), test_labels)
    
    print(f"Processed datasets saved to {output_dir}")
    print(f"Training set shape: {train_images.shape}")
    print(f"Validation set shape: {val_images.shape}")
    print(f"Test set shape: {test_images.shape}")

if __name__ == "__main__":
    main()