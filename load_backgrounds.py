import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def load_backgrounds(file_path):
    """Load backgrounds from .npy file."""
    print(f"Loading backgrounds from {file_path}...")
    backgrounds = np.load(file_path)
    print(f"Loaded {len(backgrounds)} backgrounds with shape {backgrounds.shape}")
    return backgrounds

def display_backgrounds(backgrounds, num_samples=5):
    """Display a few sample backgrounds."""
    # Create a figure with subplots
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    
    # Display random samples
    indices = np.random.choice(len(backgrounds), num_samples, replace=False)
    for i, idx in enumerate(indices):
        axes[i].imshow(backgrounds[idx], cmap='gray')
        axes[i].set_title(f'Background {idx}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    # Path to the backgrounds file
    file_path = "../datasets/drawing-agent-test/mnist/backgrounds/backgrounds.npy"
    
    try:
        # Load backgrounds
        backgrounds = load_backgrounds(file_path)
        
        # Display some samples
        display_backgrounds(backgrounds)
        
    except FileNotFoundError:
        print(f"Error: Could not find the backgrounds file at {file_path}")
        print("Please make sure the file exists and the path is correct.")
    except Exception as e:
        print(f"Error loading backgrounds: {str(e)}")

if __name__ == "__main__":
    main() 