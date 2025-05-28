import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import yaml

def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def get_random_video_frame(video_dir):
    # Get a random video file
    video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    if not video_files:
        raise FileNotFoundError("No video files found in the specified directory")
    
    video_path = os.path.join(video_dir, random.choice(video_files))
    cap = cv2.VideoCapture(video_path)
    
    # Get total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Select a random frame
    random_frame = random.randint(0, total_frames - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise ValueError("Failed to read frame from video")
    
    # Convert BGR to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame

def get_random_mnist_image(mnist_dir):
    # Get a random MNIST image
    image_files = [f for f in os.listdir(mnist_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        raise FileNotFoundError("No image files found in the specified directory")
    
    image_path = os.path.join(mnist_dir, random.choice(image_files))
    image = Image.open(image_path)
    return np.array(image)

def create_comparison(video_frame, mnist_image, output_path):
    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Display video frame
    ax1.imshow(video_frame)
    ax1.set_title('Random Video Frame')
    ax1.axis('off')
    
    # Display MNIST image
    ax2.imshow(mnist_image, cmap='gray')
    ax2.set_title('Random MNIST Image')
    ax2.axis('off')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    config = load_config()
    
    # Define paths using base directory
    base_dir = config['output']['base_dir']
    video_dir = os.path.join(base_dir, 'videos')
    mnist_dir = os.path.join(base_dir, 'mnist', 'samples')
    output_dir = os.path.join(base_dir, 'comparisons')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate 5 comparisons
    for i in range(10):
        try:
            video_frame = get_random_video_frame(video_dir)
            mnist_image = get_random_mnist_image(mnist_dir)
            
            output_path = os.path.join(output_dir, f'comparison_{i+1}.png')
            create_comparison(video_frame, mnist_image, output_path)
            print(f"Created comparison {i+1}")
            
        except Exception as e:
            print(f"Error creating comparison {i+1}: {str(e)}")

if __name__ == "__main__":
    main() 