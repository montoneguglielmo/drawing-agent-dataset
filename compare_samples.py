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
    # Get a random MNIST image from train, val, or test directory
    train_dir = os.path.join(mnist_dir, 'train')
    val_dir = os.path.join(mnist_dir, 'val')
    test_dir = os.path.join(mnist_dir, 'test')
    
    # Randomly choose between train, val, and test directories
    chosen_dir = random.choice([train_dir, val_dir, test_dir])
    
    # Randomly choose a class directory
    class_dirs = [d for d in os.listdir(chosen_dir) if d.startswith('class')]
    if not class_dirs:
        raise FileNotFoundError(f"No class directories found in {chosen_dir}")
    
    class_dir = os.path.join(chosen_dir, random.choice(class_dirs))
    
    # Get a random MNIST image from the chosen class directory
    image_files = [f for f in os.listdir(class_dir) if f.endswith('.png')]
    if not image_files:
        raise FileNotFoundError(f"No image files found in {class_dir}")
    
    image_path = os.path.join(class_dir, random.choice(image_files))
    image = Image.open(image_path)
    return np.array(image)

def get_random_curve_line_image(curve_lines_dir):
    # Get a random curve/line image from train, val, or test directory
    train_dir = os.path.join(curve_lines_dir, 'train')
    val_dir = os.path.join(curve_lines_dir, 'val')
    test_dir = os.path.join(curve_lines_dir, 'test')
    
    # Randomly choose between train, val, and test directories
    chosen_dir = random.choice([train_dir, val_dir, test_dir])
    
    # Randomly choose a class directory (class0 for straight lines, class1 for curves)
    class_dirs = [d for d in os.listdir(chosen_dir) if d.startswith('class')]
    if not class_dirs:
        raise FileNotFoundError(f"No class directories found in {chosen_dir}")
    
    class_dir = os.path.join(chosen_dir, random.choice(class_dirs))
    
    # Get a random curve/line image from the chosen class directory
    image_files = [f for f in os.listdir(class_dir) if f.endswith('.png')]
    if not image_files:
        raise FileNotFoundError(f"No image files found in {class_dir}")
    
    image_path = os.path.join(class_dir, random.choice(image_files))
    image = Image.open(image_path)
    return np.array(image)

def create_comparison(video_frame, mnist_image, curve_line_image, output_path):
    # Create a figure with three subplots side by side
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Display video frame
    ax1.imshow(video_frame)
    ax1.set_title('Random Video Frame')
    ax1.axis('off')
    
    # Display MNIST image
    ax2.imshow(mnist_image, cmap='gray')
    ax2.set_title('Random MNIST Image')
    ax2.axis('off')
    
    # Display curve/line image
    ax3.imshow(curve_line_image, cmap='gray')
    ax3.set_title('Random Curve/Line Image')
    ax3.axis('off')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    config = load_config()
    
    # Define paths using base directory
    base_dir = config['output']['base_dir']
    video_dir = os.path.join(base_dir, 'videos')
    mnist_dir = os.path.join(base_dir, 'mnist')  # Updated to point to mnist root directory
    curve_lines_dir = os.path.join(base_dir, 'curve_lines_dataset')
    output_dir = os.path.join(base_dir, 'comparisons')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate 10 comparisons
    for i in range(10):
        try:
            video_frame = get_random_video_frame(video_dir)
            mnist_image = get_random_mnist_image(mnist_dir)
            curve_line_image = get_random_curve_line_image(curve_lines_dir)
            
            output_path = os.path.join(output_dir, f'comparison_{i+1}.png')
            create_comparison(video_frame, mnist_image, curve_line_image, output_path)
            print(f"Created comparison {i+1}")
            
        except Exception as e:
            print(f"Error creating comparison {i+1}: {str(e)}")

if __name__ == "__main__":
    main() 