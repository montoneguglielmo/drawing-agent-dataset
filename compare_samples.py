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

def get_random_shape_image(shape_dir):
    # Get a random shape image from train, val, or test directory
    train_dir = os.path.join(shape_dir, 'train')
    val_dir = os.path.join(shape_dir, 'val')
    test_dir = os.path.join(shape_dir, 'test')
    
    # Randomly choose between train, val, and test directories
    chosen_dir = random.choice([train_dir, val_dir, test_dir])
    
    # Randomly choose a class directory (class0, class1, etc. for different shapes)
    class_dirs = [d for d in os.listdir(chosen_dir) if d.startswith('class')]
    if not class_dirs:
        raise FileNotFoundError(f"No class directories found in {chosen_dir}")
    
    class_dir = os.path.join(chosen_dir, random.choice(class_dirs))
    
    # Get a random shape image from the chosen class directory
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

def create_video_curve_comparison(video_frame, curve_line_image, output_path):
    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Display video frame
    ax1.imshow(video_frame)
    ax1.set_title('Random Video Frame')
    ax1.axis('off')
    
    # Display curve/line image
    ax2.imshow(curve_line_image, cmap='gray')
    ax2.set_title('Random Curve/Line Image')
    ax2.axis('off')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def create_video_shape_comparison(video_frame, shape_image, output_path):
    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Display video frame
    ax1.imshow(video_frame)
    ax1.set_title('Random Video Frame')
    ax1.axis('off')
    
    # Display shape image
    ax2.imshow(shape_image, cmap='gray')
    ax2.set_title('Random Shape Image')
    ax2.axis('off')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def get_frames_with_interval(video_dir, frame_interval=4, num_frames=4):
    """
    Extract frames from a random video with specified interval between them.
    For example, with frame_interval=4 and num_frames=4, we get frames: 4, 8, 12, 16
    """
    # Get a random video file
    video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    if not video_files:
        raise FileNotFoundError("No video files found in the specified directory")
    
    video_path = os.path.join(video_dir, random.choice(video_files))
    cap = cv2.VideoCapture(video_path)
    
    # Get total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame positions
    # Start from a random frame to get more variety
    max_start_frame = total_frames - (num_frames * frame_interval)
    if max_start_frame <= 0:
        # If video is too short, start from frame 1
        start_frame = 1
    else:
        start_frame = random.randint(1, max_start_frame)
    
    frame_positions = [start_frame + i * frame_interval for i in range(num_frames)]
    
    # Check if we have enough frames
    if max(frame_positions) >= total_frames:
        # Adjust if video is too short
        max_possible = total_frames - 1
        frame_positions = [max(1, max_possible - (num_frames - 1 - i) * frame_interval) for i in range(num_frames)]
    
    frames = []
    for frame_pos in frame_positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        else:
            raise ValueError(f"Failed to read frame {frame_pos} from video")
    
    cap.release()
    return frames, frame_positions

def create_frame_interval_comparison(frames, frame_positions, output_path):
    """Create a comparison image showing 4 frames with intervals"""
    # Create a figure with four subplots in a 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()  # Flatten the 2D array for easier indexing
    
    for i, (frame, frame_pos) in enumerate(zip(frames, frame_positions)):
        axes[i].imshow(frame)
        axes[i].set_title(f'Frame {frame_pos}')
        axes[i].axis('off')
    
    # Add overall title
    fig.suptitle(f'Video Frames with {frame_positions[1] - frame_positions[0]} Frame Intervals', fontsize=16)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    config = load_config()
    
    # Define paths using base directory
    base_dir = config['output']['base_dir']
    
    # Get video directories from config
    video_configs = config['video']
    video_dirs = []
    for video_config in video_configs:
        video_folder = video_config['folder_name']
        video_dir = os.path.join(base_dir, video_folder)
        if os.path.exists(video_dir):
            video_dirs.append(video_dir)
    
    if not video_dirs:
        raise FileNotFoundError("No video directories found. Please check your config and ensure videos have been generated.")
    
    # Use the first available video directory for frame interval comparisons
    primary_video_dir = video_dirs[0]
    
    # Get curve lines directory from config
    curve_lines_configs = config['curve_lines_dataset']
    curve_lines_dirs = []
    for curve_config in curve_lines_configs:
        curve_folder = curve_config['folder_name']
        curve_dir = os.path.join(base_dir, curve_folder)
        if os.path.exists(curve_dir):
            curve_lines_dirs.append(curve_dir)
    
    if not curve_lines_dirs:
        raise FileNotFoundError("No curve lines directories found. Please check your config and ensure curve lines dataset has been generated.")
    
    # Use the first available curve lines directory
    primary_curve_lines_dir = curve_lines_dirs[0]
    
    # Get shape dataset directories from config
    shape_configs = config['shape_dataset']
    shape_dirs = []
    for shape_config in shape_configs:
        shape_folder = shape_config['folder_name']
        shape_dir = os.path.join(base_dir, shape_folder)
        if os.path.exists(shape_dir):
            shape_dirs.append(shape_dir)
    
    # Use the first available shape directory
    primary_shape_dir = shape_dirs[0] if shape_dirs else None
    
    mnist_dir = os.path.join(base_dir, 'mnist')  # Updated to point to mnist root directory
    output_dir = os.path.join(base_dir, 'comparisons')
    frame_interval_dir = os.path.join(base_dir, 'comparisons/frame_intervals')
    shape_comparison_dir = os.path.join(base_dir, 'comparisons/video_shape')
    
    # Check if MNIST directory exists
    mnist_available = os.path.exists(mnist_dir)
    if not mnist_available:
        print("MNIST directory not found. Will only generate comparisons between video frames and curve lines.")
    
    # Check if shape dataset directory exists
    shape_available = primary_shape_dir is not None
    if not shape_available:
        print("Shape dataset directory not found. Will not generate video-shape comparisons.")
    
    # Create output directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(frame_interval_dir, exist_ok=True)
    if shape_available:
        os.makedirs(shape_comparison_dir, exist_ok=True)
    
    # Generate frame interval comparisons
    print("Generating frame interval comparisons...")
    for i in range(5):  # Generate 5 frame interval comparisons
        try:
            frames, frame_positions = get_frames_with_interval(primary_video_dir, frame_interval=4, num_frames=4)
            output_path = os.path.join(frame_interval_dir, f'frame_interval_{i+1}.png')
            create_frame_interval_comparison(frames, frame_positions, output_path)
            print(f"Created frame interval comparison {i+1} with frames: {frame_positions}")
            
        except Exception as e:
            print(f"Error creating frame interval comparison {i+1}: {str(e)}")
    
    # Generate original comparisons
    print("\nGenerating original comparisons...")
    for i in range(10):
        try:
            video_frame = get_random_video_frame(primary_video_dir)
            curve_line_image = get_random_curve_line_image(primary_curve_lines_dir)
            
            output_path = os.path.join(output_dir, f'comparison_{i+1}.png')
            
            if mnist_available:
                # Generate comparison with MNIST, video frames, and curve lines
                mnist_image = get_random_mnist_image(mnist_dir)
                create_comparison(video_frame, mnist_image, curve_line_image, output_path)
                print(f"Created comparison {i+1} (with MNIST)")
            else:
                # Generate comparison with only video frames and curve lines
                create_video_curve_comparison(video_frame, curve_line_image, output_path)
                print(f"Created comparison {i+1} (video + curve lines only)")
            
        except Exception as e:
            print(f"Error creating comparison {i+1}: {str(e)}")
    
    # Generate video-shape comparisons
    if shape_available:
        print("\nGenerating video-shape comparisons...")
        for i in range(10):
            try:
                video_frame = get_random_video_frame(primary_video_dir)
                shape_image = get_random_shape_image(primary_shape_dir)
                
                output_path = os.path.join(shape_comparison_dir, f'video_shape_comparison_{i+1}.png')
                create_video_shape_comparison(video_frame, shape_image, output_path)
                print(f"Created video-shape comparison {i+1}")
                
            except Exception as e:
                print(f"Error creating video-shape comparison {i+1}: {str(e)}")

if __name__ == "__main__":
    main() 