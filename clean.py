import yaml
import os
import shutil
import sys

def get_confirmation():
    # Load config to get the base directory
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    base_dir = config['output']['base_dir']
    
    print("\nWARNING: This operation will delete all generated data including:")
    print("- MNIST dataset files")
    print("- Generated videos")
    print("- Comparison images")
    print("- Background images")
    print("- Masked videos")
    print("- Curve lines dataset")
    print("- Configuration files")
    print(f"\nFiles will be deleted from: {base_dir}")
    print("\nThis operation cannot be undone!")
    response = input("\nAre you sure you want to proceed? (yes/no): ").lower()
    return response == 'yes'

def clean_directories():
    print("Starting clean operation...")
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    base_dir = config['output']['base_dir']
    print('Base directory:', base_dir)
    
    # Remove all subdirectories and their contents
    for subdir in ['mnist', 'videos', 'comparisons', 'backgrounds', 'masked_videos_examples', 'curve_lines_dataset']:
        dir_path = os.path.join(base_dir, subdir)
        print('Checking directory:', dir_path)
        if os.path.exists(dir_path):
            print('Removing directory:', dir_path)
            try:
                shutil.rmtree(dir_path)
                print('Successfully removed:', dir_path)
            except Exception as e:
                print('Error removing directory:', dir_path)
                print('Error:', str(e))
    
    # Remove config file
    config_path = os.path.join(base_dir, 'config.yaml')
    if os.path.exists(config_path):
        try:
            os.remove(config_path)
            print('Removed config file:', config_path)
        except Exception as e:
            print('Error removing config file:', config_path)
            print('Error:', str(e))
    
    # Remove base directory if empty
    if os.path.exists(base_dir) and not os.listdir(base_dir):
        try:
            os.rmdir(base_dir)
            print('Removed empty base directory:', base_dir)
        except Exception as e:
            print('Error removing base directory:', base_dir)
            print('Error:', str(e))

if __name__ == "__main__":
    if get_confirmation():
        clean_directories()
    else:
        print("Operation cancelled.") 