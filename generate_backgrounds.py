import numpy as np
from noise import pnoise2
import yaml
import os
from tqdm import tqdm

def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def generate_perlin_noise(width, height, scale, octaves, persistence, lacunarity):
    """Generate a Perlin noise background."""
    background = np.zeros((height, width))
    for y in range(height):
        for x in range(width):
            background[y][x] = pnoise2(x/scale, 
                                     y/scale, 
                                     octaves=octaves, 
                                     persistence=persistence, 
                                     lacunarity=lacunarity)
    
    # Normalize to 0-1 range
    background = (background - background.min()) / (background.max() - background.min())
    return background

def generate_backgrounds(config):
    """Generate multiple background images with random parameters."""
    width = config['image']['width']
    height = config['image']['height']
    bg_config = config['background']
    num_backgrounds = bg_config['num_backgrounds']
    
    backgrounds = []
    for _ in tqdm(range(num_backgrounds), desc="Generating backgrounds"):
        # Randomize parameters within config ranges
        scale = np.random.uniform(bg_config['scale']['min'], bg_config['scale']['max'])
        octaves = int(np.random.uniform(bg_config['octaves']['min'], bg_config['octaves']['max']))
        persistence = np.random.uniform(bg_config['persistence']['min'], bg_config['persistence']['max'])
        lacunarity = np.random.uniform(bg_config['lacunarity']['min'], bg_config['lacunarity']['max'])
        
        # Generate background
        background = generate_perlin_noise(width, height, scale, octaves, persistence, lacunarity)
        backgrounds.append(background)
    
    return np.array(backgrounds)

def main():
    # Load configuration
    config = load_config()
    
    # Get output directory from config
    output_dir = os.path.join(config['output']['base_dir'], 'backgrounds')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate backgrounds
    print(f"Generating {config['background']['num_backgrounds']} backgrounds...")
    backgrounds = generate_backgrounds(config)
    
    # Save backgrounds
    output_path = os.path.join(output_dir, 'backgrounds.npy')
    np.save(output_path, backgrounds)
    print(f"Saved {len(backgrounds)} backgrounds to {output_path}")

if __name__ == "__main__":
    main() 