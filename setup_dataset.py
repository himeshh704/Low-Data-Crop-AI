import os
import random
from PIL import Image
import numpy as np
from config import cfg

def generate_synthetic_data(samples_per_class, target_dir):
    """Generates synthetic crop data to simulate PlantVillage for immediate end-to-end execution."""
    print(f"Generating synthetic data in {target_dir}...")
    os.makedirs(target_dir, exist_ok=True)
    
    colors = {
        "Healthy": [34, 139, 34], # Green
        "Stressed": [218, 165, 32], # Yellow-ish
        "Diseased": [139, 69, 19] # Brown-ish
    }
    
    for cls_name in cfg.CLASSES:
        cls_dir = os.path.join(target_dir, cls_name)
        os.makedirs(cls_dir, exist_ok=True)
        
        base_color = colors[cls_name]
        
        for i in range(samples_per_class):
            img_arr = np.zeros((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, 3), dtype=np.uint8)
            noise = np.random.randint(-20, 20, (cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, 3))
            img_arr[:, :] = np.clip(np.array(base_color) + noise, 0, 255)
            
            for _ in range(5):
                x, y = random.randint(0, cfg.IMAGE_SIZE-1), random.randint(0, cfg.IMAGE_SIZE-1)
                r = random.randint(10, 40)
                
                if cls_name == "Diseased":
                    spot_color = [30, 30, 30]
                elif cls_name == "Stressed":
                    spot_color = [255, 255, 100]
                else:
                    spot_color = [0, 200, 0]
                    
                img_arr[max(0, x-r):min(cfg.IMAGE_SIZE, x+r), max(0, y-r):min(cfg.IMAGE_SIZE, y+r)] = spot_color
            
            img = Image.fromarray(img_arr)
            img.save(os.path.join(cls_dir, f"simulated_{i}.jpg"))
            
    print(f"Created {samples_per_class} per class in {target_dir}.")

if __name__ == "__main__":
    generate_synthetic_data(cfg.SAMPLES_PER_CLASS_LABELED, cfg.FEW_SHOT_DIR)
    generate_synthetic_data(cfg.SAMPLES_PER_CLASS_UNLABELED, cfg.UNLABELED_DIR)
    print("\nDataset setup complete! You can now run train.py")
