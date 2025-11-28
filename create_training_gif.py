# create_training_gif.py
"""
Create a GIF from training visualization images.
Combines all images saved during training into an animated GIF.
"""
import os
import glob
from PIL import Image
import argparse

def create_gif(image_dir, output_path, duration=500, loop=0):
    """
    Create a GIF from images in a directory.
    
    Args:
        image_dir: Directory containing training images
        output_path: Output GIF file path
        duration: Frame duration in milliseconds
        loop: Number of loops (0 = infinite)
    """
    # Find all PNG images
    pattern = os.path.join(image_dir, "episode_*.png")
    image_files = sorted(glob.glob(pattern))
    
    if not image_files:
        print(f"❌ No images found in {image_dir}")
        print(f"   Looking for pattern: {pattern}")
        return False
    
    print(f"📸 Found {len(image_files)} images")
    print(f"   First: {os.path.basename(image_files[0])}")
    print(f"   Last:  {os.path.basename(image_files[-1])}")
    
    # Load images
    images = []
    for img_file in image_files:
        try:
            img = Image.open(img_file)
            images.append(img.copy())
        except Exception as e:
            print(f"⚠️  Warning: Could not load {img_file}: {e}")
    
    if not images:
        print("❌ No valid images to create GIF")
        return False
    
    # Create GIF
    print(f"🎬 Creating GIF with {len(images)} frames...")
    try:
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=loop,
            optimize=True
        )
        print(f"✅ GIF created: {output_path}")
        print(f"   Size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
        return True
    except Exception as e:
        print(f"❌ Error creating GIF: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Create training GIF from images")
    parser.add_argument(
        "--model",
        type=str,
        choices=["ppo", "dqn", "both"],
        default="both",
        help="Which model's training images to use"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=500,
        help="Frame duration in milliseconds (default: 500)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output GIF filename (default: auto-generated)"
    )
    
    args = parser.parse_args()
    
    models_to_process = []
    if args.model in ["ppo", "both"]:
        models_to_process.append(("PPO", "visualizations/training/ppo"))
    if args.model in ["dqn", "both"]:
        models_to_process.append(("DQN", "visualizations/training/dqn"))
    
    for model_name, image_dir in models_to_process:
        if not os.path.exists(image_dir):
            print(f"⚠️  Directory not found: {image_dir}")
            print(f"   Make sure you've trained {model_name} with visualization callback")
            continue
        
        if args.output:
            output_path = args.output
        else:
            output_path = f"visualizations/{model_name.lower()}_training.gif"
        
        print(f"\n{'='*60}")
        print(f"Creating {model_name} Training GIF")
        print(f"{'='*60}")
        create_gif(image_dir, output_path, duration=args.duration)

if __name__ == "__main__":
    main()

