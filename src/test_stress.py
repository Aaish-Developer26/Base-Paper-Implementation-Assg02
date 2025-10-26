"""
Create heavily compressed/degraded test images to stress test the detector
"""

from PIL import Image
from pathlib import Path


def create_stress_test():
    """Create heavily compressed/degraded test images"""
    
    # Source directories
    test_real = Path("data/test/real")
    test_fake = Path("data/test/fake")
    
    # Create stress test directories
    stress_real = Path("data/test_stress/real")
    stress_fake = Path("data/test_stress/fake")
    stress_real.mkdir(parents=True, exist_ok=True)
    stress_fake.mkdir(parents=True, exist_ok=True)
    
    print("Creating stressed test set...")
    print("=" * 60)
    
    # Check if source directories exist
    if not test_real.exists() or not test_fake.exists():
        print("Error: data/test/real or data/test/fake not found!")
        print("Please create these directories and add test images first.")
        return
    
    real_count = 0
    fake_count = 0
    
    # Process real images
    print("\nProcessing REAL images (heavy compression)...")
    for img_path in test_real.glob("*"):
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            try:
                img = Image.open(img_path).convert('RGB')
                
                # Heavy JPEG compression (quality 50)
                output_path = stress_real / f"{img_path.stem}.jpg"
                img.save(output_path, "JPEG", quality=50)
                
                print(f"  ✓ {img_path.name} -> JPEG Q=50")
                real_count += 1
            except Exception as e:
                print(f"  ✗ Error processing {img_path.name}: {e}")
    
    # Process fake images
    print("\nProcessing FAKE images (heavy compression + resize)...")
    for img_path in test_fake.glob("*"):
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            try:
                img = Image.open(img_path).convert('RGB')
                
                # Resize to 60% (more aggressive)
                w, h = img.size
                new_w, new_h = int(w * 0.6), int(h * 0.6)
                img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                
                # Heavy compression
                output_path = stress_fake / f"{img_path.stem}.jpg"
                img_resized.save(output_path, "JPEG", quality=50)
                
                print(f"  ✓ {img_path.name} -> {new_w}x{new_h}, JPEG Q=50")
                fake_count += 1
            except Exception as e:
                print(f"  ✗ Error processing {img_path.name}: {e}")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"✓ Created {real_count} stressed real images in: {stress_real}")
    print(f"✓ Created {fake_count} stressed fake images in: {stress_fake}")
    print(f"\nTotal stressed test images: {real_count + fake_count}")
    
    if real_count == 0 or fake_count == 0:
        print("\n⚠️  WARNING: No images created!")
        print("   Make sure data/test/real/ and data/test/fake/ contain images.")
    else:
        print("\n✓ Stress test set ready!")
        print("\nNext steps:")
        print("1. Update your eval.py to use data/test_stress")
        print("2. Run: python -m src.eval")


def create_extreme_stress_test():
    """Create EXTREME stress test (for demo purposes)"""
    
    test_real = Path("data/test/real")
    test_fake = Path("data/test/fake")
    
    extreme_real = Path("data/test_extreme/real")
    extreme_fake = Path("data/test_extreme/fake")
    extreme_real.mkdir(parents=True, exist_ok=True)
    extreme_fake.mkdir(parents=True, exist_ok=True)
    
    print("\nCreating EXTREME stress test set...")
    print("=" * 60)
    
    if not test_real.exists() or not test_fake.exists():
        print("Error: data/test/real or data/test/fake not found!")
        return
    
    real_count = 0
    fake_count = 0
    
    # Process real images
    print("\nProcessing REAL images (extreme degradation)...")
    for img_path in test_real.glob("*"):
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            try:
                img = Image.open(img_path).convert('RGB')
                
                # Resize to 40%
                w, h = img.size
                img = img.resize((int(w*0.4), int(h*0.4)), Image.Resampling.LANCZOS)
                
                # Extreme JPEG compression (quality 30)
                output_path = extreme_real / f"{img_path.stem}.jpg"
                img.save(output_path, "JPEG", quality=30)
                
                print(f"  ✓ {img_path.name} -> 40% size, JPEG Q=30")
                real_count += 1
            except Exception as e:
                print(f"  ✗ Error: {e}")
    
    # Process fake images
    print("\nProcessing FAKE images (extreme degradation)...")
    for img_path in test_fake.glob("*"):
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            try:
                img = Image.open(img_path).convert('RGB')
                
                # Resize to 40%
                w, h = img.size
                img = img.resize((int(w*0.4), int(h*0.4)), Image.Resampling.LANCZOS)
                
                # Extreme compression
                output_path = extreme_fake / f"{img_path.stem}.jpg"
                img.save(output_path, "JPEG", quality=30)
                
                print(f"  ✓ {img_path.name} -> 40% size, JPEG Q=30")
                fake_count += 1
            except Exception as e:
                print(f"  ✗ Error: {e}")
    
    print("\n" + "=" * 60)
    print(f"✓ Created {real_count} extreme real images")
    print(f"✓ Created {fake_count} extreme fake images")
    print("=" * 60)


def main():
    import sys
    
    print("=" * 60)
    print("STRESS TEST IMAGE GENERATOR")
    print("=" * 60)
    
    if len(sys.argv) > 1 and sys.argv[1] == "--extreme":
        create_extreme_stress_test()
    else:
        create_stress_test()
    
    print("\n💡 TIP: Run with --extreme for even harder test set:")
    print("   python test_stress.py --extreme")


if __name__ == "__main__":
    main()