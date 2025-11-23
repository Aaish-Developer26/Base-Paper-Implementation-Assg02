# expand_test_set.py
from pathlib import Path
import shutil

def expand_test_set():
    """Copy some training images to test set (for demo only)"""
    
    data_real = Path("data/real")
    data_fake = Path("data/fake")
    test_real = Path("data/test/real")
    test_fake = Path("data/test/fake")
    
    # Get training images
    train_real = list(data_real.glob("*.jpg"))[:30]
    train_fake = list(data_fake.glob("*.jpg"))[:30]
    
    # Copy last 20 to test (assuming first 10 were already in test)
    print("Expanding test set...")
    
    for i, img in enumerate(train_real[10:30], start=11):
        dest = test_real / f"test_real_{i:03d}.jpg"
        shutil.copy(img, dest)
        print(f"  Copied: {img.name} -> {dest.name}")
    
    for i, img in enumerate(train_fake[10:30], start=11):
        dest = test_fake / f"test_fake_{i:03d}.jpg"
        shutil.copy(img, dest)
        print(f"  Copied: {img.name} -> {dest.name}")
    
    print(f"\n✓ Test set expanded!")
    print(f"  Real: {len(list(test_real.glob('*')))} images")
    print(f"  Fake: {len(list(test_fake.glob('*')))} images")

if __name__ == "__main__":
    expand_test_set()