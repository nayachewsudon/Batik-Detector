import os
import cv2
import numpy as np
from pathlib import Path
import random

class ImageAugmenter:
    """
    Augments images with various transformations to increase dataset size
    and improve model generalization.
    """
    
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
    
    def resize_and_normalize(self, img):
        """Resize image and normalize pixel values to [0, 1]"""
        img = cv2.resize(img, self.target_size)
        img = img.astype(np.float32) / 255.0
        return img
    
    def random_rotation(self, img, max_angle=30):
        """Rotate image by random angle"""
        angle = random.uniform(-max_angle, max_angle)
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, matrix, (w, h), 
                                 borderMode=cv2.BORDER_REFLECT)
        return rotated
    
    def random_flip(self, img):
        """Randomly flip image horizontally or vertically"""
        flip_type = random.choice([-1, 0, 1])  # -1=both, 0=vertical, 1=horizontal
        if flip_type >= 0:
            return cv2.flip(img, flip_type)
        return img
    
    def random_brightness(self, img, factor_range=(0.7, 1.3)):
        """Adjust brightness randomly"""
        factor = random.uniform(*factor_range)
        img = img.astype(np.float32)
        img = np.clip(img * factor, 0, 255)
        return img.astype(np.uint8)
    
    def random_crop(self, img, crop_factor=0.8):
        """Randomly crop and resize back to original size"""
        h, w = img.shape[:2]
        new_h, new_w = int(h * crop_factor), int(w * crop_factor)
        
        top = random.randint(0, h - new_h)
        left = random.randint(0, w - new_w)
        
        cropped = img[top:top+new_h, left:left+new_w]
        return cv2.resize(cropped, (w, h))
    
    def color_jitter(self, img, hue_shift=20, sat_shift=30, val_shift=30):
        """Apply random color jittering in HSV space"""
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # Random shifts
        img_hsv[:, :, 0] += random.uniform(-hue_shift, hue_shift)
        img_hsv[:, :, 1] += random.uniform(-sat_shift, sat_shift)
        img_hsv[:, :, 2] += random.uniform(-val_shift, val_shift)
        
        # Clip values
        img_hsv[:, :, 0] = np.clip(img_hsv[:, :, 0], 0, 179)
        img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1], 0, 255)
        img_hsv[:, :, 2] = np.clip(img_hsv[:, :, 2], 0, 255)
        
        return cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    def augment_image(self, img, augmentations=['rotation', 'flip', 'brightness', 'crop', 'color']):
        """Apply random augmentations to an image"""
        aug_img = img.copy()
        
        # Randomly select which augmentations to apply
        selected_augs = random.sample(augmentations, 
                                     k=random.randint(1, len(augmentations)))
        
        for aug in selected_augs:
            if aug == 'rotation':
                aug_img = self.random_rotation(aug_img)
            elif aug == 'flip':
                aug_img = self.random_flip(aug_img)
            elif aug == 'brightness':
                aug_img = self.random_brightness(aug_img)
            elif aug == 'crop':
                aug_img = self.random_crop(aug_img)
            elif aug == 'color':
                aug_img = self.color_jitter(aug_img)
        
        return aug_img


def process_dataset(input_dir, output_dir, augmentations_per_image=5, target_size=(224, 224)):
    """
    Process entire dataset: resize, normalize, and augment images.
    
    Args:
        input_dir: Path to folder with subfolders (one per class/motif)
        output_dir: Path where augmented images will be saved
        augmentations_per_image: Number of augmented versions per original image
        target_size: Target size for all images (width, height)
    """
    
    augmenter = ImageAugmenter(target_size)
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Get all class folders
    class_folders = [f for f in input_path.iterdir() if f.is_dir()]
    
    print(f"Found {len(class_folders)} motif classes")
    
    for class_folder in class_folders:
        class_name = class_folder.name
        print(f"\nProcessing class: {class_name}")
        
        # Create output directories
        output_class_dir = output_path / class_name
        output_class_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all images in this class
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(class_folder.glob(ext))
        
        print(f"  Found {len(image_files)} images")
        
        for idx, img_file in enumerate(image_files):
            # Read original image
            img = cv2.imread(str(img_file))
            if img is None:
                print(f"  Warning: Could not read {img_file}")
                continue
            
            # Save original (resized and normalized)
            original = augmenter.resize_and_normalize(img)
            original_uint8 = (original * 255).astype(np.uint8)
            output_file = output_class_dir / f"{img_file.stem}_original.png"
            cv2.imwrite(str(output_file), original_uint8)
            
            # Generate augmented versions
            for aug_idx in range(augmentations_per_image):
                aug_img = augmenter.augment_image(img)
                aug_img = augmenter.resize_and_normalize(aug_img)
                aug_img_uint8 = (aug_img * 255).astype(np.uint8)
                
                output_file = output_class_dir / f"{img_file.stem}_aug{aug_idx+1}.png"
                cv2.imwrite(str(output_file), aug_img_uint8)
            
            if (idx + 1) % 10 == 0:
                print(f"  Processed {idx + 1}/{len(image_files)} images")
        
        total_images = len(image_files) * (augmentations_per_image + 1)
        print(f"  Total images for {class_name}: {total_images}")


if __name__ == "__main__":
    # Configuration
    TARGET_SIZE = (224, 224)  # Standard size for many CNNs
    
    print("=" * 60)
    print("IMAGE AUGMENTATION SCRIPT")
    print("=" * 60)
    
    # Process TRAINING data with augmentation
    print("\n>>> Processing TRAINING data (with augmentation)...")
    process_dataset(
        input_dir="train",
        output_dir="train_augmented",
        augmentations_per_image=5,  # Create 5 augmented versions per image
        target_size=TARGET_SIZE
    )
    
    # Process TEST data WITHOUT augmentation (only resize & normalize)
    print("\n>>> Processing TEST data (no augmentation, only resize/normalize)...")
    process_dataset(
        input_dir="test",
        output_dir="test_processed",
        augmentations_per_image=0,  # No augmentation for test set!
        target_size=TARGET_SIZE
    )
    
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE!")
    print("=" * 60)
    print("\nOutput:")
    print("  - train_augmented/  (training data with augmentation)")
    print("  - test_processed/   (test data, original only)")