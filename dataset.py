import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random


class SegNetDataset(Dataset):
    def __init__(self, image_dir, mask_dir, input_size=(224, 224), augment=True):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.input_size = input_size
        self.augment = augment
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])

        # Extract the base name without the file extension
        base_name = os.path.splitext(self.images[index])[0]
        
        mask_path_jpg = os.path.join(self.mask_dir, base_name + ".jpg")
        mask_path_png = os.path.join(self.mask_dir, base_name + ".png")

        if os.path.exists(mask_path_jpg):
            mask_path = mask_path_jpg
        elif os.path.exists(mask_path_png):
            mask_path = mask_path_png
        else:
            raise FileNotFoundError(f"Maske dosyası bulunamadı: {mask_path_jpg} veya {mask_path_png}")
        
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        
        # Augmentation
        if self.augment:
            image, mask = self.data_augmentation(image, mask)
            
        image, mask = self.resize_and_normalize(image, mask)
        
        return image, mask

    def data_augmentation(self, image, mask):
        # Convert numpy arrays back to PIL for augmentation
        image = Image.fromarray(image.astype(np.uint8))
        mask = Image.fromarray(mask.astype(np.uint8))
        
        # Random horizontal flipping
        if random.random() > 0.5:
            image = transforms.functional.hflip(image)
            mask = transforms.functional.hflip(mask)
        
        # Random vertical flipping
        if random.random() > 0.5:
            image = transforms.functional.vflip(image)
            mask = transforms.functional.vflip(mask)
        
        # Random rotation
        angle = transforms.RandomRotation.get_params([-30, 30])
        image = transforms.functional.rotate(image, angle)
        mask = transforms.functional.rotate(mask, angle)

        # Random crop (Patch Sampling)
        i, j, h, w = transforms.RandomCrop.get_params(
            image, output_size=self.input_size)
        image = transforms.functional.crop(image, i, j, h, w)
        mask = transforms.functional.crop(mask, i, j, h, w)
        
        # Random zoom
        resize = transforms.RandomResizedCrop(
            size=self.input_size, scale=(0.8, 1.0))
        image = resize(image)
        mask = resize(mask)
        
        return np.array(image), np.array(mask)
    
    def resize_and_normalize(self, image, mask):
        # Convert numpy arrays back to PIL for resizing
        image = Image.fromarray(image.astype(np.uint8))
        mask = Image.fromarray(mask.astype(np.uint8))
        
        # Resize
        resize = transforms.Resize(self.input_size, Image.BILINEAR)
        image = resize(image)
        mask = resize(mask)
        
        # Convert to tensor and normalize
        image = transforms.ToTensor()(image)
        image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)
        
        mask = transforms.ToTensor()(mask)
        
        return image, mask

if __name__ == '__main__':
    dataset = SegNetDataset("path_to_images", "path_to_masks", (224, 224), augment=True)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    for images, masks in dataloader:
        pass

