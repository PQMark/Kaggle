import torch
import torch.utils.data.dataloader
import os
from tqdm import tqdm
from PIL import Image
import torchvision.transforms.v2 as T

def create_transforms(image_size: int = 112, augment: bool = True) -> T.Compose:
    """Create transform pipeline for face recognition."""

    # Basic transformations
    transform_list = [
        T.Resize((image_size, image_size)),

        T.ToTensor(),

        # Convert image to float32 and scale the pixel values to [0, 1]
        T.ToDtype(torch.float32, scale=True),
    ]

    # Data augmentation
    if augment: 
        transform_list.extend([
            T.RandomHorizontalFlip(p=0.5), 
            T.RandomRotation(degrees=10), 
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), 
            T.RandomErasing(p=0.1, scale=(0.02, 0.15), ratio=(0.5, 2,0), value=0), 
            T.RandomGrayscale(p=0.1),
            T.RandomPerspective(distortion_scale=0.1, p=0.1)
        ])

    # Standard normalization for image recognition tasks
    # The Normalize transformation requires mean and std values for each channel (R, G, B).
    # Normalize the pixel values to have a mean of 0.5 and std of 0.5 for each channel.
    transform_list.extend([
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Standard mean and std for face recognition tasks
    ])

    # Return the composed transformation pipeline
    return T.Compose(transform_list)


class ImageDataset(torch.utils.data.Dataset):
    """Custom dataset for loading image-label pairs."""
    def __init__(self, root, transform, num_classes=None):
        """
        Args:
            root (str): Path to the directory containing the images folder.
            transform (callable): Transform to be applied to the images.
            num_classes (int, optional): Number of classes to keep. If None, keep all classes.
        """

        self.root = root
        self.labels_file = os.path.join(self.root, "labels.txt")  # fig_name, class
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.classes = set()

        # Read image-label pairs from the file
        with open(self.labels_file, 'r') as f:
            lines = f.readlines()
        
        lines = sorted(lines, key=lambda x: int(x.strip().split(' ')[-1]))

        # Get all unique labels
        all_labels = sorted(set(int(line.strip().split(' ')[1]) for line in lines))
        
        # Select subset of classes if specified
        if num_classes is not None:
            selected_classes = set(all_labels[:num_classes])
        else:
            selected_classes = set(all_labels)
        
        # Store image paths and labels with a progress bar
        for line in tqdm(lines, desc="Loading dataset"):
            img_path, label = line.strip().split(' ')
            label = int(label)

            # Only add if label is in selected classes
            if label in selected_classes:
                self.image_paths.append(os.path.join(self.root, 'images', img_path))
                self.labels.append(label)
                self.classes.add(label)

        assert len(self.image_paths) == len(self.labels), "Images and labels mismatch!"

        # Convert classes to a sorted list
        self.classes = sorted(self.classes)
    
    def __len__(self):
        """Returns the total number of samples."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (transformed image, label)
        """
        # Load and transform image on-the-fly
        image = Image.open(self.image_paths[idx]).convert('RGB')
        image = self.transform(image)
        label = self.labels[idx]
        return image, label


class ImagePairDataset(torch.utils.data.Dataset):
    """Custom dataset for loading and transforming image pairs for verification."""
    def __init__(self, root, pairs_file, transform, test=False):
        """
        Args:
            root (str): Path to the directory containing the images.
            pairs_file (str): Path to the file containing image pairs and match labels.
            transform (callable): Transform to be applied to the images.
        """
        self.root      = root
        self.transform = transform
        self.test = test

        self.matches     = [] 
        self.image1_list = []
        self.image2_list = []

        # Read and load image pairs and match labels
        with open(pairs_file, 'r') as f:
            lines = f.readlines()

        for line in tqdm(lines, desc="Loading image pairs"):
            parts = line.strip().split(' ')
            img_path1, img_path2 = parts[:2]
            img1 = Image.open(os.path.join(self.root, img_path1)).convert('RGB')
            img2 = Image.open(os.path.join(self.root, img_path2)).convert('RGB')

            self.image1_list.append(img1)
            self.image2_list.append(img2)
            
            if not self.test:
                self.matches.append(int(parts[2]))  # Convert match to integer

        if not self.test:
            assert len(self.image1_list) == len(self.image2_list) == len(self.matches), "Image pair mismatch"
        else:
            assert len(self.image1_list) == len(self.image2_list), "Image pair mismatch"

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.image1_list)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (transformed image1, transformed image2, match label)
        """
        img1 = self.image1_list[idx]
        img2 = self.image2_list[idx]
        
        if not self.test:
            match = self.matches[idx]
            return self.transform(img1), self.transform(img2), match
        else:
            return self.transform(img1), self.transform(img2)

