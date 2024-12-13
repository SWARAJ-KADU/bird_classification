import torch
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import cv2
import numpy as np

class ImageDataset(Dataset):
    def __init__(self, image_vectors, labels, target_size=(224, 224)):
        """
        Args:
            image_vectors (list or np.ndarray): List/array of 3D image vectors (H, W, C).
            labels (list or np.ndarray): Corresponding labels for the images.
            target_size (tuple): The target size to resize the images (width, height).
        """
        self.image_vectors = image_vectors
        self.labels = labels
        self.target_size = target_size

    def __len__(self):
        return len(self.image_vectors)

    def __getitem__(self, idx):
        # Load the image and label
        image = self.image_vectors[idx]
        label = self.labels[idx]

        # Resize the image to target size
        resized_image = cv2.resize(image, self.target_size)

        # Normalize the image (optional)
        resized_image = resized_image / 255.0

        # If the image is grayscale, add a channel dimension
        if len(resized_image.shape) == 2:
            resized_image = np.stack([resized_image] * 3, axis=-1)

        # Convert to PyTorch tensor and permute dimensions to (C, H, W)
        resized_image = torch.tensor(resized_image, dtype=torch.float32).permute(2, 0, 1)  # (C, H, W)
        label = torch.tensor(label, dtype=torch.long)

        return resized_image, label


# Create DataLoader
def get_dataloaders(image_vectors, labels, batch_size=32, test_split=0.2, target_size=(224, 224)):
    """
    Create train and test DataLoaders from image vectors and labels.
    Args:
        image_vectors (list or np.ndarray): List/array of 3D image vectors (H, W, C).
        labels (list or np.ndarray): Corresponding labels for the images.
        batch_size (int): Number of samples per batch.
        test_split (float): Proportion of data to use for testing.
        target_size (tuple): The target size to resize the images (width, height).
    Returns:
        train_loader, test_loader: PyTorch DataLoaders for training and testing.
    """
    dataset = ImageDataset(image_vectors, labels, target_size)

    # Split the dataset into training and testing sets
    test_size = int(len(dataset) * test_split)
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader