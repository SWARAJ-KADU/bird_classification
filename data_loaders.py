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


def get_dataloaders(image_vectors, labels, batch_size=32, test_split=0.2, target_size=(224, 224), seed=42):
    """
    Create train and test DataLoaders from image vectors and labels after shuffling indices with an optional seed.
    Args:
        image_vectors (list or np.ndarray): List/array of 3D image vectors (H, W, C).
        labels (list or np.ndarray): Corresponding labels for the images.
        batch_size (int): Number of samples per batch.
        test_split (float): Proportion of data to use for testing.
        target_size (tuple): The target size to resize the images (width, height).
        seed (int, optional): Seed for reproducibility. Default is None.
    Returns:
        train_loader, test_loader: PyTorch DataLoaders for training and testing.
    """
    # Ensure inputs are numpy arrays for indexing
    image_vectors = np.array(image_vectors)
    labels = np.array(labels)

    # Set random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)

    # Shuffle the dataset indices
    indices = np.random.permutation(len(image_vectors))

    # Split indices for train and test
    test_size = int(len(indices) * test_split)
    train_indices = indices[:-test_size]
    test_indices = indices[-test_size:]

    # Create datasets using the shuffled indices
    train_dataset = ImageDataset(image_vectors[train_indices], labels[train_indices], target_size)
    test_dataset = ImageDataset(image_vectors[test_indices], labels[test_indices], target_size)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
