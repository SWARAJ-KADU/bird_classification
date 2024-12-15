import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR  # Learning rate scheduler
from model import BirdClassifier
from data_loaders import get_dataloaders
from checkpoints import save_checkpoint, load_checkpoint
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# Training Function with Checkpoint Integration
def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=10, device='cuda', checkpoint_path="checkpoint.pth"):
    model.to(device)
    # Load checkpoint if available
    model, optimizer, scheduler, start_epoch = load_checkpoint(checkpoint_path, model, optimizer, scheduler, device)

    for epoch in range(start_epoch, num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 20)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        # Step the scheduler after each epoch
        scheduler.step()

        # Save a checkpoint after each epoch
        checkpoint = {
            'epoch': epoch + 1,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
        }
        save_checkpoint(checkpoint, checkpoint_path)

    return model


# Main Function
if __name__ == "__main__":
    # Hyperparameters
    data_dir = "image_vectors_labels.pkl"  # Path to dataset (train and val folders)
    num_classes = 200
    batch_size = 32
    learning_rate = 1e-4
    num_epochs = 30
    weight_decay = 1e-5  # Weight decay (L2 regularization)
    checkpoint_path = "bird_classifier_checkpoint.pth"

    df = pd.read_pickle(data_dir)
    image_vector = df["vector"]
    label_vector = df["label"]

    df['label'] = LabelEncoder().fit_transform(label_vector)
    label_vector = df['label']

    # Data Preparation
    train_loader, val_loader = get_dataloaders(image_vector, label_vector, batch_size, 0.35)
    dataloaders = {'train': train_loader, 'val': val_loader}

    # Model, Loss, and Optimizer
    model = BirdClassifier(num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()   

    # Adam optimizer with weight decay
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=weight_decay)

    # Learning rate scheduler (StepLR)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)  # Decrease LR by a factor of 10 every 10 epochs

    # Train the Model
    trained_model = train_model(
        model,
        dataloaders,
        criterion,
        optimizer,
        scheduler,
        num_epochs=num_epochs,
        checkpoint_path=checkpoint_path
    )

    # Save final model
    torch.save(trained_model.state_dict(), "bird_classifier.pth")
