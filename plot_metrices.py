import matplotlib.pyplot as plt

def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, 
                 train_precisions, val_precisions, train_recalls, val_recalls):
    epochs = range(1, len(train_losses) + 1)

    # Plot Loss
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss', color='b')
    plt.plot(epochs, val_losses, label='Validation Loss', color='r')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Accuracy
    plt.subplot(2, 2, 2)
    plt.plot(epochs, train_accuracies, label='Training Accuracy', color='b')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy', color='r')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot Precision-Recall Curve for Training
    plt.subplot(2, 2, 3)
    plt.plot(train_recalls, train_precisions, label='Train P-R Curve', color='b')
    plt.title('Training Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()

    # Plot Precision-Recall Curve for Validation
    plt.subplot(2, 2, 4)
    plt.plot(val_recalls, val_precisions, label='Validation P-R Curve', color='r')
    plt.title('Validation Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()

    # Save the plots
    plt.tight_layout()
    plt.savefig("training_validation_metrics.png")
    plt.show()
