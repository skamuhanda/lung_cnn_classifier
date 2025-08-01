import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode
import matplotlib.pyplot as plt
from tqdm import tqdm

# -----------------------------
# Configuration
# -----------------------------
DATA_DIR = "E:/lung_cnn_diagnosis/train_val_test_split"
BATCH_SIZE = 32
IMG_SIZE = 224
EPOCHS = 20
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = "best_model.pth"

PATIENCE = 5
early_stop_counter = 0
best_val_loss = float('inf')

# -----------------------------
# Data Augmentation & Normalization
# -----------------------------
train_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=InterpolationMode.BILINEAR),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.9, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

val_test_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# -----------------------------
# Dataset Loading
# -----------------------------
train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transforms)
val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=val_test_transforms)
test_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=val_test_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# -----------------------------
# Define Custom 10-layer CNN
# -----------------------------
class LungCancerCNN(nn.Module):
    def __init__(self):
        super(LungCancerCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # Layer 1
            nn.ReLU(),
            nn.MaxPool2d(2),  # Layer 2

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Layer 3
            nn.ReLU(),
            nn.MaxPool2d(2),  # Layer 4

            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # Layer 5
            nn.ReLU(),  # Layer 6

            nn.Dropout(0.3),  # Layer 7
            nn.Flatten(),     # Layer 8

            nn.Linear(128 * 56 * 56, 128),  # Layer 9
            nn.ReLU(),
            nn.Linear(128, 1),              # Layer 10
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# -----------------------------
# Initialize Model, Loss, Optimizer
# -----------------------------
model = LungCancerCNN().to(DEVICE)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# -----------------------------
# Training Loop (Protected for Windows)
# -----------------------------
if __name__ == "__main__":
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        correct = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - Training"):
            images = images.to(DEVICE)
            labels = labels.float().unsqueeze(1).to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            preds = (outputs > 0.5).int()
            correct += (preds == labels.int()).sum().item()

        train_loss /= len(train_loader.dataset)
        train_acc = correct / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        val_correct = 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validating"):
                images = images.to(DEVICE)
                labels = labels.float().unsqueeze(1).to(DEVICE)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                preds = (outputs > 0.5).int()
                val_correct += (preds == labels.int()).sum().item()

        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / len(val_loader.dataset)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        print(f"\nEpoch {epoch+1}/{EPOCHS}: "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print("✅ Saved new best model.")
        else:
            early_stop_counter += 1
            print(f"⚠️ No improvement in validation loss. Early stop counter: {early_stop_counter}/{PATIENCE}")
            if early_stop_counter >= PATIENCE:
                print("🛑 Early stopping triggered. Ending training.")
                break

    # -----------------------------
    # Save Accuracy/Loss Plots
    # -----------------------------
    epochs_range = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label='Training Loss')
    plt.plot(epochs_range, val_losses, label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accuracies, label='Training Accuracy')
    plt.plot(epochs_range, val_accuracies, label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_validation_curves.png", dpi=300)
    plt.close()

    print("📊 Training and validation curves saved as 'training_validation_curves.png'")