import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# -----------------------
# Configuration
# -----------------------
DATA_DIR = "E:/lung_cnn_diagnosis/train_val_test_split/test"
MODEL_PATH = "E:/lung_cnn_diagnosis/best_model.pth"
IMG_SIZE = 224
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# Transforms (same as validation)
# -----------------------
test_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# -----------------------
# Dataset & Dataloader
# -----------------------
test_dataset = datasets.ImageFolder(DATA_DIR, transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# -----------------------
# CNN Model (Same as before)
# -----------------------
class LungCancerCNN(nn.Module):
    def __init__(self):
        super(LungCancerCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Flatten(),
            nn.Linear(128 * 56 * 56, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# -----------------------
# Load Model
# -----------------------
model = LungCancerCNN().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# -----------------------
# Inference on Test Set
# -----------------------
y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        preds = (outputs > 0.5).int().cpu().numpy()
        y_pred.extend(preds)
        y_true.extend(labels.cpu().numpy())

# -----------------------
# Confusion Matrix
# -----------------------
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", 
            xticklabels=test_dataset.classes,
            yticklabels=test_dataset.classes)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Test Set")
plt.tight_layout()
plt.savefig("test_confusion_matrix.png", dpi=300)
plt.show()

# -----------------------
# Classification Report
# -----------------------
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=test_dataset.classes))
from sklearn.metrics import roc_curve, auc

# -----------------------
# Compute ROC Curve and AUC
# -----------------------

# You need predicted probabilities, not just binary predictions
y_scores = []

with torch.no_grad():
    for images, _ in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        y_scores.extend(outputs.cpu().numpy())

y_scores = np.array(y_scores)
y_true_np = np.array(y_true)

# Compute ROC and AUC
fpr, tpr, thresholds = roc_curve(y_true_np, y_scores)
roc_auc = auc(fpr, tpr)

# Plot ROC Curve
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curve – Test Set")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("test_roc_curve.png", dpi=300)
plt.show()
