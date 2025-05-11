"""Evaluate model performance on test data."""
import pathlib
import argparse
import torch
import torchvision
from torchvision import transforms as T
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from utils import NucleusDataset

def evaluate_model(model, test_loader, device="cpu"):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    return np.array(all_preds), np.array(all_labels)

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()

def main(args):
    # Load model
    model = torchvision.models.resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    model.to(args.device)
    
    # Prepare data
    transform = T.Compose([
        T.Resize((224, 224), antialias=True),
        T.Lambda(lambda x: x[:3] if x.shape[0] == 4 else x.repeat(3, 1, 1)),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    dataset = NucleusDataset(args.data_dir, args.csv_path, transform)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=32)
    
    # Evaluate
    predictions, true_labels = evaluate_model(model, test_loader, args.device)
    
    # Print metrics
    print("\nClassification Report:")
    print(classification_report(true_labels, predictions, 
                              target_names=['Normal', 'Bleb']))
    
    # Plot confusion matrix
    plot_confusion_matrix(true_labels, predictions, ['Normal', 'Bleb'])
    print("\nConfusion matrix saved as 'confusion_matrix.png'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="model.pt")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--csv_path", default="data/annotations/labels.csv")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    main(parser.parse_args()) 