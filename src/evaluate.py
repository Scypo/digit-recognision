from .model import Model
from .data_loader import test_loader
from .image_processing import preprocess
import torch
import torch.nn as nn
import os
def evaluate(model):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    os.system('cls')
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.view(images.size(0), -1)
            predictions = model(images)
            loss = criterion(predictions, labels)

            total_loss += loss.item()
            total += labels.size(0)
            predicted_class = predictions.argmax(dim=1)
            correct += (predicted_class == labels).sum().item()
    os.system('cls')
    avg_loss = total_loss / len(test_loader)
    accuracy = correct / total
    print(
        f"Test Loss: {avg_loss:.4f} | "
        f"Test Accuracy: {accuracy:.2%}"
    )
def evaluate_image(model, path: str):
    img = preprocess(path)
    img = img.view(img.size(0), -1)
    model.eval()
    with torch.no_grad():
        prediction = model(img)
        probability = torch.softmax(prediction, dim=1)
        predicted_class = prediction.argmax(dim=1).item()
        confidence = probability[0][predicted_class].item() * 100
        os.system('cls')
        print("The Number Is:", predicted_class)
        print("The Confidence Is:", confidence)
    return predicted_class