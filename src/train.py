from .model import Model
from .data_loader import train_loader
import torch
import torch.nn as nn
import os
def train(model, epochs = 5):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    total_loss = 0
    correct = 0
    total = 0

    model.train()
    os.system('cls')    
    for epoch in range(epochs):
        print("On epoch:", epoch + 1, "out of:", epochs)
        for images, labels in train_loader:
            images = images.view(images.size(0), -1)
            optimizer.zero_grad()
            predictions = model(images)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        total_loss += loss.item()
        total += labels.size(0)
        predicted_class = predictions.argmax(dim=1)
        correct += (predicted_class == labels).sum().item()
        os.system('cls')
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        print(
            f"Test Loss: {avg_loss:.4f} | "
            f"Test Accuracy: {accuracy:.2%}"
        )