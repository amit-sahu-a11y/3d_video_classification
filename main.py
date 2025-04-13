import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import UCF101Dataset
from models.simple3dcnn import Simple3DCNN
from tqdm import tqdm
import os
from multiprocessing import freeze_support

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root_dir = "dataset/UCF-101"
    classes = sorted(os.listdir(root_dir))

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
    ])

    dataset = UCF101Dataset(root_dir, classes, transform=transform)
    
    # Set num_workers=0 to avoid multiprocessing issues
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    model = Simple3DCNN(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        model.train()
        total_loss = 0
        correct = 0

        for inputs, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()

        print(f"Epoch {epoch+1}: Loss = {total_loss/len(dataloader):.4f}, Accuracy = {correct/len(dataset):.4f}")

    os.makedirs("saved_models", exist_ok=True)
    torch.save(model.state_dict(), "saved_models/model.pth")
    print("✅ Model saved to: saved_models/model.pth")


if __name__ == "__main__":
    freeze_support()  # 👈🏽 REQUIRED on Windows for multiprocessing
    train()
