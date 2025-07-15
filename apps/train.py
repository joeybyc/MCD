import os

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


import cv2
import random

from lib.model.spatial_attention_network import Net
from lib.common.config import cfg
import argparse

import os


class CellDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.original_data = []
        self.sampled_data = []

        # Populate original_data
        for label in ['0', '1']:
            label_dir = os.path.join(self.root_dir, label)
            for img_filename in os.listdir(label_dir):
                self.original_data.append((os.path.join(label_dir, img_filename), int(label)))

        # Initially, sampled_data is the same as original_data
        self.sampled_data = list(self.original_data)

        # Count labels to find the minority class count
        label_counts = {0: 0, 1: 0}
        for _, label in self.original_data:
            label_counts[label] += 1
        self.min_count = min(label_counts.values())

    def __len__(self):
        return len(self.sampled_data)

    def __getitem__(self, idx):
        img_path, label = self.sampled_data[idx]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if self.transform:
            img = self.transform(img)
        return img, label

    def down_sample(self):
        # Shuffle original data to ensure randomness
        random.shuffle(self.original_data)

        # Downsample data
        downsampled_data = []
        counts = {0: 0, 1: 0}
        for item in self.original_data:
            _, label = item
            if counts[label] < self.min_count:
                downsampled_data.append(item)
                counts[label] += 1
        self.sampled_data = downsampled_data


# Training function
def train_model(model, criterion, optimizer, train_dataset, val_dataset, model_save_path, epochs=100, patience=30, balanced_flag = False, batch_size=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    best_val_loss = np.inf
    patience_counter = 0

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    if not balanced_flag:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # Reinitialize DataLoader
    for epoch in range(epochs):
        if balanced_flag:
            # Downsample and reload training data
            train_dataset.down_sample()  # Call down_sample before each epoch
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # Reinitialize DataLoader
        

        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device).float(), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        val_loss = 0.0
        model.eval()

        # val_dataset.down_sample()  # Call down_sample before validation
        # val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device).float(), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}, Val Loss: {val_loss/len(val_loader)}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)  # Save the best model
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping")
                break

print("")
print("-------------------------------Start Training---------------------------------------------------------------")

parser = argparse.ArgumentParser()
parser.add_argument("-cfg", "--config", type=str, default="./configs/training_settings.yaml")
args = parser.parse_args()
cfg.merge_from_file(args.config)

dataset_folder = cfg.dataset.root

print(f'The device is {torch.device("cuda" if torch.cuda.is_available() else "cpu")}')
# Load datasets

img_size = cfg.cell_classifier.size_L
transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((img_size, img_size)), transforms.ToTensor()])
transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(p=0.3), 
    transforms.RandomVerticalFlip(p=0.3), 
    transforms.RandomRotation(30),
    transforms.Resize((img_size, img_size)), 
    transforms.ToTensor(),
])

data_path_train = os.path.join(dataset_folder, cfg.dataset.train)
data_path_val = os.path.join(dataset_folder, cfg.dataset.val)
data_path_test = os.path.join(dataset_folder, cfg.dataset.test)

model_folder = cfg.model.folder
os.makedirs(model_folder, exist_ok=True)
model_save_path = os.path.join(model_folder, cfg.model.name)

print(f'Path for training data: {data_path_train}')
print(f'Path for val data: {data_path_val}')
print(f'Path for test data: {data_path_test}')
print(f'Path for saving model: {model_save_path}')

train_dataset = CellDataset(data_path_train, transform=transform_train)
val_dataset = CellDataset(data_path_val, transform=transform)
test_dataset = CellDataset(data_path_test, transform=transform)


test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Initialize model, loss, and optimizer
model = Net(size_img=img_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Train the model
train_model(model, criterion, optimizer, train_dataset, val_dataset, model_save_path, balanced_flag = False)

# Load the best model
model.load_state_dict(torch.load(model_save_path))
print("-------------------------------End Training----------------------------------------------------------------------------")

print("-------------------------------Start Testing----------------------------------------------------------------------------")
# Testing
def test_model(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device).float()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    accuracy = accuracy_score(all_labels, all_preds)

    print(f"Precision: {precision}, Recall: {recall}, F1: {f1}, Accuracy: {accuracy}")

# Evaluate the model
test_model(model, test_loader)


print("-------------------------------End Testing----------------------------------------------------------------------------")

print("")