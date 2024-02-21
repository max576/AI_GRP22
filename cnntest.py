import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

num_epochs = 0
batch_size = 4
learning_rate = 0.001

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Define data transforms (you can add more as needed)
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),  # Resize images to a fixed size
#     transforms.ToTensor(),           # Convert images to PyTorch tensors
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images
# ])

# Define path to your data
data_path = 'Data_collection/data_150'

# Create dataset
full_dataset = datasets.ImageFolder(data_path, transform=transform)

# Define the ratio of train and test data
train_ratio = 0.8  # 80% of data for training, 20% for testing
dataset_size = len(full_dataset)
train_size = int(train_ratio * dataset_size)
test_size = dataset_size - train_size

# Split dataset into train and test sets
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# You can iterate over train_loader and test_loader to get batches of data
for images, labels in train_loader:
    # Your training loop goes here
    pass

for images, labels in test_loader:
    # Your testing loop goes here
    pass


def imshow(img):
    img = img / 2 * 0.5 # unnormalise
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# random training images
dataiter = iter(train_loader)
images, labels = next(dataiter)

# show images
imshow(torchvision.utils.make_grid(images))

conv1 = nn.Conv2d(3, 6, 5)
pool = nn.MaxPool2d(2, 2)
conv2 = nn.Conv2d(6, 16, 5)
#fc1 = nn.Linear(16*5*5, 120) 
#fc2 = nn.Linear(120, 84)
#fc3 = nn.Linear(84, 10) #change 10 to num of classes
print(images.shape) # image size will be 3: rgb. 4: batch size, 32, 32
x = conv1(images)
print(x.shape) 
x = pool(x)
print(x.shape)
x = conv2(x)
print(x.shape)
x = pool(x)
print(x.shape)
