import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
torch.manual_seed(0)


# Check whether we have a GPU.  Use it if we do.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Assuming your datasets are in a folder named 'car_models' and they are in the form of images
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize images to 32x32
    transforms.ToTensor(),  # Convert PIL image to tensor
])

# Load your car models dataset
train_dataset = torchvision.datasets.ImageFolder(root='Data/cars_train', transform=transform)
test_dataset = torchvision.datasets.ImageFolder(root='Data/cars_test', transform=transform)

# Data loaders
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)

# Assuming you have 10 different car models
num_classes = 10

model = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=100, kernel_size=3, padding=1),  # Assuming images are RGB, so in_channels=3
    nn.ReLU(),
    nn.Conv2d(in_channels=100, out_channels=num_classes, kernel_size=3, padding=1),
    nn.AdaptiveAvgPool2d(1)
).to(device)

# Optimizer
opt = torch.optim.SGD(model.parameters(), lr=0.1)

def train():
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        logits = model(images).squeeze((-1, -2))

        # Backpropagation and optimization
        loss = nn.functional.cross_entropy(logits, labels)
        loss.backward()
        opt.step()
        opt.zero_grad()

def test(epoch):
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            logits = model(images).squeeze((-1, -2))

            # Compute total correct so far
            predicted = torch.argmax(logits, -1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        print(f'Test accuracy after {epoch+1} epochs: {100 * correct / total} %')

# Run training
for epoch in range(5):
    train()
    test(epoch)

#########################

# # Check whether we have a GPU.  Use it if we do.
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # Assuming your datasets are in a folder named 'car_models' and they are in the form of images
# transform = transforms.Compose([
#     transforms.Resize((32, 32)),  # Resize images to 32x32
#     transforms.ToTensor(),  # Convert PIL image to tensor
# ])

# # Load your car models dataset
# train_dataset = torchvision.datasets.ImageFolder(root='car_models/train', transform=transform)
# test_dataset = torchvision.datasets.ImageFolder(root='car_models/test', transform=transform)

# # Data loaders
# train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)
# test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)

# # Assuming you have 10 different car models
# num_classes = 10

# model = nn.Sequential(
#     nn.Conv2d(in_channels=3, out_channels=100, kernel_size=3, padding=1),  # Assuming images are RGB, so in_channels=3
#     nn.ReLU(),
#     nn.Conv2d(in_channels=100, out_channels=num_classes, kernel_size=3, padding=1),
#     nn.AdaptiveAvgPool2d(1)
# ).to(device)

# # Optimizer
# opt = torch.optim.SGD(model.parameters(), lr=0.1)

# def train():
#     for images, labels in train_loader:
#         images = images.to(device)
#         labels = labels.to(device)

#         # Forward pass
#         logits = model(images).squeeze((-1, -2))

#         # Backpropagation and optimization
#         loss = nn.functional.cross_entropy(logits, labels)
#         loss.backward()
#         opt.step()
#         opt.zero_grad()

# def test(epoch):
#     with torch.no_grad():
#         correct = 0
#         total = 0
#         for images, labels in test_loader:
#             images = images.to(device)
#             labels = labels.to(device)

#             # Forward pass
#             logits = model(images).squeeze((-1, -2))

#             # Compute total correct so far
#             predicted = torch.argmax(logits, -1)
#             correct += (predicted == labels).sum().item()
#             total += labels.size(0)
#         print(f'Test accuracy after {epoch+1} epochs: {100 * correct / total} %')

# # Run training
# for epoch in range(5):
#     train()
#     test(epoch)
