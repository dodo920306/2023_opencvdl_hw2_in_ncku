import os
from torch import device, float32, load, no_grad, save, tensor
from torch.cuda import is_available
from torch.nn import Linear
from torch.nn.modules.loss import BCEWithLogitsLoss
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from torchvision.transforms.v2 import Compose, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, ToImage, ToDtype, Resize, RandomErasing
from torchvision.models import resnet50
from collections import OrderedDict
from PIL import Image, ImageFile
from tqdm import tqdm
import matplotlib.pyplot as plt


ImageFile.LOAD_TRUNCATED_IMAGES = True

class MyDataset(Dataset):
    def __init__(self, transform=None, train=True):
        self.img_labels = []
        file_list = []
        labels = []
        folder_path = "Dataset_OpenCvDl_Hw2_Q5/dataset/"

        if train:
            folder_path += "training_dataset/"
            for filename in os.listdir(folder_path + "Cat"):
                file_path = os.path.join(folder_path + "Cat", filename)
                if os.path.isfile(file_path):
                    file_list.append(filename)
                    labels.append(0)

            for filename in os.listdir(folder_path + "Dog"):
                file_path = os.path.join(folder_path + "Dog", filename)
                if os.path.isfile(file_path):
                    file_list.append(filename)
                    labels.append(1)
        else:
            folder_path += "validation_dataset/"
            for filename in os.listdir(folder_path + "Cat"):
                file_path = os.path.join(folder_path + "Cat", filename)
                if os.path.isfile(file_path):
                    file_list.append(filename)
                    labels.append(0)

            for filename in os.listdir(folder_path + "Dog"):
                file_path = os.path.join(folder_path + "Dog", filename)
                if os.path.isfile(file_path):
                    file_list.append(filename)
                    labels.append(1)

        self.img_labels.append(file_list)
        self.img_labels.append(labels)

        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.img_labels[0])

    def __getitem__(self, idx):
        label = self.img_labels[1][idx]
        folder_path = "Dataset_OpenCvDl_Hw2_Q5/dataset/"
        folder_path += "training_dataset/" if self.train else "validation_dataset/"
        folder_path += "Cat" if label == 0 else "Dog"
        img_path = os.path.join(folder_path, self.img_labels[0][idx])

        # There is a chance that the image isn't seen like .jpeg to read_image(), so
        # use PIL.Image to convert it first.
        # There is to_tensor in transform, so no need to do it here.
        # Don't save to prevent I/O problem.
        image = Image.open(img_path).convert("RGB")

        image = self.transform(image)
        return image, label

if __name__ == '__main__':
    local_device = device("cuda" if is_available() else "cpu")
    y = []

    transforms = Compose([
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        RandomRotation(30)
    ])

    to_tensor = Compose([
        Resize((224, 224), antialias=True),
        # ToImage() transforms PIL Image into torch.Tensor subclass for images.
        ToImage(),
        ToDtype(float32, scale=True)
    ])

    transformses = [Compose([
        transforms,
        to_tensor
    ]), Compose([
        transforms,
        to_tensor,
        RandomErasing()
    ])]

    num_epochs = 20

    for transforms in transformses:
        model = resnet50()
        model.fc = Linear(model.fc.in_features, 1)
        model = model.to(local_device)

        trainset = MyDataset(train=True, transform=transforms)
        trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=4)

        testset = MyDataset(train=False, transform=to_tensor)
        testloader = DataLoader(testset, batch_size=32, shuffle=False, num_workers=4)

        # binary cross entropy with sigmoid
        criterion = BCEWithLogitsLoss()
        optimizer = Adam(model.parameters(), lr=0.0001)

        best_accuracy = 0.0
        best_model_weights = OrderedDict()
        train_losses, train_accuracies, val_losses, val_accuracies = [], [], [], []

        for epoch in range(num_epochs):
            # train
            model.train()

            running_loss = 0.0
            correct = 0
            total = 0
            for i, data in tqdm(enumerate(trainloader), total = len(trainloader)):
                inputs, labels = data
                inputs, labels = inputs.to(local_device), labels.unsqueeze(1).float().to(local_device)
                
                # Before computing the gradients for a new minibatch, 
                # it is common to zero out the gradients from the previous minibatch.
                # This is done to prevent the gradients from accumulating across multiple batches.
                # If you don't zero out the gradients, the new gradients will be added to the existing gradients, 
                # and it might lead to incorrect updates during optimization.
                optimizer.zero_grad()

                # Obtain the model's predictions (outputs) for the input data.
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Compute the gradients of the loss with respect to the model parameters, enabling backpropagation.
                loss.backward()

                # Update the model parameters using the optimizer and the computed gradients.
                optimizer.step()

                # .item() converts the result to a Python number.
                running_loss += loss.item()

                total += labels.size(0)
                # Different with labels, output elements won't simply be 0. or 1., so
                # turn every element smaller than 0.5 to 0, vice versa.
                predicted = (outputs >= 0.5).float()
                correct += predicted.eq(labels).sum().item()

            # average training loss
            train_loss = running_loss / len(trainloader)
            train_accuracy = 100.0 * correct / total

            # test
            model.eval()
            running_loss = 0.0
            correct = 0
            total = 0

            # PyTorch operations won't build up a computation graph for automatic differentiation.
            # This is done for efficiency during evaluation since you don't need to compute gradients for the parameters.
            with no_grad():
                for i, data in tqdm(enumerate(testloader), total = len(testloader)):
                    inputs, labels = data
                    inputs, labels = inputs.to(local_device), labels.unsqueeze(1).float().to(local_device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    running_loss += loss.item()
                    predicted = (outputs > 0.5).float()
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
            val_loss = running_loss / len(testloader)
            val_accuracy = 100.0 * correct / total

            # save
            print(f"Epoch {epoch+1}/{num_epochs}, "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
            
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)

            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_model_weights = model.state_dict()

        y.append(best_accuracy)
        save(best_model_weights, "best_resnet18_weights.pth")

    x = ["Without Random erasing", "With Random erasing"]
    plt.bar(x, y, width=0.5)
    plt.ylim(0, 100)

    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Comparison')

    for i, value in enumerate(y):
        plt.text(i, value + 0.1, str(round(value)), ha='center', va='bottom')

    plt.savefig("Comparison.png")
