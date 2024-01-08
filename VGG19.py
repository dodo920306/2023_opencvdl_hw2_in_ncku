from PyQt5.QtWidgets import QApplication, QPushButton, QHBoxLayout, QVBoxLayout, QGroupBox, QLabel, QWidget
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPainter, QPen, QPixmap
import sys, os, cv2
from torch import device, float32, load, no_grad, save, from_numpy
from torch.cuda import is_available
from torch.nn import CrossEntropyLoss
from torch.nn.functional import softmax
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision.transforms.v2 import Compose, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, ToImage, ToDtype, Resize, Grayscale
from torchvision.models import vgg19_bn
from torchvision.datasets import MNIST
from torchsummary import summary
import matplotlib.pyplot as plt
from collections import OrderedDict
from numpy import uint8, array, transpose


local_device = device("cuda" if is_available() else "cpu")
model = vgg19_bn(num_classes=10).to(local_device)

transforms = Compose([
    RandomHorizontalFlip(),
    RandomVerticalFlip(),
    RandomRotation(30)
])

to_tensor = Compose([
    # 28 x 28 is too small vgg19. There will be no pixel left in the end.
    Resize((32, 32), antialias=True),
    # MNIST that used here is purely in black/white.
    # Thus, the images are in only 1 channel.
    # vgg19_bn only deal with RGB pictures, which are in 3 channels.
    # Hence, we transform them into 3 channels.
    # All 3 channels share the same element values here.
    Grayscale(3),
    ToImage(),
    ToDtype(float32, scale=True)
])

class MyPainter(QWidget):
    def __init__(self):
        super().__init__()
        # Track mouseMoveEvent only when clicking.
        self.setMouseTracking(False)
        self.setFixedSize(600, 400)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet("background-color:black;")
        # Delcare the array to collect all mouse positions.
        self.positions = []
        self.pos_xy = []

    def paintEvent(self, event):
        try:
            painter = QPainter()
            painter.begin(self)
            painter.setPen(QPen(Qt.white, 20, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))

            if len(self.positions) > 0:
                for pos_xy in self.positions:
                    if len(pos_xy) > 0:
                        point_start = pos_xy[0]
                        for point_end in pos_xy:
                            # The real draw function
                            painter.drawLine(point_start[0], point_start[1], point_end[0], point_end[1])
                            point_start = point_end

            if len(self.pos_xy) > 0:
                point_start = self.pos_xy[0]
                for point_end in self.pos_xy:
                    # The real draw function
                    painter.drawLine(point_start[0], point_start[1], point_end[0], point_end[1])
                    point_start = point_end
        except Exception as e:
            print(str(e))

        painter.end()

    def mouseReleaseEvent(self, event):
        self.positions.append(self.pos_xy)
        self.pos_xy = []

    def mouseMoveEvent(self, event):
        pos_tmp = (event.pos().x(), event.pos().y())
        self.pos_xy.append(pos_tmp)
        # self.update() will call self.paintEvent().
        # QPainter() must be called in the paintEvent().
        self.update()

class MyWidget(QGroupBox):
    def __init__(self):
        super().__init__()
        self.ui()

    def ui(self):
        self.setTitle("4. MNIST Classifier Using VGG19")
        layout = QHBoxLayout()
        layout1 = QVBoxLayout()
        layout2 = QVBoxLayout()

        button1 = QPushButton("4.1 Show Model Structure")
        button2 = QPushButton("4.2 Show Acc and Loss")
        button3 = QPushButton("4.3 Predict")
        button4 = QPushButton("4.4 Reset")
        self.label = QLabel("")
        self.label.setAlignment(Qt.AlignCenter)
        self.painter = MyPainter()

        layout1.addWidget(button1)
        layout1.addSpacing(10)
        layout1.addWidget(button2)
        layout1.addSpacing(10)
        layout1.addWidget(button3)
        layout1.addSpacing(10)
        layout1.addWidget(button4)
        layout1.addSpacing(10)
        layout1.addWidget(self.label)
        layout1.addSpacing(10)
        layout1.setAlignment(Qt.AlignVCenter)

        layout2.addWidget(self.painter)

        layout.addLayout(layout1)
        layout.addSpacing(20)
        layout.addLayout(layout2)
        self.setLayout(layout)

        button1.clicked.connect(self.show_model_structure)
        button2.clicked.connect(self.show_acc_and_loss)
        button3.clicked.connect(self.predict)
        button4.clicked.connect(self.reset)

        self.pos_xy = []

    def show_model_structure(self):
        # If input_data, in this case, it's its size, (3, 32, 32), is not provided, 
        # no forward pass through the network is performed, 
        # and the provided model information is limited to layer names.
        # An example input in torch.tensor is also valid as input_data.
        summary(model, (3, 32, 32))

    def show_acc_and_loss(self):
        try:
            img = cv2.imread("training_results.png")
            cv2.imshow("training results", img)
        except cv2.error:
            pass

    def predict(self):
        try:
            model.load_state_dict(load("best_vgg19_weights.pth"))
            model.eval()
            img = from_numpy(transpose(cv2.cvtColor(array(self.painter.grab().toImage().bits().asarray(600 * 400 * 4), dtype=uint8).reshape(400, 600, 4), cv2.COLOR_RGBA2RGB), (2, 0, 1)))

            with no_grad():
                # .unsqueeze(0) method is used to add a new dimension at the specified position in a tensor.
                # For consistency.
                output = model(to_tensor(img).to(local_device).unsqueeze(0))

            _, predicted = output.max(1)
            self.label.setText(str(predicted.item()))
            plt.figure(figsize=(8, 6))
            plt.bar(range(10), softmax(output[0], dim=0).cpu())
            plt.xticks(range(10))
            plt.xlabel('Class')
            plt.ylabel('Probability')
            plt.title('probability of each class')
            plt.show()
        except AttributeError as e:
            print(e)
        except FileNotFoundError:
            pass

    def reset(self):
        self.label.setText("")
        self.painter.positions.clear()
        # There shouldn't be value, but ... just in case.
        self.painter.pos_xy.clear()
        self.update()


if __name__ == '__main__':
    if len(sys.argv) == 1:
        # VGG19
        transforms = Compose([
            transforms,
            to_tensor,
        ])

        trainset = MNIST(root='./data', train=True, download=True, transform=transforms)
        trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4)

        testset = MNIST(root='./data', train=False, download=True, transform=to_tensor)
        testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=4)

        # Cross entropy loss is commonly used as a loss function in classification problems.
        # It measures the difference between the predicted probability distribution (output of the neural network) 
        # and the true probability distribution (the ground truth labels). 
        # The goal during training is to minimize this loss.
        criterion = CrossEntropyLoss()

        # ADAM: A METHOD FOR STOCHASTIC OPTIMIZATION
        # https://arxiv.org/pdf/1412.6980.pdf
        # a.k.a. Adaptive Moment Estimate
        # No need for momentum. Change learning rate adaptively.
        # 
        # VERY IMPORTANT NOTE:
        # After experiments, it's shown that the lr when using Adam should be way smaller than that when using SGD.
        optimizer = Adam(model.parameters(), lr=0.0001)

        num_epochs = 30
        best_accuracy = 0.0
        best_model_weights = OrderedDict()
        train_losses, train_accuracies, val_losses, val_accuracies = [], [], [], []

        for epoch in range(num_epochs):
            # train
            model.train()

            running_loss = 0.0
            correct = 0
            total = 0
            for inputs, labels in trainloader:
                inputs, labels = inputs.to(local_device), labels.to(local_device)

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

                # loss is now a torch.Tensor object,
                # which is in form "tensor(2.4382, device='cuda:0', grad_fn=<NllLossBackward0>)".
                # .item() converts the result to a Python number i.e. get the value above.
                running_loss += loss.item()
                # Extracts the class predictions by selecting the index with the maximum value
                # returns a tuple containing two tensors: 
                # the maximum value along dimension 1 (the predicted class scores) 
                # and the index of the maximum value (the predicted class labels).
                # outputs.shape = (64, 10)
                # That is, for each picture, the predict result for all classes.
                # outputs.max(1) means the maximums in every outputs[0] obejct.
                # It will return in tuple (value, index).
                # Thus, predicted.shape = (64), which is,
                # for every picture, which index (class) the model give the maximum value.
                # Likewise, labels.shape = (64), which is, for every picture, which index (class) is ground truth.
                # The 64 here is the max batch size used here. It could change with different context.
                _, predicted = outputs.max(1)
                total += labels.size(0)
                # predicted.eq(labels) is a boolean matrix that mark each object True/False 
                # due to whether they are in the same value.
                # Hence, predicted.eq(labels).shape = (64)
                # predicted.eq(labels).sum() is still a torch.Tensor object
                # i.e. is in form like "tensor(7, device='cuda:0')".
                # .item() converts the result to a Python number i.e. get the value above.
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
                for inputs, labels in testloader:
                    inputs, labels = inputs.to(local_device), labels.to(local_device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    running_loss += loss.item()
                    _, predicted = outputs.max(1)
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
                
        save(best_model_weights, "best_vgg19_weights.pth")
        
        plt.figure(figsize=(10, 5))
        plt.subplot(2, 1, 1)
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Val Loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.title("Loss")

        plt.subplot(2, 1, 2)
        plt.plot(train_accuracies, label="Train Acc")
        plt.plot(val_accuracies, label="Val Acc")
        plt.xlabel("epoch")
        plt.ylabel("accuracy(%)")
        plt.legend()
        plt.title("Accuracy")

        plt.tight_layout()
        plt.savefig("training_results.png")
    elif sys.argv[1] == "--show":
        app = QApplication(sys.argv)
        MainWindow = MyWidget()
        MainWindow.show()
        sys.exit(app.exec_())
    else:
        print("Invalid input.")
        x = 0
