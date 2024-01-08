from PyQt5.QtWidgets import QApplication, QPushButton, QHBoxLayout, QVBoxLayout, QGroupBox, QLabel, QWidget, QFileDialog
from PyQt5.QtGui import QPixmap
import sys, os
from torch.cuda import is_available
from torch import device, float32, load, no_grad
from torch.nn import Linear
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.v2 import Compose, ToImage, ToDtype, Resize
from torchvision.models import resnet50
from torchsummary import summary
import matplotlib.pyplot as plt
from PIL import Image
from numpy import transpose
from PyQt5.QtCore import Qt
from cv2 import imread, imshow


local_device = device("cuda" if is_available() else "cpu")
model = resnet50()
model.fc = Linear(model.fc.in_features, 1)
model = model.to(local_device)

to_tensor = Compose([
    Resize((224, 224), antialias=True),
    # ToImage() transforms PIL Image into torch.Tensor subclass for images.
    ToImage(),
    ToDtype(float32, scale=True)
])

def collate_fn(batch):
    selected_data = {}
    label_0_selected = False
    label_1_selected = False

    for item in batch:
        img, label = item

        if label == 0 and not label_0_selected:
            selected_data[0] = img
            label_0_selected = True
        elif label == 1 and not label_1_selected:
            selected_data[1] = img
            label_1_selected = True

        if label_0_selected and label_1_selected:
            break

    return selected_data

class MyDataset(Dataset):
    def __init__(self, transform=None):
        self.img_labels = []
        file_list = []
        labels = []
        folder_path = "inference_dataset/"

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

    def __len__(self):
        return len(self.img_labels[0])

    def __getitem__(self, idx):
        label = self.img_labels[1][idx]
        folder_path = "inference_dataset/"
        folder_path += "Cat" if label == 0 else "Dog"
        img_path = os.path.join(folder_path, self.img_labels[0][idx])

        # There is a chance that the image isn't seen like .jpeg to read_image(), so
        # use PIL.Image to convert it first.
        # There is to_tensor in transform, so no need to do it here.
        # Don't save to prevent I/O problem.
        image = Image.open(img_path).convert("RGB")

        image = self.transform(image)
        return image, label

class MyWidget(QGroupBox):
    def __init__(self):
        super().__init__()
        self.ui()

    def ui(self):
        self.setTitle("5. ResNet50")
        layout = QHBoxLayout()
        layout1 = QVBoxLayout()
        layout2 = QVBoxLayout()

        button0 = QPushButton("Load Image")
        button1 = QPushButton("5.1 Show Images")
        button2 = QPushButton("5.2 Show Model Structure")
        button3 = QPushButton("5.3 Show Comparison")
        button4 = QPushButton("5.4 Inference")
        self.image = QLabel()
        self.label = QLabel("Predict = ")
        self.label.setAlignment(Qt.AlignCenter)
        self.image.setFixedSize(224, 224)

        layout1.addWidget(button0)
        layout1.addSpacing(10)
        layout1.addWidget(button1)
        layout1.addSpacing(10)
        layout1.addWidget(button2)
        layout1.addSpacing(10)
        layout1.addWidget(button3)
        layout1.addSpacing(10)
        layout1.addWidget(button4)
        layout1.setAlignment(Qt.AlignVCenter)

        layout2.addWidget(self.image)
        layout2.addWidget(self.label)
        layout2.setAlignment(Qt.AlignCenter)

        layout.addLayout(layout1)
        layout.addSpacing(20)
        layout.addLayout(layout2)
        self.setLayout(layout)

        self.setLayout(layout)

        button0.clicked.connect(self.load_image)
        button1.clicked.connect(self.show_images)
        button2.clicked.connect(self.show_model_structure)
        button3.clicked.connect(self.show_comparison)
        button4.clicked.connect(self.inference)

    def show_images(self):
        testset = MyDataset(transform=to_tensor)
        testloader = DataLoader(testset, batch_size=32, shuffle=True, collate_fn=collate_fn)
        _, axes = plt.subplots(ncols=2, figsize=(8, 3))
        axes[0].imshow(transpose(next(iter(testloader))[0], (1,2,0)))
        axes[1].imshow(transpose(next(iter(testloader))[1], (1,2,0)))
        axes[0].set_title("Cat")
        axes[1].set_title("Dog")
        axes[0].axis('off')
        axes[1].axis('off')

        plt.tight_layout()
        plt.show()

    def show_model_structure(self):
        summary(model, (3, 224, 224))

    def show_comparison(self):
        try:
            imshow("Figure 1", imread("Comparison.png"))
        except FileNotFoundError:
            pass

    def inference(self):
        try:
            self.label.setText("Predict = ")
            classes = ["Cat", "Dog"]
            model.load_state_dict(load("best_resnet18_weights.pth"))
            model.eval()
            img = Image.open(self.filename)

            with no_grad():
                # .unsqueeze(0) method is used to add a new dimension at the specified position in a tensor.
                # Here it's for adding the batch dimension for consistency since the model expects (1, 3, 32, 32) not (3, 32, 32).
                output = model(to_tensor(img).to(local_device).unsqueeze(0))

            predicted = (output > 0.5).int()
            self.label.setText("Predict = " + str(classes[predicted.item()]))
        except AttributeError:
            pass
        except FileNotFoundError:
            pass

    def load_image(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        filename, _ = QFileDialog.getOpenFileName(self, "Select an Image", "", "Images (*.png *.jpg)", options=options)
        if filename != "":
            self.filename = filename
            pixmap = QPixmap(self.filename)
            self.image.setPixmap(pixmap.scaled(224, 224))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = MyWidget()
    MainWindow.show()
    sys.exit(app.exec_())
