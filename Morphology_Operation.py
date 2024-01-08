from PyQt5.QtWidgets import QApplication, QPushButton, QVBoxLayout, QGroupBox, QDesktopWidget, QLabel
import sys, cv2
import numpy as np
import matplotlib.pyplot as plt


def Erosion(img):
    zero_padding = np.zeros((img.shape[0] + 2, img.shape[1] + 2), dtype='uint8')
    zero_padding[1:-1, 1:-1] = img
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            img[x, y] = zero_padding[x : x + 3, y : y + 3].min()
    return img

def Dilation(img):
    zero_padding = np.zeros((img.shape[0] + 2, img.shape[1] + 2), dtype='uint8')
    zero_padding[1:-1, 1:-1] = img
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            img[x, y] = zero_padding[x : x + 3, y : y + 3].max()
    return img

class MyWidget(QGroupBox):
    def __init__(self):
        super().__init__()
       
        self.ui()

    def ui(self):
        self.setTitle("3. Morphology Operation")
        layout = QVBoxLayout()

        button1 = QPushButton("3.1 Closing")
        button2 = QPushButton("3.2 Opening")

        layout.addSpacing(20)
        layout.addWidget(button1)
        layout.addSpacing(30)
        layout.addWidget(button2)
        layout.addSpacing(20)

        self.setLayout(layout)

        button1.clicked.connect(self.closing)
        button2.clicked.connect(self.opening)

    def closing(self):
        try:
            # The order:
            # cv2.imread(self.filename): read the file
            # cv2.cvtColor(cv2.imread(self.filename), cv2.COLOR_BGR2GRAY): turn it gray.
            # cv2.threshold(cv2.cvtColor(cv2.imread(self.filename), cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)[1]: turn it binary,
            # and get the second value for the first value is the threshold.
            # Then, dilate and erose it. Done.
            cv2.imshow('closing', Erosion(Dilation(cv2.threshold(cv2.cvtColor(cv2.imread(self.filename), cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)[1])))
        except AttributeError as e:
            # Image not loaded.
            pass

    def opening(self):
        try:
            cv2.imshow('opening', Dilation(Erosion(cv2.threshold(cv2.cvtColor(cv2.imread(self.filename), cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)[1])))
        except AttributeError:
            # Image not loaded.
            pass


if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = MyWidget()
    MainWindow.show()
    sys.exit(app.exec_())
