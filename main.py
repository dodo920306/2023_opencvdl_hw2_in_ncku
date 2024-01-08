from PyQt5.QtWidgets import QApplication, QWidget, QDesktopWidget, QPushButton, QVBoxLayout, QGroupBox, QHBoxLayout, QFileDialog, QLabel
from PyQt5.QtCore import Qt
from Hough_Circle_Transformation import MyWidget as GroupBox1
from Histogram_Equalization import MyWidget as GroupBox2
from Morphology_Operation import MyWidget as GroupBox3
from VGG19 import MyWidget as GroupBox4
from ResNet50 import MyWidget as GroupBox5
import sys


class MyWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hw2")
        self.ui()
        screen_geo = QDesktopWidget().screenGeometry()
        widget_geo = self.geometry()
        x = (screen_geo.width() - widget_geo.width()) // 2 - 280
        y = (screen_geo.height() - widget_geo.height()) // 2 - 200

        self.move(x, y)

    def ui(self):
        mainLayout = QHBoxLayout()
        layout1 = QVBoxLayout()
        layout2 = QVBoxLayout()
        layout3 = QVBoxLayout()

        button = QPushButton("Load Image")
        self.label = QLabel("No image loaded")

        self.groupBox1 = GroupBox1()
        self.groupBox2 = GroupBox2()
        self.groupBox3 = GroupBox3()
        self.groupBox4 = GroupBox4()
        self.groupBox5 = GroupBox5()

        layout1.addWidget(button)
        layout1.addWidget(self.label)
        layout1.setAlignment(Qt.AlignVCenter)

        layout2.addWidget(self.groupBox1)
        layout2.addWidget(self.groupBox2)
        layout2.addWidget(self.groupBox3)
        
        layout3.addWidget(self.groupBox4)
        layout3.addWidget(self.groupBox5)
        layout3.setAlignment(Qt.AlignTop)

        mainLayout.addLayout(layout1)
        mainLayout.addSpacing(20)
        mainLayout.addLayout(layout2)
        mainLayout.addSpacing(20)
        mainLayout.addLayout(layout3)
        mainLayout.setContentsMargins(50, 20, 50, 20)

        self.setLayout(mainLayout)

        button.clicked.connect(self.load_image)

    def load_image(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        filename, _ = QFileDialog.getOpenFileName(self, "Select an Image", "", "Images (*.png *.jpg)", options=options)
        if filename != "":
            self.label.setText(filename.split('/')[-1])
            self.groupBox1.filename = filename
            self.groupBox2.filename = filename
            self.groupBox3.filename = filename


if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = MyWidget()
    MainWindow.show()
    sys.exit(app.exec_())
