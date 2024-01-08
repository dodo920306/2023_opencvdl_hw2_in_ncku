from PyQt5.QtWidgets import QApplication, QPushButton, QVBoxLayout, QGroupBox, QDesktopWidget, QLabel
import sys, cv2
import numpy as np
import matplotlib.pyplot as plt


class MyWidget(QGroupBox):
    def __init__(self):
        super().__init__()
       
        self.ui()

    def ui(self):
        self.setTitle("1. Hough Circle Transform")
        layout = QVBoxLayout()

        button1 = QPushButton("1.1 Draw Contour")
        button2 = QPushButton("1.2 Count Coins")
        self.label = QLabel("There are _ coins in the image.")

        layout.addSpacing(20)
        layout.addWidget(button1)
        layout.addSpacing(30)
        layout.addWidget(button2)
        layout.addSpacing(30)
        layout.addWidget(self.label)
        layout.addSpacing(20)

        self.setLayout(layout)

        button1.clicked.connect(self.draw_contour)
        button2.clicked.connect(self.count_coins)

    def draw_contour(self):
        try:
            _, axes = plt.subplots(1, 3, figsize=(10, 4))
            axes.ravel()[0].set_title("Img_src")
            axes.ravel()[1].set_title("Img_process")
            axes.ravel()[2].set_title("Circle_center")

            cimg = cv2.imread(self.filename)
            # While the picture read in by cv2.imread() is stored in 'BGR', the picture shown by plt.imshow() is treated as in 'RGB'.
            axes.ravel()[0].imshow(cv2.cvtColor(cimg, cv2.COLOR_BGR2RGB))
            img = cv2.cvtColor(cimg, cv2.COLOR_BGR2GRAY)
            img = cv2.GaussianBlur(img, (5, 5), 0)
            # (image, method, dp, minDist, param1 = 100, param2 = 100, minRadius = 0, maxRadius = 0)
            # 
            # dp is the inverse ratio of the accumulator resolution to the image resolution.
            # The smaller the circles needed to be detected is, the greater the value should be.
            # 
            # minDist is the minimum distance between the centers of the detected circles.
            # 
            # param1 is the higher threshold of the two passed to the Canny edge detector (the lower one is twice smaller).
            # 
            # param2 is the accumulator threshold for the circle centers at the detection stage.
            # The smaller it is, the more false circles may be detected. 
            # Circles, corresponding to the larger accumulator values, will be returned first. 
            circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 40, param1=100, param2=40, minRadius=20, maxRadius=30)
            # circles = (1, # of circles, |(x, y, r)|)
            circles = np.uint16(np.around(circles))

            img = np.zeros_like(img, dtype='uint8')

            for i in circles[0,:]:
                # draw the outer circle
                # (image, center_coordinates, radius, color, thickness)
                cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
                cv2.circle(img, (i[0], i[1]), 2, 255, 2)

            axes.ravel()[1].imshow(cv2.cvtColor(cimg, cv2.COLOR_BGR2RGB))
            axes.ravel()[2].imshow(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))

            plt.tight_layout()
            plt.show()

        except AttributeError as e:
            # Image not loaded.
            pass

    def count_coins(self):
        try:
            cimg = cv2.imread(self.filename)
            img = cv2.cvtColor(cimg, cv2.COLOR_BGR2GRAY)
            img = cv2.GaussianBlur(img, (5, 5), 0) 
            circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 40, param1=100, param2=45, minRadius=20, maxRadius=30)
            circles = np.uint16(np.around(circles))
            if circles.shape[1] == 0 or circles.shape[1] == 1:
                self.label.setText("There is %d coin in the image." % circles.shape[1])
            else:
                self.label.setText("There are %d coins in the image." % circles.shape[1])
        except AttributeError:
            # Image not loaded.
            pass


if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = MyWidget()
    MainWindow.show()
    sys.exit(app.exec_())
