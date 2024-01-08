from PyQt5.QtWidgets import QApplication, QPushButton, QVBoxLayout, QGroupBox, QDesktopWidget
import sys, cv2
import numpy as np
import matplotlib.pyplot as plt


class MyWidget(QGroupBox):
    def __init__(self):
        super().__init__()
       
        self.ui()

    def ui(self):
        self.setTitle("2. Histogram Equalization")
        layout = QVBoxLayout()

        button = QPushButton("2. Histogram Equalization")

        layout.addSpacing(20)
        layout.addWidget(button)
        layout.addSpacing(20)

        self.setLayout(layout)

        button.clicked.connect(self.histogram_equalization)

    def histogram_equalization(self):
        try:
            _, axes = plt.subplots(2, 3, figsize=(10, 5))

            # While the picture read in by cv2.imread() is stored in 'BGR', the picture shown by plt.imshow() is treated as in 'RGB'.
            cimg = cv2.imread(self.filename)
            axes[0][0].imshow(cv2.cvtColor(cimg, cv2.COLOR_BGR2RGB))
            axes[0][0].set_title('Original Image')
            axes[0][0].axis('off')

            # bins = 0 ~ 256
            # so x = 0 ~ 256
            y, x = np.histogram(cv2.cvtColor(cimg, cv2.COLOR_BGR2GRAY).flatten(), bins=range(257))
            # x will have one element more than y, and that element is the upper bound. In this case, x[-1] = 256.
            # This is because the last bin is [255, 256) to include 255.
            axes[1][0].bar(x[:-1], y)
            axes[1][0].set_xlabel('Gray Scale')
            axes[1][0].set_ylabel('Frequency')
            axes[1][0].set_title('Histogram of Original')

            cdf = y.cumsum() # cumulative sum
            # For each cdf element, multiply it with the maximum of y, and divide it with the sum of the image.
            cdf_normalized = cdf * float(y.max()) / cdf.max()
            cdf_m = np.ma.masked_equal(cdf, 0) # exclude 0
            # Normalization:
            # For each cdf_m element, subtract it with the minimum of cdf_m, multiply it with 255, and
            # divide it with the range of cdf_m.
            cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
            cdf = np.ma.filled(cdf_m, 0).astype('uint8') # Give the 0 back.

            # Now, Histogram Equalization of cimg is cdf[cimg].
            img = cdf[cv2.cvtColor(cimg, cv2.COLOR_BGR2GRAY)]
            axes[0][2].imshow(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
            axes[0][2].set_title('Equalized Manually')
            axes[0][2].axis('off')

            y, x = np.histogram(img.flatten(), bins=range(257))
            axes[1][2].bar(x[:-1], y)
            axes[1][2].set_xlabel('Gray Scale')
            axes[1][2].set_ylabel('Frequency')
            axes[1][2].set_title('Histogram of Equalized (Manual)')
            array1 = y.copy()


            img = cv2.equalizeHist(cv2.cvtColor(cimg, cv2.COLOR_BGR2GRAY))
            axes[0][1].imshow(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
            axes[0][1].set_title('Equalized with OpenCV')
            axes[0][1].axis('off')

            y, x = np.histogram(img.flatten(), bins=range(257))
            axes[1][1].bar(x[:-1], y)
            axes[1][1].set_xlabel('Gray Scale')
            axes[1][1].set_ylabel('Frequency')
            axes[1][1].set_title('Histogram of Equalized (OpenCV)')

            plt.tight_layout()
            plt.show()

        except AttributeError as e:
            # Image not loaded.
            pass


if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = MyWidget()
    MainWindow.show()
    sys.exit(app.exec_())