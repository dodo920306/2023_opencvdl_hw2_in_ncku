# Opencvdl HW2 In NCKU CSIE, 2023

Implementation of a personal solution of the second homework in Introduction to Image Processing, Computer Vision, and Deep Learning courses in NCKU CSIE, 2023.

## Prerequisite

* Python 3.8

* Pip

## Environment

* Windows 11
* Ubuntu 20.04 WSL

## Get Started

Use

```bash
$ git clone https://github.com/dodo920306/2023_opencvdl_hw2_in_ncku.git
$ pip install -r requirements.txt
```

to clone the repo and install the prerequisites.

Run

```bash
$ ldd ~/.local/lib/python3.8/site-packages/PyQt5/Qt5/plugins/platforms/libqxcb.so | grep "not found"
```

to check if there are dependent shared libraries missing. It's usual that there's a lot of them.

Use

```bash
$ sudo apt update && sudo apt install <missing shared libraries> -y
```

to collect them.

For example, if you get

```bash
$ ldd ~/.local/lib/python3.8/site-packages/PyQt5/Qt5/plugins/platforms/libqxcb.so | grep "not found"
    libxcb-icccm.so.4 => not found
    libxcb-image.so.0 => not found
```

, run

```bash
$ sudo apt update && sudo apt install libxcb-icccm4 libxcb-image0 -y
```

in response.

The environment I used for develop this is WSL on Windows 11. If you're doing the same, please use

```bash
$ ipconfig
```

on Windows host to check its IP for the WSL.

You should be able to ping that IP from WSL. If you can't, please run

```bash
$ New-NetFirewallRule -DisplayName "WSL" -Direction Inbound -InterfaceAlias "vEthernet (WSL)" -Action Allow
```

as the administrator on Windows and try again.

Finally, run

```bash
$ python main.py
```

to start the program. You should see the window pop up on your screen.

You may encouter `Segmentation fault` when closing the window, it's normal. If you know how to fix it, please send pull request for it because I'm not sure how to solve that.

If you can actually ping Windows from WSL but still can't run main.py, please update your wsl, restart it, and try again.

## Functionality

Once you run main.py successfully, you should see some UI like this

<img src="image.png" width="400"/>

As you can see, the features are divided into 5 main parts: Hough Circle Transformation, Histogram Equalization, Morphology Operation, VGG19, ResNet50

### Load Image

The button will load a picture from the computer, enabling some of the following operation.

### Hough Circle Transformation

There are 2 buttons can be clicked providing 2 different features:

1. Draw contour for the circles on the loaded picture.

2. Count the number of circles on the loaded picture.

Load `Dataset_OpenCvDl_Hw2/Q1/coins.jpg` for a simple illustration.

run

```bash
$ python Hough_Circle_Transformation.py
```

to get this part of UI independently.

### Histogram Equalization

There are 1 buttons can be clicked providing 1 feature:

1. Apply 2 histogram Equalizations on the loaded picture to enhance clarity, 1 is by OpenCV `cv2.equalizeHist()`, the other is done manually.

Load `Dataset_OpenCvDl_Hw2/Q2/histoEqualGray2.png` for a simple illustration.

run

```bash
$ python Histogram_Equalization.py
```

to get this part of UI independently.


### Morphology Operation

There are 2 buttons can be clicked providing 2 different features:

1. Apply dilation and erosion to the loaded picture to fill loopholes on it.

2. Apply dilation and erosion to the loaded picture to wash away stains on it.

Load `Dataset_OpenCvDl_Hw2/Q3/closing.png` and `Dataset_OpenCvDl_Hw2/Q3/opening.png` for a simple illustration.

run

```bash
$ python Morphology_Operation.py
```

to get this part of UI independently.

### MNIST Classifier Using VGG19

There are 4 buttons can be clicked providing 4 different features:

1. Show the model structure of this MNIST Classifier Using VGG19.

2. Show Training/Validating Accuracy and Loss

3. Predict the number drawn on the canvas aside.

4. Reset the canvas.

Draw a number on the canvas aside for a simple illustration.

run

```bash
$ python VGG19.py
```

and type 1 to get this part of UI independently, or type 2 to train your own model with the CIFAR-10 dataset.

### ResNet50

The picture here has to be loaded with a seperated button.

There are 4 buttons can be clicked providing 4 different features:

1. Show random pictures from `inference_dataset/Cat` and `inference_dataset/Dog` respectively.

2. Show the model structure of this ResNet50.

3. Show the accuracy with and without random erasing.

4. Inference the loaded picture.

Load a picture from `inference_dataset/` for a simple illustration.

run

```
$ python ResNet50.py
```

to get this part of UI independently.

run

```
$ python train.py
```

to train this.

> Pictures in `Dataset_OpenCvDl_Hw2_Q5/` are used to train this.
