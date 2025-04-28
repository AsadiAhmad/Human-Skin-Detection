# Human-Skin-Detection
Human Skin Detection Using RGB, HSV and YCbCr Color. this code is all algorithmic way to detect human skin and best for the preprocessing of the Deep learning as zero shot methods. This method is implemented from a paper!

<img src="/Pictures/Output/jensen_huang.jpg"/>

## Tech :hammer_and_wrench: Languages and Tools :

<div>
  <img src="https://github.com/devicons/devicon/blob/master/icons/python/python-original.svg" title="Python" alt="Python" width="40" height="40"/>&nbsp;
  <img src="https://github.com/devicons/devicon/blob/master/icons/jupyter/jupyter-original.svg" title="Jupyter Notebook" alt="Jupyter Notebook" width="40" height="40"/>&nbsp;
  <img src="https://assets.st-note.com/img/1670632589167-x9aAV8lmnH.png" title="Google Colab" alt="Google Colab" width="40" height="40"/>&nbsp;
  <img src="https://github.com/devicons/devicon/blob/master/icons/opencv/opencv-original.svg" title="OpenCV" alt="OpenCV" width="40" height="40"/>&nbsp;
  <img src="https://github.com/devicons/devicon/blob/master/icons/numpy/numpy-original.svg" title="Numpy" alt="Numpy" width="40" height="40"/>&nbsp;
  <img src="https://github.com/devicons/devicon/blob/master/icons/matplotlib/matplotlib-original.svg"  title="MatPlotLib" alt="MatPlotLib" width="40" height="40"/>&nbsp;
</div>

- Python : Popular language for implementing Neural Network
- Jupyter Notebook : Best tool for running python cell by cell
- Google Colab : Best Space for running Jupyter Notebook with hosted server
- OpenCV : Best Library for working with images
- Numpy : Best Library for working with arrays in python
- MatPlotLib : Library for showing the charts in python

## Run the Notebook on Google Colab

You can easily run this code on google colab by just clicking this badge [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AsadiAhmad/Human-Skin-Detection/blob/main/Code/Human-Skin-Detector.ipynb)

## Tutorial

### Step 1: Import Libraries

we need to import these libraries :

`cv2`, `numpy`, `matplotlib`

```sh
import cv2
import numpy as np
import matplotlib.pyplot as plt
```

### Step 2: Download Images

We need to Download the images from my `Github` repository or you can download other human pictures by your own.

```sh
!wget https://raw.githubusercontent.com/AsadiAhmad/Human-Skin-Detection/main/Pictures/Input/jensen_huang.jpg -O jensen_huang.jpg
!wget https://raw.githubusercontent.com/AsadiAhmad/Human-Skin-Detection/main/Pictures/Input/elon_musk.jpg -O elon_musk.jpg
!wget https://raw.githubusercontent.com/AsadiAhmad/Human-Skin-Detection/main/Pictures/Input/mark_zukerberg.jpg -O mark_zukerberg.jpg
!wget https://raw.githubusercontent.com/AsadiAhmad/Human-Skin-Detection/main/Pictures/Input/linus_torvalds.jpg -O linus_torvalds.jpg
```

### Step 3: Load Images

We need to load images into `python` variables we ues `OpenCV` library to read the images also the format of the images are `nd.array`.

```sh
bgr_image = cv2.imread('jensen_huang.jpg')
elon_musk = cv2.imread('elon_musk.jpg')
mark_zukerberg = cv2.imread('mark_zukerberg.jpg')
linus_torvalds = cv2.imread('linus_torvalds.jpg')
```

### Step 4: Extract HSV and YCrCb color formats

In this algorithmic approch we need to get teh HSV and YCrCb color formats for calculating the conditions

```sh
hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
ycrcb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2YCrCb)
```

HSV color format :

<div display=flex align=center>
  <img src="/Pictures/Colors/HSV.jpg" width="400px"/>
</div>

YCbCr color format :

<div display=flex align=center>
  <img src="/Pictures/Colors/YCbCr.jpg" width="400px"/>
</div>

### Step 5: Split Image Channels

For calculating the conditions we need to split each channel from each color space.

```sh
B, G, R = cv2.split(bgr_image)
H, S, V = cv2.split(hsv_image)
Y, Cr, Cb = cv2.split(ycrcb_image)
```

### Step 6: Apply Conditions

We should apply all conditions from each color space. if each pixel pass all conditions it means that pixel is a skin pixel.

```sh
def conditions(r, g, b, h, s, v, y, cr, cb):
    s = s / 255.0
    condition_rgb = (r > 95) and (g > 40) and (b > 20) and (r > g) and (r > b) and (abs(r-g) > 15)
    condition_hsv = (0 <= h <= 50) and (0.23 <= s <= 0.68)
    condition_ycrcb = (cr > 135) and (cb > 85) and (y > 80) and (cr <= (1.5862*cb)+20) and (cr>=(0.3448*cb)+76.2069) and (cr >= (-4.5652*cb)+234.5652) and (cr <= (-1.15*cb)+301.75) and (cr <= (-2.2857*cb)+432.85)
    return condition_rgb and condition_hsv and condition_ycrcb
```

```sh
height, width, channels = bgr_image.shape
image = np.zeros((height, width), np.uint8)
main_image = bgr_image.copy()
for i in range(height):
    for j in range(width):
        if conditions(R[i, j], G[i, j], B[i, j], H[i, j], S[i, j], V[i, j], Y[i, j], Cr[i, j], Cb[i, j]):
            image[i, j] = 255
        else:
            main_image[i, j] = [0, 0, 0]
```

### Step 7: Plot images

Now we need to see what have done to images.

```sh
fig, axs = plt.subplots(1, 3, figsize=(12, 4))

axs[0].imshow(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB))
axs[0].set_title('Original Image')
axs[0].axis('off')

axs[1].imshow(image, cmap='gray')
axs[1].set_title('Skin Mask')
axs[1].axis('off')

axs[2].imshow(cv2.cvtColor(main_image, cv2.COLOR_BGR2RGB))
axs[2].set_title('Detected Skin Areas')
axs[2].axis('off')

plt.tight_layout()
plt.show()
```

<div display=flex align=center>
  <img src="/Pictures/Output/jensen_huang.jpg" width="400px"/>
</div>

### Step 8: Put all together

So in this step we put all of things together and test that for other pictures.

```sh
def skin_detector(bgr_image):
    hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    ycrcb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2YCrCb)
    B, G, R = cv2.split(bgr_image)
    H, S, V = cv2.split(hsv_image)
    Y, Cr, Cb = cv2.split(ycrcb_image)
    height, width, channels = bgr_image.shape
    image = np.zeros((height, width), np.uint8)
    main_image = bgr_image.copy()
    for i in range(height):
        for j in range(width):
            if conditions(R[i, j], G[i, j], B[i, j], H[i, j], S[i, j], V[i, j], Y[i, j], Cr[i, j], Cb[i, j]):
                image[i, j] = 255
            else:
                main_image[i, j] = [0, 0, 0]

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    axs[0].imshow(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB))
    axs[0].set_title('Original Image')
    axs[0].axis('off')

    axs[1].imshow(image, cmap='gray')
    axs[1].set_title('Skin Mask')
    axs[1].axis('off')

    axs[2].imshow(cv2.cvtColor(main_image, cv2.COLOR_BGR2RGB))
    axs[2].set_title('Detected Skin Areas')
    axs[2].axis('off')

    plt.tight_layout()
    plt.show()
```

<img src="/Pictures/Output/elon_musk.jpg"/>

<img src="/Pictures/Output/mark_zukerberg.jpg"/>

<img src="/Pictures/Output/linus_torvalds.jpg"/>

## License

This project is licensed under the MIT License.
