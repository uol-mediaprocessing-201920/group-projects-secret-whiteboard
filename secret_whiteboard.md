Abgeschlossen
- Code Refactoring
- Closing im Preprocessing
- Contour Detection neugeschrieben, basierend auf Hierarchy Tree Depth
- Implementierung einer Filtermaske für die Sub-Images
- Blurring durch Schwärzen ersetzt aufgrund von besserer Performance

Sind dabei
- Parameter anpassen
- Contours, die in die Margin reichen, nicht zählen lassen
- Doku schreiben

Zukünftig
- Videos rendern

--------------------

# Confidential Whiteboard Contents

## Team Members

Fenno Boomgaarden (Fach-Bachelor Informatik)

Hauke Redemann (Fach-Bachelor Informatik)

Keno Rott (Fach-Bachelor Informatik)

# Project Overview

Handwritten notes on whiteboards or blackboards can be sensitive or contain confidential parts. To protect handwritten notes like this, the goal of our project was to develop a program that finds a specified pattern in images and blurs its content to be unrecognizable.
Our pattern is a rectangle containing a slightly smaller dashed rectangle with the confidential texts inside.

![](https://raw.githubusercontent.com/uol-mediaprocessing/group-projects-secret-whiteboard/master/img/doc_images/sample0.jpg)

# Use cases

## Lecture recording software

Some universities (like the Carl von Ossietzky university in Oldenburg) use **automated systems** for lecture recording und uploading as a service for students. Since the lectures may contain **sensitive data** that should not be public, like access credentials to private webservers, copyrighted information or solutions for exam exercises (so they can be used again in the future), providing a way to hide sensitive information **without manually editing** the video would be helpful.

## Privacy in real-life video streaming (possible in the future when livestreams are implemented as the input)

When livestreaming videogames, with common screen-capturing softwares like OBS it is possible to select which windows with non-sensitive information are captured. When doing streaming in real-life there are **no mechanisms** like that and the only option to protect the personal information of the streamer and/or others is **image processing**.

# 0. Project Setup

## Install dependencies

```python=
!pip install opencv-python==4.1.1.26
```

## Imports and constant definition

```python=
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from skimage import io
import math
import time

cv.DRAW_ALL_CONTOURS = -1
```

## Data Model

The `Image` class is the base class for all other classes in the data model. Its `data` attribute contains image data as a numpy array.

```python=
class Image:
    def __init__(self, data):
        self.data = data
```

The following classes are used throughout the project and represent the data structure. All images to be checked are stored in `Frame` objects. A `Frame` might contain several rectangular shapes that may contain the secret pattern. These shapes are stored in `SubImage` objects. While a `SubImage` is checked against our algorithm, we need to extract several areas into `Slice` objects, as defined below.

```python=
class Frame(Image):
    def __init__(self, data):
        super().__init__(cv.cvtColor(data, cv.COLOR_BGR2GRAY))
        self.original_data = data
        self.contours = None
        self.rectangles = None
        self.sub_images = []

class SubImage(Image):
    def __init__(self, data, rectangle):
        super().__init__(data)
        self.rectangle = rectangle
        self.slices = []

class Slice(Image):
    def __init__(self, data):
        super().__init__(data)
        self.line_data = None
```

## Read images
At first we need to download the test samples and turn them into `Frame` objects. The following method makes use of scikit-image's `io.imread()` for easy handling of local and remote URLs.

```python=
def create_frame(path):
    return Frame(io.imread(path))
```

All frames are stored in an array called `frames`.

```python=
paths = [
    # image urls
]

frames = [create_frame(path) for path in paths]
```

# 1. Preprocessing

The idea behind this step is to filter out as much noise as possible and differentiate the drawing from the whiteboard background. A good method for whiteboards is applying a [Gaussian Adaptive Threshold](https://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html) with a relatively high block size. To close small gaps between adjacent whiteboard marker strokes (e. g. a rectangle that is not closed properly) a [Closing](https://docs.opencv.org/master/d9/d61/tutorial_py_morphological_ops.html) operation is applied as well.

```python=
def apply_threshold(data):
    rows, cols = data.shape
    out = cv.GaussianBlur(data, (5, 5), 0)
    out = cv.adaptiveThreshold(out, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 201, 8)
    out = cv.morphologyEx(out, cv.MORPH_CLOSE, np.ones((int(0.0005 * rows) * 2 + 1, int(0.0005 * cols) * 2 + 1), np.uint8), iterations=5)
    return out

for frame in frames:
    frame.data = apply_threshold(frame.data)
```

# 2. Contour detection

Because OpenCV's [Contour Detection](https://docs.opencv.org/master/d4/d73/tutorial_py_contours_begin.html) by itself would detect an extremely large amount of contours for each frame, a pre-filtering method is defined. As each contour has a specific position in the hierarchy tree, we can follow the path of each node back to the highest parent available. Thereby, a depth is calculated for each contour.

As shown in the next figure, an inner contour of a shape always has an uneven depth in the tree. Additionally, we only need inner contours for further processing. Because of that, only contours with an even depth are included in the returned array.

![](https://raw.githubusercontent.com/uol-mediaprocessing/group-projects-secret-whiteboard/master/img/doc_images/contour_counter.png)

```python=
def find_contours_filtered(data):
    
    contours, hierarchy = cv.findContours(data, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    result = []
    
    for i, element in list(enumerate(hierarchy[0])):
        parent = element[3]
        
        # Traverse the hierarchy back to the root parent
        # and calculate the depth.
        counter = 0
        while parent != -1:
            parent = hierarchy[0][parent][3]
            counter += 1
        
        if counter % 2 != 0 and element[2] >= 0:
            result.append(contours[i])
        
    return result

for frame in frames:
    frame.contours = find_contours_filtered(frame.data)
```

# 3. Rectangle detection

In this step, a rectangle detection is performed to gather coordinates of potential matches for the secret pattern. To detect a rectangle, all filtered contours are approximated with less vertices. If an approximated contour consists of exactly four vertices and is convex, it is classified as a rectangle and included in the `frame.rectangles` attribute for further processing.

```python=
def detect_rectangles(contours, accuracy):
    approx_contours = []
    original_contours = []
    for contour in contours:
        approx = cv.approxPolyDP(contour, cv.arcLength(contour, True) * accuracy, True)
        if len(approx) == 4 and cv.isContourConvex(approx):
            approx_contours.append(approx)
            original_contours.append(contour)
    return approx_contours, original_contours

for frame in frames:
    frame.rectangles = detect_rectangles(frame.contours, 0.05)
```

# 4. Detection of dashed lines

For each detected rectangle, the presence or absence of a dashed line inside the contours has to be determined.

## Create transformed sub-images

Transform the content of every detected rectangle to a axis aligned new image

```python=
def create_transformed_sub_image(frame_data, rectangle, size):
    rows, cols = frame_data.shape
    rect = rectangle[0]
    #pts1 = order_points([rect[0][0], rect[1][0], rect[2][0], rect[3][0]])
    pts1 = np.float32(rect)
    #pts2 = np.float32([[0, 0], [cols, 0], [cols, rows], [0, rows]])
    pts2 = np.float32([[0, 0], [size, 0], [size, size], [0, size]])
    
    dst = []
        
    data = np.zeros(frame_data.shape, dtype="uint8")
    cv.drawContours(data, [rectangle[1]], cv.DRAW_ALL_CONTOURS, 1, cv.FILLED)
    data = cv.erode(data, np.ones((int(0.0050 * rows) * 2 + 1, int(0.0050 * cols) * 2 + 1), np.uint8))

    data = cv.bitwise_and(frame_data, data)

    m = cv.getPerspectiveTransform(pts1, pts2)
    dst = cv.warpPerspective(data, m, (size, size))

    return dst


for i, frame in enumerate(frames):
    for rectangle in zip(frame.rectangles[0],frame.rectangles[1]):
        data = create_transformed_sub_image(frame.data, rectangle, 500)
        if len(data) > 0:
            frame.sub_images.append(SubImage(data, rectangle))
```


## Slice Extraction 
For every edge of the detected rectangles, extract a slice margin from the transformed subimage. 

```python=
def extract_slices(data, margin, sub_margin):
    v_length = len(data)
    h_length = len(data[0])
    left = data[0:v_length, 0:int(v_length * margin)]
    left = np.transpose(left)
    top = data[0:int(h_length * margin), 0:h_length]
    right = data[0:v_length, h_length-int(v_length * margin):h_length]
    right = np.transpose(right)
    bottom = data[v_length-int(h_length * margin):v_length, 0:h_length]
    
    slices = [left, top, right, bottom]
    for i, s in enumerate(slices):
        slice_length = len(s)
        slices[i] = s[int(slice_length * sub_margin):slice_length-int(slice_length * sub_margin)][:]
    
    return slices


for i, frame in enumerate(frames):
    for sub_image in frame.sub_images:
        sub_image.slices = [Slice(data) for data in extract_slices(sub_image.data, 1/7, 0)]
```

## Detect dashed line from slice
### 1. Projection of the 2D slice to 1D array
- Create a 1-dimensional array with the same length as the slice
- Initialize the array with zeroes
- Detect contours of line segments
- Get leftmost and rightmost pixel of every countour
- Set all corresponding x values in the array to 255 (interval $[leftmost,rightmost]$)
-> Projection to 1D

```python=
def create_line_data(data):
    line_data = np.zeros(len(data[0]))
    
    contours, hierarchy = cv.findContours(data, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for i, cnt in enumerate(contours):
        leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])[0]
        rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])[0]
        length = rightmost - leftmost
        line_data[leftmost:rightmost] = np.ones(length)
    
    return line_data

for frame in frames:
    for sub_image in frame.sub_images:
        for sli in sub_image.slices:
            sli.line_data = create_line_data(sli.data)

```

### 2. Analyse the 1D-Projection
Set a maximum streak length of ones and zeros in the 1D representation of the slice. A dashed line is detected if every streak length is below the given maximum.

```python=
def check_line_dashed(line_data, max_dash_length_pct):
    last_pixel = 0
    distance = 0
    result = True
    for pixel in line_data:
        distance += 1
        if distance > len(line_data) * max_dash_length_pct:
            result = False
            break
        
        if pixel != last_pixel:
            distance = 0
            last_pixel = pixel
    
    return result


for frame in frames:
    for sub_image in frame.sub_images:
        sub_image.valid = np.all([check_line_dashed(sli.line_data, 1/4) for sli in sub_image.slices])

```


# 5. Image Manipulation
Draw and fill out the detected rectangle. A gaussian blur with transformation of the original image data is way too slow while a simple black rectangle is much faster.


```python=
def blur_rectangle(image, rect):
    #simple black rectangle
    return cv.drawContours(image.copy(), [rect], cv.DRAW_ALL_CONTOURS, 1, cv.FILLED)
    
    #Old code of gaussian blurred rectangle: 
    #rows, cols, layers = image.shape
    
    #dst = []
    #
    #pts1 = order_points([rect[0][0],rect[1][0],rect[2][0],rect[3][0]])
    #pts2 = np.float32([[0, 0], [cols, 0], [cols, rows], [0, rows]])
    #m = cv.getPerspectiveTransform(pts1, pts2)
    #dst = cv.warpPerspective(image, m, (cols, rows))
    
    #dst = cv.GaussianBlur(dst,(251,251),251)
    #dst = np.zeros(image.shape, np.uint8)

    #return cv.warpPerspective(dst, m, (cols, rows), image, cv.WARP_INVERSE_MAP, cv.BORDER_TRANSPARENT)

for i, frame in enumerate(frames):
    for sub_image in frame.sub_images:
        if sub_image.valid:
            frame.original_data = blur_rectangle(frame.original_data, sub_image.rectangle[0])

```
