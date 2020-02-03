Abgeschlossen
- Code Refactoring
- Closing im Preprocessing
- Contour Detection neugeschrieben, basierend auf Hierarchy Tree Depth
- Implementierung einer Filtermaske für die Sub-Images
- Blurring durch Schwärzen ersetzt aufgrund von besserer Performance

Sind dabei
- Blurring statt Schwärzung verwenden
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

Handwritten notes on whiteboards or blackboards can be sensitive or contain confidential parts. To protect handwritten notes like this, the goal of our project was to develop a program that finds a specified pattern in images and blurs its content to be unrecognizable. Our pattern, called "secret pattern", is a rectangle containing a slightly smaller dashed rectangle with the confidential texts inside.

![secret pattern](https://raw.githubusercontent.com/uol-mediaprocessing/group-projects-secret-whiteboard/master/img/doc_images/sample0.jpg)

# Use cases

## Lecture recording software

Some universities (like the Carl von Ossietzky university in Oldenburg) use automated systems for lecture recording und uploading as a service for students. Since the lectures may contain sensitive data that should not be public, like access credentials to private webservers, copyrighted information or solutions for exam exercises (so they can be used again in the future), providing a way to hide sensitive information without manually editing the video would be helpful.

## Privacy in real-life video streaming (possible in the future when livestreams are implemented as the input)

When livestreaming videogames, with common screen-capturing softwares like OBS it is possible to select which windows with non-sensitive information are captured. When doing streaming in real-life there are no mechanisms like that and the only option to protect the personal information of the streamer and/or others is image processing.

# 0. Project Setup

## Install dependencies

As the default OpenCV version in Google Colab is outdated, we need to satisfy this dependency manually by upgrading to at least OpenCV 4.1.1.26.

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

# This constant signals that cv.drawContours() should draw all contours
cv.DRAW_ALL_CONTOURS = -1
```

## Data Model

The `Image` class is the base class for all other classes of the data model. Its `data` attribute contains image data as a numpy array.

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

All frames are then stored in a global array called `frames`.

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

In this step, a rectangle detection is performed to gather coordinates of potential matches for the secret pattern. To detect a rectangle, all filtered contours are approximated with reduced vertices. If an approximated contour consists of exactly four vertices and is convex, it is classified as a rectangle and thus included in the `frame.rectangles` attribute for further processing.

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

For each detected rectangle, the presence or absence of a dashed line inside the contours has to be determined. Several steps are required to perform this operation.

## Create transformed sub-images

To be able to do further processing, the first step is to cut out each rectangle and store it as a separate image. To exclude any content from outside the contour, a mask is generated and applied using a [Bitwise And Operation](https://docs.opencv.org/master/d0/d86/tutorial_py_image_arithmetics.html). Additionally, a [Perspective Transformation](https://docs.opencv.org/master/da/d6e/tutorial_py_geometric_transformations.html) is performed to correct any perspective distortion.

```python=
def create_transformed_sub_image(frame_data, rectangle, size):
    rows, cols = frame_data.shape
    rect = rectangle[0]
    pts1 = np.float32(rect)
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

As all parts of the dashed line are parallel to the outline of the rectangle, we can create new images containing only the dashed line segments by slicing the margin areas. Furthermore, by rotating the left and right slice, all sides are aligned horizontally. All four sides can thereby be treated equally in the next steps. The `margin` parameters allows us to specify the width of the extracted slices. The width of these slices is then scaled according to the `sub_margin` parameter.

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

## Detect dashed lines in each slice

To determine if a slice contains a dashed line, we first create a 1-dimensional array with the same length as the width of the slice. Using OpenCV's contour detection, we can then detect all contours of the line segments. A 1-dimensional projection of the dashed is created by setting all corresponding x values of each contour (interval $[leftmost, rightmost]$) in the array to 255.

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

We can then define a method that checks if every stroke length of a line is below the given maximum. If all four slices are contain a dashed line, our secret pattern is considered as detected.

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

The last step is to draw and fill out the detected rectangle. Because a gaussian blur with perspective transformation of the original image data was too slow, we decided to simply blacken it.


```python=
def blur_rectangle(image, rect):
    #simple black rectangle
    return cv.drawContours(image.copy(), [rect], cv.DRAW_ALL_CONTOURS, 1, cv.FILLED)
    

for i, frame in enumerate(frames):
    for sub_image in frame.sub_images:
        if sub_image.valid:
            frame.original_data = blur_rectangle(frame.original_data, sub_image.rectangle[0])

```

# Conclusion

## Average Execution Time

We tried out multiple ideas for the image manipulation and since blurring was extremly slow compared to the other ones and blackening not being very aestetic, we decided to use pixelation at the final step of our program.
With this the processing chain is fast enough to handle 1 image per second, so the usage in video processing like our use cases defined is very likely to be possible in a future version of the project.

| Step                           | Average execution time  |
| ------------------------------ | ----------------------- |
| Preprocessing                  | 468 ms                  |
| Contour Detection              | 6 ms                    |
| Rectangle Detection            | 4 ms                    |
| Create sub-images              | 63 ms                   |
| Extract slices                 | 1 ms                    |
| Detect dashed lines            | 1 ms                    |
| Image Manipulation (blackened) | 4 ms                    |
| Image Manipulation (blurring)  | 3000 ms                 |
| Image Manipulation (pixelation)| 37 ms                   |


## Working Examples

Our program works with a variety of images, is translational symmetric and works with a lot of imperfect drawn squares as well.

![](https://raw.githubusercontent.com/uol-mediaprocessing/group-projects-secret-whiteboard/master/img/samples/images%20that%20work/downscaled/image013.jpg) ![](https://raw.githubusercontent.com/uol-mediaprocessing/group-projects-secret-whiteboard/master/img/samples/images%20that%20work%20blurred/downscaled/image003.jpg)

![](https://raw.githubusercontent.com/uol-mediaprocessing/group-projects-secret-whiteboard/master/img/samples/images%20that%20work/downscaled/image002.jpg) ![](https://raw.githubusercontent.com/uol-mediaprocessing/group-projects-secret-whiteboard/master/img/samples/images%20that%20work%20blurred/downscaled/image005.jpg)

![](https://raw.githubusercontent.com/uol-mediaprocessing/group-projects-secret-whiteboard/master/img/samples/images%20that%20work/downscaled/image001.jpg) ![](https://raw.githubusercontent.com/uol-mediaprocessing/group-projects-secret-whiteboard/master/img/samples/images%20that%20work%20blurred/downscaled/image004.jpg)

![](https://raw.githubusercontent.com/uol-mediaprocessing/group-projects-secret-whiteboard/master/img/samples/images%20that%20work/downscaled/image011.jpg) ![](https://raw.githubusercontent.com/uol-mediaprocessing/group-projects-secret-whiteboard/master/img/samples/images%20that%20work%20blurred/downscaled/image001.jpg)

![](https://raw.githubusercontent.com/uol-mediaprocessing/group-projects-secret-whiteboard/master/img/samples/images%20that%20work/downscaled/image004.jpg) ![](https://raw.githubusercontent.com/uol-mediaprocessing/group-projects-secret-whiteboard/master/img/samples/images%20that%20work%20blurred/downscaled/image007.jpg)

![](https://raw.githubusercontent.com/uol-mediaprocessing/group-projects-secret-whiteboard/master/img/samples/images%20that%20work/downscaled/image006.jpg) ![](https://raw.githubusercontent.com/uol-mediaprocessing/group-projects-secret-whiteboard/master/img/samples/images%20that%20work%20blurred/downscaled/image009.jpg)

![](https://raw.githubusercontent.com/uol-mediaprocessing/group-projects-secret-whiteboard/master/img/samples/images%20that%20work/downscaled/image009.jpg) ![](https://raw.githubusercontent.com/uol-mediaprocessing/group-projects-secret-whiteboard/master/img/samples/images%20that%20work%20blurred/downscaled/image012.jpg)

![](https://raw.githubusercontent.com/uol-mediaprocessing/group-projects-secret-whiteboard/master/img/samples/images%20that%20work/downscaled/image007.jpg) ![](https://raw.githubusercontent.com/uol-mediaprocessing/group-projects-secret-whiteboard/master/img/samples/images%20that%20work%20blurred/downscaled/image010.jpg)

## Failing Examples

There are 4 criterias for our program to fail. The first is too long dashes that are longer than a 4th of the whole quadrilateral (which is our threshold), because then our pattern detection is not triggering. The second is holes in the outlines, that are so big, that they are not closed in our preprocessing. If the rectangle is not closed our rectangle detection is not working at all. The third criteria is too much between the lines, since our code is scanning just a small area around the margins for dashed lines and the detection of those is failing through this. The forth reason for failure is too curvy rectangles, that slip through the detection process.

![](https://raw.githubusercontent.com/uol-mediaprocessing/group-projects-secret-whiteboard/master/img/samples/images%20that%20don't%20work/downscaled/image005.jpg) ![](https://raw.githubusercontent.com/uol-mediaprocessing/group-projects-secret-whiteboard/master/img/samples/images%20that%20don't%20work/downscaled/image002.jpg)

![](https://raw.githubusercontent.com/uol-mediaprocessing/group-projects-secret-whiteboard/master/img/samples/images%20that%20don't%20work/downscaled/image003.jpg) ![](https://raw.githubusercontent.com/uol-mediaprocessing/group-projects-secret-whiteboard/master/img/samples/images%20that%20don't%20work/downscaled/image001.jpg)

![](https://raw.githubusercontent.com/uol-mediaprocessing/group-projects-secret-whiteboard/master/img/samples/images%20that%20don't%20work/downscaled/image004.jpg) ![](https://raw.githubusercontent.com/uol-mediaprocessing/group-projects-secret-whiteboard/master/img/samples/images%20that%20don't%20work/downscaled/image006.jpg)

# Outlook Deep Learning

Although the proposed algorithm is working very well with patterns exactly fitting our metric, it is still lacking the flexibility needed for perfect results with the variability of human drawn images. Therefore, as an outlook for further developement, a deep learning approach may be required. We have identified two major error sources in our processing chain, that lead to the examples in which our algorithm fails.

The first error source is our rectangle detection that is not able to detect rectangles without completely continuous lines.
Because there are many faster algorithmic solutions, that we have not yet explored, this problem may be less suitable for deep learning. For example the Hough Line Transform method may lead to great results, whilst a Deep Learning approach may require a lot of effort due to the fact, that we have to find the four vertices of every rectangle in a consistent manner.

The second and much broader error source of our algorithm, is the metric for the detection of dashed lines. Lines with big gaps or too long dashes are not detected, just as the distance to the outline or object reaching within the slice can distort the detection. 

As a solution to this second error source, we find a Deep Learning approach to be perfectly suitable. As an input image, the already preprocessed and sliced margin areas of the transformed subimages can simplify the task a lot. A convolutional neural network, for example, would lead to an highly improved detection given the existence of enough training data. The mass of training data needed could be aquired partly through generation, since dashed lines are easy to generate in great variations, and in other parts through a larger, more sophisticated crowd survey. Data augmentation is also applicable, although rotation augmentation must be limited to a certain degree. 