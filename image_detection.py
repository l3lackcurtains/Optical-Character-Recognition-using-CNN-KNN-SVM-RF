import PIL
import numpy as np
from PIL import Image
import cv2

# Creates the sliding window
def sliding_window(path):
    from itertools import islice
    img = cv2.imread(path,0)
    small = cv2.resize(img, (40, 30))
    img_arr = np.asarray(small).flatten()

    def window(seq, n):
        it = iter(seq)
        result = tuple(islice(it, n))
        if len(result) == n:
            yield result
        for elem in it:
            result = result[1:] + (elem,)
            yield result
    slides = []

    # Removes images that only have white pixels.
    for w in window(img_arr, 1200):
        count_white = w.count(255)
        if count_white < 1200:
            value = tuple(wi/255 for wi in w)
            slides.append(np.array(value))

    slide_array = np.array(slides)
    return slide_array

