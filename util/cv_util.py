import cv2 as cv
import numpy as np

def show_image(img, winname = 'Default', height = None, width = None, format="bgr") -> None:  
    """show image directly"""  
    if format.lower() == "rgb":
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
            
    cv.namedWindow(winname, cv.WINDOW_AUTOSIZE)
    if height or width:
        img = resize(img, height, width)
    cv.imshow(winname, img)
    cv.waitKey(0)
    cv.destroyWindow(winname)

def resize(img : np.ndarray, height=None, width=None) -> np.ndarray:
    """resize image according to height or width"""
    if height is None and width is None:
        raise ValueError("not None at the same time")
    if height is not None and width is not None:
        raise ValueError("not not None at the same time")
    h, w = img.shape[0], img.shape[1]
    if height:
        width = int(w / h * height)
    else:
        height = int(h / w * width)
    target_img = cv.resize(img, dsize=(width, height))
    return target_img