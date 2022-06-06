import cv2 as cv

def resize(img, height = -1, width = -1):
    o_h, o_w = img.shape[0], img.shape[1]
    if height > 0:
        new_height = int(height)
        new_width = int(new_height / o_h * o_w)
        return cv.resize(img, (new_width, new_height))
    elif width > 0:
        new_width = int(width)
        new_height = int(new_width / o_w * o_h)
        return cv.resize(img, (new_width, new_height))
    else:
        return img

def showImage(img, winname='Default', height = -1, width = -1):
    cv.namedWindow(winname, cv.WINDOW_AUTOSIZE)
    cv.imshow(winname, resize(img, height, width))
    cv.waitKey(0)
    cv.destroyWindow(winname)