import win32gui
import win32com.client
import win32gui
import win32api
import win32con
import time

GAME_NAME = "Counter-Strike"

class Config:
    handle : int
    win_pos : tuple

def query_handle_info() -> dict:
    hwnd_title = dict()
    def _get_all_hwnd(hwnd, mouse):
        if win32gui.IsWindow(hwnd) and win32gui.IsWindowEnabled(hwnd) and win32gui.IsWindowVisible(hwnd):
            hwnd_title.update({hwnd: win32gui.GetWindowText(hwnd)})

    win32gui.EnumWindows(_get_all_hwnd, 0)
    return hwnd_title

def setForeground(hwnd):
    """
        将窗口设置为最前面
    :param hwnd: 窗口句柄 一个整数
    """
    if hwnd != win32gui.GetForegroundWindow():
        shell = win32com.client.Dispatch("WScript.Shell")
        shell.SendKeys('%')
        win32gui.SetForegroundWindow(hwnd)


from PyQt5.QtWidgets import QApplication
import sys
import numpy as np
from PyQt5.QtGui import *

app = QApplication(sys.argv)

def qtpixmap_to_cvimg(qtpixmap):
    qimg = qtpixmap.toImage()
    temp_shape = (qimg.height(), qimg.bytesPerLine() * 8 // qimg.depth())
    temp_shape += (4,)
    ptr = qimg.bits()
    ptr.setsize(qimg.byteCount())
    result = np.array(ptr, dtype=np.uint8).reshape(temp_shape)
    result = result[..., :3]
    return result

def qt_shot():
    screen = QApplication.primaryScreen()
    pix = screen.grabWindow(QApplication.desktop().winId())
    return pix

def shot_display() -> np.ndarray:
    pix = qt_shot()
    img = qtpixmap_to_cvimg(pix)
    if Config.win_pos:
        x1, y1, x2, y2 = Config.win_pos
        img = img[y1 : y2, x1 : x2]
    
    return np.array(img, dtype=np.uint8)


def getGameWindow():
    # FindWindow(lpClassName=None, lpWindowName=None) 
    window = win32gui.FindWindow(None, GAME_NAME)

    while not window:
        print('Fail to acquire windows, try after 10 secs', flush=True)
        time.sleep(10)
        window = win32gui.FindWindow(None, GAME_NAME)

    win32gui.SetForegroundWindow(window)
    pos = win32gui.GetWindowRect(window)
    print("Game windows at " + str(pos))
    return pos


def init_shot():
    handle = None
    info = query_handle_info()
    for k in info:
        if info[k] == GAME_NAME:
            handle = k
            break
    if handle is None:
        print("Game is not launched")
        exit(0)

    pos = getGameWindow()
    setForeground(handle)

    Config.handle = handle
    Config.win_pos = pos
    return pos

def move_here(pts):
    win32api.SetCursorPos(pts) 


if __name__ == '__main__':
    init_shot()

    for _ in range(10):
        s = time.time()
        shot_display()
        print(time.time() - s)