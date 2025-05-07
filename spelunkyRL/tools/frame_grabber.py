import threading
import win32gui, win32ui, win32con
import numpy as np
import ctypes
from PIL import Image

def _grab(hwnd):
    left, top, right, bottom = win32gui.GetClientRect(hwnd)
    width, height = right - left, bottom - top

    hwndDC = win32gui.GetWindowDC(hwnd)
    mfcDC  = win32ui.CreateDCFromHandle(hwndDC)
    saveDC = mfcDC.CreateCompatibleDC()
    saveBitMap = win32ui.CreateBitmap()
    saveBitMap.CreateCompatibleBitmap(mfcDC, width, height)
    saveDC.SelectObject(saveBitMap)

    result = ctypes.windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 2)
    bmpinfo = saveBitMap.GetInfo()
    bmpstr = saveBitMap.GetBitmapBits(True)
    img = Image.frombuffer('RGB', (bmpinfo['bmWidth'], bmpinfo['bmHeight']), bmpstr, 'raw', 'BGRX', 0, 1)

    win32gui.DeleteObject(saveBitMap.GetHandle())
    saveDC.DeleteDC()
    mfcDC.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwndDC)

    return np.array(img)

class FrameGrabber(threading.Thread):
    """
    1. Waits until `need_frame` is set.
    2. Grabs exactly ONE frame and stores it in `self.frame`.
    3. Sets `frame_ready` to wake the main Gym thread.
    """
    def __init__(self, hwnd):
        super().__init__(daemon=True)
        self.hwnd = hwnd
        self.frame  = None
        self.start()

    def run(self):
        while True:
            self.frame = _grab(self.hwnd)
