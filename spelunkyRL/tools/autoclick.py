# import time
# import pyautogui
# import win32gui
# import win32process
# import win32con
# from ctypes import windll

# def autoclick(pid):
#     def callback(hwnd, hwnds):
#         _, found_pid = win32process.GetWindowThreadProcessId(hwnd)
#         if found_pid == pid and win32gui.IsWindowVisible(hwnd):
#             hwnds.append(hwnd)
#         return True

#     hwnds = []
#     win32gui.EnumWindows(callback, hwnds)
#     if not hwnds:
#         return

#     current_pid = win32process.GetCurrentProcessId()
#     try:
#         windll.user32.AllowSetForegroundWindow(current_pid)
#     except Exception as e:
#         print(f"Error allowing foreground window: {e}")
#         pass

#     try:
#         win32gui.ShowWindow(hwnds[0], win32con.SW_RESTORE)
#         time.sleep(0.1)  # Give the window time to update its state
#         pyautogui.press('alt')
#         win32gui.SetForegroundWindow(hwnds[0])
#     except Exception as e:
#         print(f"Error setting foreground window: {e}")
#         return

#     pyautogui.hotkey('ctrl', 'f4')

#     left, top, right, bottom = win32gui.GetWindowRect(hwnds[0])
#     button1_x = left + 600
#     button1_y = top + 40
#     pyautogui.moveTo(button1_x, button1_y)
#     pyautogui.click()

#     button2_x = left + 600
#     button2_y = top + 100
#     pyautogui.moveTo(button2_x, button2_y)
#     pyautogui.click()




# import time
# import win32gui
# import win32process
# import win32con
# import win32api

# def autoclick(pid):
#     """
#     Sends Ctrl+F4 and two mouse clicks to the first visible window 
#     owned by the specified PID, without forcing the window to the foreground.
#     """

#     # 1. Find the window belonging to the target PID
#     def callback(hwnd, hwnds):
#         _, found_pid = win32process.GetWindowThreadProcessId(hwnd)
#         if found_pid == pid and win32gui.IsWindowVisible(hwnd):
#             hwnds.append(hwnd)
#         return True

#     hwnds = []
#     win32gui.EnumWindows(callback, hwnds)
#     if not hwnds:
#         print("No visible window found for PID:", pid)
#         return

#     target_hwnd = hwnds[0]  # Use the first matching window

#     # 2. Send Ctrl+F4 keystroke in the background
#     win32api.PostMessage(target_hwnd, win32con.WM_KEYDOWN,  win32con.VK_CONTROL, 0)
#     win32api.PostMessage(target_hwnd, win32con.WM_KEYDOWN,  win32con.VK_F4,      0)
#     win32api.PostMessage(target_hwnd, win32con.WM_KEYUP,    win32con.VK_F4,      0)
#     win32api.PostMessage(target_hwnd, win32con.WM_KEYUP,    win32con.VK_CONTROL, 0)

#     # 3. Calculate click coordinates relative to the window
#     left, top, right, bottom = win32gui.GetWindowRect(target_hwnd)

#     # First click at (600, 40) relative to window top-left
#     click_x1 = left + 600
#     click_y1 = top + 40
#     lParam1 = (click_y1 << 16) | click_x1
#     # Press mouse down then up
#     win32api.PostMessage(target_hwnd, win32con.WM_LBUTTONDOWN, win32con.MK_LBUTTON, lParam1)
#     time.sleep(0.05)  # small delay between down/up
#     win32api.PostMessage(target_hwnd, win32con.WM_LBUTTONUP,   0,                  lParam1)

#     # Second click at (600, 100) relative to window top-left
#     click_x2 = left + 600
#     click_y2 = top + 100
#     lParam2 = (click_y2 << 16) | click_x2
#     win32api.PostMessage(target_hwnd, win32con.WM_LBUTTONDOWN, win32con.MK_LBUTTON, lParam2)
#     time.sleep(0.05)
#     win32api.PostMessage(target_hwnd, win32con.WM_LBUTTONUP,   0,                  lParam2)















import time
import ctypes
from ctypes import wintypes
import win32gui
import win32process
import win32con
import win32api

# We'll prepare a helper for AttachThreadInput using ctypes
user32 = ctypes.WinDLL('user32', use_last_error=True)
user32.AttachThreadInput.argtypes = (wintypes.DWORD, wintypes.DWORD, wintypes.BOOL)
user32.AttachThreadInput.restype = wintypes.BOOL

def attach_thread_input(id_attach, id_attach_to, f_attach):
    """
    Thin wrapper around the user32.AttachThreadInput call.
    """
    if not user32.AttachThreadInput(id_attach, id_attach_to, f_attach):
        raise ctypes.WinError(ctypes.get_last_error())

def force_foreground_window(target_hwnd):
    """
    Attempts to force the target_hwnd to the foreground using AttachThreadInput via ctypes.
    """

    # Grab our thread ID
    this_thread_id = win32api.GetCurrentThreadId()

    # Get the current foreground window and its thread
    fg_window = win32gui.GetForegroundWindow()
    if not fg_window:
        print("No current foreground window found.")
        return

    fg_thread_id, _ = win32process.GetWindowThreadProcessId(fg_window)

    # Get the thread ID of the target window
    target_thread_id, _ = win32process.GetWindowThreadProcessId(target_hwnd)

    # If it's already foreground, nothing to do
    if fg_window == target_hwnd:
        return

    try:
        # Attach the target thread to the foreground thread's input queue
        attach_thread_input(fg_thread_id, target_thread_id, True)
        # Also attach this current thread if needed
        attach_thread_input(fg_thread_id, this_thread_id, True)

        # Now we can try to bring it to the foreground
        win32gui.SetForegroundWindow(target_hwnd)
        # Make sure it's not minimized/hidden
        win32gui.ShowWindow(target_hwnd, win32con.SW_SHOWNORMAL)

    except WindowsError as e:
        # attach_thread_input might raise a WinError if it fails
        print("AttachThreadInput failed:", e)
    finally:
        # Detach the threads
        try:
            attach_thread_input(fg_thread_id, target_thread_id, False)
            attach_thread_input(fg_thread_id, this_thread_id, False)
        except WindowsError:
            # Even if detach fails, we can't do much about it
            pass


def autoclick(pid):
    """
    Example usage: finds a visible window belonging to PID,
    attempts to force-foreground it, then sends Ctrl+F4 and two clicks.
    """

    def callback(hwnd, hwnds):
        _, found_pid = win32process.GetWindowThreadProcessId(hwnd)
        if found_pid == pid and win32gui.IsWindowVisible(hwnd):
            hwnds.append(hwnd)
        return True

    hwnds = []
    win32gui.EnumWindows(callback, hwnds)
    if not hwnds:
        print("No visible window found for PID:", pid)
        return

    target_hwnd = hwnds[0]

    # Try to force this window to the foreground
    force_foreground_window(target_hwnd)
    time.sleep(0.1)

    # Now send Ctrl+F4 keystroke (via PostMessage or keybd_event)
    win32api.PostMessage(target_hwnd, win32con.WM_KEYDOWN,  win32con.VK_CONTROL, 0)
    win32api.PostMessage(target_hwnd, win32con.WM_KEYDOWN,  win32con.VK_F4,      0)
    win32api.PostMessage(target_hwnd, win32con.WM_KEYUP,    win32con.VK_F4,      0)
    win32api.PostMessage(target_hwnd, win32con.WM_KEYUP,    win32con.VK_CONTROL, 0)

    # Calculate some click positions (screen coords)
    left, top, right, bottom = win32gui.GetWindowRect(target_hwnd)
    click_points_screen = [
        (left + 600, top + 40),
        (left + 600, top + 100)
    ]

    for (screen_x, screen_y) in click_points_screen:
        # Convert screen => client
        client_x, client_y = win32gui.ScreenToClient(target_hwnd, (screen_x, screen_y))
        lParam = (client_y << 16) | client_x

        # Post the messages
        win32api.PostMessage(target_hwnd, win32con.WM_LBUTTONDOWN, win32con.MK_LBUTTON, lParam)
        time.sleep(0.05)
        win32api.PostMessage(target_hwnd, win32con.WM_LBUTTONUP,   0,                  lParam)
        time.sleep(0.1)


if __name__ == "__main__":
    # Replace 1234 with the actual PID
    target_pid = 1234
    autoclick(target_pid)
