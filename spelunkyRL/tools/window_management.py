import win32gui
import win32process
import win32con
import win32api

def force_foreground_window(hwnd):
    win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
    win32api.keybd_event(win32con.VK_MENU, 0, 0, 0)
    win32gui.SetForegroundWindow(hwnd)
    win32api.keybd_event(win32con.VK_MENU, 0, win32con.KEYEVENTF_KEYUP, 0)

def ensure_window_visible(hwnd: int) -> None:
    if win32gui.GetForegroundWindow() == hwnd and not win32gui.IsIconic(hwnd):
        return  # Fast path – nothing to do.

    if win32gui.IsIconic(hwnd):                       # Was minimised → restore.
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)

    # # First try the “legal” way …
    # win32gui.SetForegroundWindow(hwnd)
    # if win32gui.GetForegroundWindow() == hwnd:
    #     return  # Success – finished.

    # … and only then fall back to the ALT key workaround once
    win32api.keybd_event(win32con.VK_MENU, 0, 0, 0)               # press ALT
    win32gui.SetForegroundWindow(hwnd)
    win32api.keybd_event(win32con.VK_MENU, 0, win32con.KEYEVENTF_KEYUP, 0)  # release ALT


def get_hwnd_for_pid(pid: int) -> int:
        """
        Return the HWND of the first visible, enabled, top‑level window
        that belongs to the given PID. Raises RuntimeError if none found.
        """
        candidates: list[int] = []

        def _enum(hwnd, _):
            if win32gui.IsWindowVisible(hwnd) and win32gui.IsWindowEnabled(hwnd):
                _, win_pid = win32process.GetWindowThreadProcessId(hwnd)
                if win_pid == pid:
                    candidates.append(hwnd)
            return True  # continue enumeration

        win32gui.EnumWindows(_enum, None)

        if not candidates:
            raise RuntimeError(f"No window found for PID {pid}")
        # If the process spawns more than one top‑level window, you can refine
        # the choice here (e.g. by title, size, or z‑order).  Usually the first is fine.
        return candidates[0]
