"""Capture Mahjong Soul's own window through Windows Graphics Capture."""
import ctypes
from ctypes import wintypes
from pathlib import Path
import threading
import time

import numpy as np
from PIL import Image


class CaptureUnavailable(RuntimeError):
    pass


def find_game_window():
    user = ctypes.WinDLL('user32', use_last_error=True)
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    user.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    user.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user.IsWindowVisible.argtypes = [wintypes.HWND]
    windows = []
    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    @callback_type
    def visit(hwnd, _):
        if not user.IsWindowVisible(hwnd):
            return True
        title = ctypes.create_unicode_buffer(512)
        user.GetWindowTextW(hwnd, title, 512)
        pid = wintypes.DWORD()
        user.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        process = kernel.OpenProcess(0x1000, False, pid.value)
        if process:
            try:
                executable = ctypes.create_unicode_buffer(32768)
                length = wintypes.DWORD(len(executable))
                if kernel.QueryFullProcessImageNameW(process, 0, executable, ctypes.byref(length)):
                    if Path(executable.value).name.lower() == 'jantama_mahjongsoul.exe' and title.value:
                        windows.append((int(hwnd), title.value))
            finally:
                kernel.CloseHandle(process)
        return True

    user.EnumWindows.argtypes = [callback_type, wintypes.LPARAM]
    user.EnumWindows(visit, 0)
    if not windows:
        raise CaptureUnavailable('未找到雀魂窗口，请先打开雀魂；助手会自动连接')
    return windows[0]


def trim_letterbox(image, with_bounds=False):
    data = np.asarray(image)
    nonblack = data[:, :, :3].max(axis=2) > 18
    rows = np.flatnonzero(nonblack.mean(axis=1) > .10)
    columns = np.flatnonzero(nonblack.mean(axis=0) > .10)
    if not len(rows) or not len(columns):
        raise CaptureUnavailable('雀魂暂未提供游戏画面，正在等待窗口恢复')
    bounds = (int(columns[0]), int(rows[0]), int(columns[-1]) + 1, int(rows[-1]) + 1)
    trimmed = image.crop(bounds)
    return (trimmed, bounds) if with_bounds else trimmed


class GameWindowCapture:
    def __init__(self):
        self.hwnd = None
        self.title = ''
        self.capture = self.control = None
        self.frame = None
        self.frame_time = None
        self.lock = threading.Lock()
        self.closed = False
        self.geometry = None

    def connect(self, hwnd, title):
        self.stop()
        from windows_capture import WindowsCapture
        self.hwnd, self.title = hwnd, title
        self.closed = False
        capture = WindowsCapture(window_hwnd=hwnd, cursor_capture=False, draw_border=False,
                                 minimum_update_interval=200)

        @capture.event
        def on_frame_arrived(frame, control):
            pixels = np.ascontiguousarray(frame.frame_buffer[:, :, [2, 1, 0]])
            image = Image.fromarray(pixels)
            with self.lock:
                self.frame = image
                self.frame_time = time.monotonic()

        @capture.event
        def on_closed():
            self.closed = True

        self.capture = capture
        self.control = capture.start_free_threaded()

    def snapshot(self):
        hwnd, title = find_game_window()
        user = ctypes.WinDLL('user32', use_last_error=True)
        user.IsIconic.argtypes = [wintypes.HWND]
        if user.IsIconic(hwnd):
            raise CaptureUnavailable('雀魂已最小化；恢复窗口后自动继续，切到其他应用无需暂停')
        if hwnd != self.hwnd or self.closed:
            self.connect(hwnd, title)
        with self.lock:
            image = self.frame.copy() if self.frame is not None else None
        if image is None:
            raise CaptureUnavailable('已连接雀魂窗口，正在接收游戏画面')
        trimmed, bounds = trim_letterbox(image, with_bounds=True)
        self.geometry = (*image.size, bounds)
        return trimmed

    def stop(self):
        if self.control is not None:
            self.control.stop()
        self.control = self.capture = None
        with self.lock:
            self.frame = None
            self.frame_time = None
