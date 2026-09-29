"""Free physical memory, from the standard library."""

import ctypes
import os
import sys


def available() -> int:
    """Bytes of physical memory available now; half the total where the platform does not say."""
    if sys.platform == "win32":

        class Status(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("load", ctypes.c_ulong),
                ("total", ctypes.c_ulonglong),
                ("available", ctypes.c_ulonglong),
                ("total_page", ctypes.c_ulonglong),
                ("available_page", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("available_virtual", ctypes.c_ulonglong),
                ("extended", ctypes.c_ulonglong),
            ]

        status = Status(length=ctypes.sizeof(Status))
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        return int(status.available)
    if sys.platform == "darwin":
        total = int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
        return total // 2
    page = os.sysconf("SC_PAGE_SIZE")
    try:
        return page * os.sysconf("SC_AVPHYS_PAGES")
    except (ValueError, OSError):
        return page * os.sysconf("SC_PHYS_PAGES") // 2
