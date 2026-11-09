"""Available physical memory, from the standard library."""

import ctypes
import ctypes.util
import os
import sys


def available() -> int:
    """Bytes of physical memory available now without swapping, at most what this process's cgroup memory limit leaves; half the total where the platform does not say."""
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
    total = int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    if sys.platform == "darwin":
        level = _sysctl("kern.memorystatus_level")
        return total * level // 100 if level and 0 < level <= 100 else total // 2
    meminfo = _meminfo_available()
    if meminfo is None:
        try:
            meminfo = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES")
        except (ValueError, OSError):
            meminfo = total // 2
    cgroup = _cgroup_available()
    return meminfo if cgroup is None else min(meminfo, cgroup)


def _cgroup_available(root: str = "/sys/fs/cgroup", own: str | None = None) -> int | None:
    """Bytes left under the tightest memory limit of this process's cgroup or its ancestors (v2, else v1), counting reclaimable file cache as free; None without a limit."""
    if own is None:
        own = _own_cgroup()
    found = []
    path = own.strip("/")
    while True:
        here = os.path.join(root, path) if path else root
        found.append(
            _left(os.path.join(here, "memory.max"), os.path.join(here, "memory.current"), "inactive_file")
        )
        if not path:
            break
        path = os.path.dirname(path)
    v1 = os.path.join(root, "memory")
    found.append(
        _left(
            os.path.join(v1, "memory.limit_in_bytes"),
            os.path.join(v1, "memory.usage_in_bytes"),
            "total_inactive_file",
        )
    )
    limited = [left for left in found if left is not None]
    return min(limited) if limited else None


def _own_cgroup() -> str:
    """This process's cgroup v2 path, or the root."""
    try:
        with open("/proc/self/cgroup", encoding="ascii") as f:
            for line in f:
                if line.startswith("0::"):
                    return line[3:].strip()
    except OSError:
        pass
    return "/"


def _left(limit: str, usage: str, inactive: str) -> int | None:
    """Bytes `limit` leaves beside `usage` and the `inactive` cache of the same cgroup; None without a limit."""
    try:
        with open(limit, encoding="ascii") as f:
            text = f.read().strip()
        if text == "max" or int(text) >= 1 << 60:
            return None
        with open(usage, encoding="ascii") as f:
            used = int(f.read().strip())
    except (OSError, ValueError):
        return None
    cache = 0
    try:
        with open(os.path.join(os.path.dirname(limit), "memory.stat"), encoding="ascii") as f:
            for line in f:
                key, _, value = line.partition(" ")
                if key == inactive:
                    cache = int(value)
                    break
    except (OSError, ValueError):
        pass
    return max(int(text) - used + cache, 0)


def _meminfo_available() -> int | None:
    """MemAvailable of /proc/meminfo: free memory plus the cache the kernel can reclaim."""
    try:
        with open("/proc/meminfo", encoding="ascii") as meminfo:
            for line in meminfo:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def _sysctl(name: str) -> int | None:
    """An integer sysctl of macOS, or None."""
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
        value = ctypes.c_uint32(0)
        size = ctypes.c_size_t(ctypes.sizeof(value))
        if libc.sysctlbyname(name.encode(), ctypes.byref(value), ctypes.byref(size), None, 0) != 0:
            return None
        return int(value.value)
    except (OSError, AttributeError):
        return None
