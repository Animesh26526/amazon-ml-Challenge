import os
import sys

def check():
    import ctypes
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]
    stat = MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
    print(f"MemoryLoad: {stat.dwMemoryLoad}%")
    print(f"AvailPhys: {stat.ullAvailPhys / (1024**3):.2f} GB / TotalPhys: {stat.ullTotalPhys / (1024**3):.2f} GB")
    print(f"AvailPage: {stat.ullAvailPageFile / (1024**3):.2f} GB / TotalPage: {stat.ullTotalPageFile / (1024**3):.2f} GB")

if __name__ == "__main__":
    check()
