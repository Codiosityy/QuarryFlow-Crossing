import ctypes
from ctypes import wintypes
import time
import os

from quarryflow.dashboard_data import run_policy_suite

# Setup GetProcessMemoryInfo for Windows
class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]

def get_memory_mb():
    # Returns Private Usage in MB
    try:
        process = ctypes.windll.kernel32.GetCurrentProcess()
        counters = PROCESS_MEMORY_COUNTERS_EX()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS_EX)
        if ctypes.windll.psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb):
            return counters.PrivateUsage / (1024 * 1024)
    except:
        pass
    return 0.0

def main():
    print(f"Initial Memory: {get_memory_mb():.2f} MB")
    start_time = time.time()
    
    peak_mem = get_memory_mb()
    
    def progress_callback(p, msg):
        nonlocal peak_mem
        m = get_memory_mb()
        if m > peak_mem: peak_mem = m
        # only print every 25%
        if int(p*100) % 25 == 0:
            print(f"Progress: {p*100:.0f}% | Mem: {m:.2f} MB | {msg}")
            
    run_policy_suite(
        'chaotic_long_gate', 
        seed=11, 
        model_path='artifacts/models/surrogate.pkl', 
        record_history=True, 
        fast_mode=False,
        progress_callback=progress_callback
    )
    
    print(f"Total Time: {time.time() - start_time:.2f} seconds")
    print(f"Final Memory: {get_memory_mb():.2f} MB")
    print(f"Peak Memory Detected: {peak_mem:.2f} MB")

if __name__ == "__main__":
    main()
