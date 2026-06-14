import time
import tracemalloc
from quarryflow.dashboard_data import run_policy_suite

def main():
    print(f"Starting memory profiler...")
    tracemalloc.start()
    start_time = time.time()
        
    run_policy_suite(
        'chaotic_long_gate', 
        seed=11, 
        model_path='artifacts/models/surrogate.pkl', 
        record_history=True, 
        fast_mode=False
    )
    
    current, peak = tracemalloc.get_traced_memory()
    print(f"Total Time: {time.time() - start_time:.2f} seconds")
    print(f"Final Memory: {current / 10**6:.2f} MB")
    print(f"Peak Memory: {peak / 10**6:.2f} MB")
    tracemalloc.stop()

if __name__ == "__main__":
    main()
