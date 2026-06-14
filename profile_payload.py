import sys
from quarryflow.dashboard_data import run_policy_suite
import pickle

def get_size(obj, seen=None):
    """Recursively finds size of objects in bytes"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size

def main():
    results = run_policy_suite(
        'chaotic_long_gate', 
        seed=11, 
        model_path='artifacts/models/surrogate.pkl', 
        record_history=True, 
        fast_mode=False
    )
    
    # Method 1: Recursive sys.getsizeof
    mem_mb = get_size(results) / (1024 * 1024)
    print(f"Memory Size of Results Object (sys.getsizeof): {mem_mb:.2f} MB")
    
    # Method 2: Pickle size (this is what Streamlit caches!)
    pickled_size = len(pickle.dumps(results)) / (1024 * 1024)
    print(f"Memory Size of Streamlit Cache Payload (Pickle): {pickled_size:.2f} MB")

if __name__ == "__main__":
    main()
