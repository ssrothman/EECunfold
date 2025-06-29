import os
import numpy as np
import json
import pickle

def wrapped_write_pickle(path, data, silent=False):
    try:
        if not silent:
            print("Writing to", path)
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    except Exception as e:
        print(f"Error writing to {path}: {e}")
        raise

def wrapped_read_pickle(path, silent=False):
    try:
        if not silent:
            print("Reading from", path)
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error reading from {path}: {e}")
        raise

def wrapped_write_json(path, data, silent=False):
    try:
        if not silent:
            print("Writing to", path)
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print(f"Error writing to {path}: {e}")
        raise

def wrapped_read_json(path, silent=False):
    try:
        if not silent:
            print("Reading from", path)
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading from {path}: {e}")
        raise

def wrapped_write_np(path, data, silent=False):
    try:
        if not silent:
            print("Writing to", path)
        np.save(path, data)
    except Exception as e:
        print(f"Error writing to {path}: {e}")
        raise

def wrapped_read_np(path, silent=False):
    try:
        if not silent:
            print("Reading from", path)
        return np.load(path)
    except Exception as e:
        print(f"Error reading from {path}: {e}")
        raise


