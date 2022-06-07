import numpy as np

def collect_info(array):
    shape = array.shape
    dimensions = array.ndim
    length = len(array)
    size = array.size
    return f'Shape: {shape}; dimensions: {dimensions}; length: {length}; size: {size}'
