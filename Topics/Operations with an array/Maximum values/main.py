import numpy as np


list_a = [int(input()) for _ in range(3)]
array_a = np.array(list_a)
print(array_a.max())
print(array_a.argmax())
