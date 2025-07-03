import numpy as np


arr = np.array([0.54, 0.2, 0.3, 0.41, 0.5], dtype=np.float32)

threshold = 0.4

indices = np.where(arr > threshold)[0]

print(indices)












