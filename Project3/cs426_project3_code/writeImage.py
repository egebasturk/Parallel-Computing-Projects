from matplotlib import pyplot as plt
import numpy as np
data = np.loadtxt('cmake-build-debug/mat.out', dtype=int)
#data = np.loadtxt('images/1.1.txt', dtype=int)
print(data)
plt.imshow(data, cmap='gray', vmin=0, vmax=255)
plt.show()
