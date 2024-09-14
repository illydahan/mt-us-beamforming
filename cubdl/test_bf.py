import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import h5py


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
f = h5py.File('../de-beamformer/data/sim_16.h5', 'r')
f.keys()

img = np.array(f['bmode'])
plt.imshow(img, cmap = 'gray')
plt.show()