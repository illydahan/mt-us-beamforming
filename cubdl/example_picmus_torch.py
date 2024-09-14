# File:       example_PICMUS.py
# Author:     Dongwoon Hyun (dongwoon.hyun@stanford.edu)
# Created on: 2020-03-12
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from das_torch import DAS_PW
import sys
from pathlib import Path

sys.path.append(str(Path(os.getcwd())))
from submissions.goudarzi.das_torch import DAS_PW as DAS_PW1
from PlaneWaveData import PICMUSData
from FocusedTxData import FocusedTxData
from PixelGrid import make_pixel_grid
import h5py
import os

import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))

from datasets.PWDataLoaders import load_data
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


eps = 1e-20
# Load PICMUS dataset
database_path = os.path.join("datasets", "data", "picmus")
#database_path = '../us-data/database'
acq = "experiments"
target = "contrast_speckle"
dtype = "iq"

#database_path = r'C:\Users\elaydahan\Documents\School\thesis\cubdl\datasets\picmus'
path2 = r'C:\Users\elaydahan\Documents\School\thesis\cubdl\datasets\picmus\experiments\contrast_speckle\contrast_speckle_expe_dataset_rf.hdf5'
P = PICMUSData(path2, acq, target, dtype)



P, xlims, zlims = load_data("INS", 2)
# Define pixel grid limits (assume y == 0)

# xlims = [P.ele_pos[0, 0], P.ele_pos[-1, 0]]
# zlims = [5e-3, 55e-3]

#P, xlims, zlims = load_data('JHU', 34)

wvln = P.c / P.fc
dx = wvln / 3
dz = dx  # Use square pixels
grid = make_pixel_grid(xlims, zlims, dx, dz)
fnum = 1

# Create a DAS_PW neural network for all angles, for 1 angle
idx = len(P.angles) // 2 # Choose center angle for 1-angle DAS
das1 = DAS_PW(P, grid, idx)

dasN = DAS_PW(P, grid)

# Store I and Q components as a tuple
iqdata = (P.idata, P.qdata)

max_val = np.maximum(np.abs(P.idata).max(), np.abs(P.qdata).max())
# Make 75-angle image
idasN, qdasN = dasN((P.idata / max_val, P.qdata / max_val))
idasN, qdasN = idasN.detach().cpu().numpy(), qdasN.detach().cpu().numpy()
iqN = idasN + 1j * qdasN  # Tranpose for display purposes
bimgN = 20 * np.log10(np.abs(iqN))  # Log-compress
bimgN_res =bimgN - np.amax(bimgN)  # Normalize by max value

# Make 1-angle image
max_val = np.maximum(np.abs(P.idata[idx]).max(), np.abs(P.qdata[[idx]]).max())




idas1, qdas1, toff = das1(iqdata, return_tof_corr = True)
idas1, qdas1 = idas1.detach().cpu().numpy(), qdas1.detach().cpu().numpy()
iq1 = idas1 + 1j * qdas1  # Transpose for display purposes
bimg1_no_apod = 20 * np.log10(np.abs(iq1) + eps)  # Log-compress
bimg1_no_apod -= np.amax(bimg1_no_apod)  # Normalize by max value

idas1, qdas1 = das1((P.idata / max_val, P.qdata / max_val))
idas1, qdas1 = idas1.detach().cpu().numpy(), qdas1.detach().cpu().numpy()
iq1 = idas1 + 1j * qdas1  # Transpose for display purposes
bimg1 = 20 * np.log10(np.abs(iq1) + eps)  # Log-compress

max_single = np.amax(bimg1)
bimg1 -= np.amax(bimg1)  # Normalize by max value

def log_norm(im):
    thresh = 1.8
    
    im_tensor = torch.from_numpy(im)
    right_side = 20 * torch.log10(torch.nn.Threshold(thresh, 1)(im_tensor))
    left_side = -3 * torch.nn.Threshold(-thresh, 0)(-im_tensor)
    im_mod =  left_side + right_side
    
    return im_mod.numpy()
    
    

print((bimg1_no_apod != bimg1).sum())
# Display images via matplotlib
extent = [xlims[0] * 1e3, xlims[1] * 1e3, zlims[1] * 1e3, zlims[0] * 1e3]
plt.subplot(131)
plt.imshow(bimg1_no_apod, vmin=-60, cmap="gray", extent=extent, origin="upper")
plt.title("no apod")
plt.subplot(132)
plt.imshow(bimg1, vmin=-60, cmap="gray", origin="upper")
plt.title("with apod")
plt.subplot(133)
plt.imshow(log_norm(np.abs(iq1)), vmax = 15, cmap="gray", origin="upper")
plt.title("with apod")
plt.show()
plt.savefig("scratch.png")
