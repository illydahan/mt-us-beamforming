import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from das_torch import DAS_FT
from FocusedTxData import FocusedTxData
from PixelGrid import make_foctx_grid

import h5py
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class FocusedTransmit(FocusedTxData):
    def __init__(self, idata, qdata, ele_pos, txdel, fc, fs, tx_ori, tx_dir,angles=(60, 120)):
        self.idata = torch.from_numpy(idata)
        self.qdata = torch.from_numpy(qdata)
        
        # Todo
        self.tx_ori = tx_ori
        self.tx_dir = tx_dir
        
        self.ele_pos = ele_pos
        self.fc = fc
        self.fs = fs
        self.fdemod = 0
        self.c = 1540
        
        self.time_zero = np.zeros((10,))
        
        super().validate()



f = h5py.File(r'C:\Users\elaydahan\Documents\School\thesis\de-beamformer\us-classification-data\abdominal_US\rf\e14.jpg.h5')

i,q = np.array(f['idata'], dtype=np.float32), np.array(f['qdata'], dtype=np.float32)



txdel = np.array(f['txdel'], dtype=np.float32)[:, 0, :]

zlims = [14e-3, 15e-2]
xlims = [-0.0094, 0.0094]
n_angles = 10

i = i.reshape(n_angles, 64, -1)
q = q.reshape(n_angles, 64, -1)

ele_pos = np.linspace(xlims[0], xlims[-1], 64, dtype=np.float32)
ele_pos = np.stack([ele_pos, np.zeros_like(ele_pos), np.zeros_like(ele_pos)], axis = 1)
tx_ori = np.zeros((n_angles, 3), dtype=np.float32)

tx_dir = np.ones((n_angles, 2), dtype=np.float32)

tx_dir[:, 0] = 60 * tx_dir[:, 0]
tx_dir[:, 1] = 120 * tx_dir[:, 1]

fc = 2720000
fs = 4*fc


PF = FocusedTransmit(i, q, ele_pos, txdel, fc, fs, tx_ori=tx_ori, tx_dir=tx_dir)


wvln = PF.c / PF.fc
dx = wvln / 3
dz = dx  # Use square pixels

dr = wvln / 4

grid = make_foctx_grid([0, 15e-2], dr, tx_ori, tx_dir)
fnum = 1

# Create a DAS_PW neural network for all angles, for 1 angle
dasN = DAS_FT(PF, grid, device='cpu')

# Store I and Q components as a tuple
iqdata = (PF.idata, PF.qdata)


xlims = rlims[1] * np.array([-0.7, 0.7])
zlims = rlims[1] * np.array([0, 1])
img_grid = make_pixel_grid(xlims, zlims, wvln / 2, wvln / 2)
grid = np.transpose(grid, (1, 0, 2))
g1 = np.stack((grid[:, :, 2], grid[:, :, 0]), -1).reshape(-1, 2)
g2 = np.stack((img_grid[:, :, 2], img_grid[:, :, 0]), -1).reshape(-1, 2)
bsc = griddata(g1, bimg.reshape(-1), g2, "linear", 1e-10)
bimg = np.reshape(bsc, img_grid.shape[:2])
grid = img_grid.transpose(1, 0, 2)
# Make 75-angle image
idasN, qdasN = dasN(iqdata)
idasN, qdasN = idasN.detach().cpu().numpy(), qdasN.detach().cpu().numpy()
iqN = idasN + 1j * qdasN  # Tranpose for display purposes
bimgN = 20 * np.log10(np.abs(iqN))  # Log-compress
bimgN -= np.amax(bimgN)  # Normalize by max value


# Display images via matplotlib
extent = [grid[0, 0, 0], grid[-1, 0, 0], grid[0, 0, 0], grid[0, 0, -1]]
extent = np.array(extent) * 1e3
plt.imshow(bimgN, vmin=-60, cmap="gray", extent=extent, origin="upper")
plt.show()
