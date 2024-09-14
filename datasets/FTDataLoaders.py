import os
import numpy as np
import h5py
from glob import glob
from scipy.signal import hilbert
from cubdl.FocusedTxData import FocusedTxData


class OSLData(FocusedTxData):
    """ Load data from University of Oslo. """

    def __init__(self, database_path, acq):

        # Make sure the selected dataset is valid
        moniker = "OSL{:03d}".format(acq) + "*.hdf5"
        fname = [
            y for x in os.walk(database_path) for y in glob(os.path.join(x[0], moniker))
        ]
        assert fname, "File not found."
        assert acq in [8, 9, 11, 12, 13, 14], "Plane Wave Data. Use PWDataLoaders"

        # Load dataset
        f = h5py.File(fname[0], "r")

        # Get data
        self.idata = np.array(f["channel_data"], dtype="float32")
        self.qdata = np.imag(hilbert(self.idata, axis=-1))
        self.tx_ori = np.array(f["transmit_origin"], dtype="float32").T
        self.tx_dir = np.array(f["transmit_direction"], dtype="float32").T
        self.tx_foc = np.array(f["transmit_focus"], dtype="float32")
        self.fc = np.array(f["modulation_frequency"]).item()
        self.fs = np.array(f["sampling_frequency"]).item()
        self.c = np.array(f["sound_speed"]).item()
        self.time_zero = np.array(f["start_time"], dtype="float32")[0]
        self.fdemod = 0
        self.ele_pos = np.array(f["element_positions"], dtype="float32").T
        self.ele_pos[:, 0] -= np.mean(self.ele_pos[:, 0])

        # Make tx_foc a list of length nxmits
        nxmits = self.idata.shape[0]
        self.tx_foc = self.tx_foc[0, 0] * np.ones((nxmits,), dtype="float32")

        # Data set has wrong information.
        if acq == 8 or acq == 9:
            self.tx_dir *= 0
            self.time_zero = 0 * self.time_zero - np.min(self.time_zero)
        else:
            self.tx_ori *= 0
            self.time_zero = -1 * self.time_zero

        # Validate that all information is properly included
        super().validate()


class JHUData(FocusedTxData):
    """ Load data from Johns Hopkins University. """

    def __init__(self, database_path, acq):

        # Make sure the selected dataset is valid
        moniker = "JHU{:03d}".format(acq) + ".hdf5"
        fname = [
            y for x in os.walk(database_path) for y in glob(os.path.join(x[0], moniker))
        ]
        assert fname, "File not found."
        assert acq in [1, 2, 19, 20, 21, 22, 23], "Plane Wave Data. Use PWDataLoaders"

        # Load dataset
        f = h5py.File(fname[0], "r")

        if acq <= 2:
            # Get data
            self.idata = np.array(f["channel_data"], dtype="float32")
            self.qdata = np.imag(hilbert(self.idata, axis=-1))
            # self.tx_ori = np.array(f["transmit_origin"], dtype="float32").T
            # self.tx_dir = np.array(f["transmit_direction"], dtype="float32").T
            self.tx_foc = np.array(f["transmit_focus"], dtype="float32")
            self.fc = np.array(f["modulation_frequency"]).item()
            self.fs = np.array(f["sampling_frequency"]).item()
            # self.c = np.array(f["sound_speed"]).item()
            # self.time_zero = np.array(f["start_time"], dtype="float32")[0]
            self.fdemod = 0
            # self.ele_pos = np.array(f["element_positions"], dtype="float32").T
            self.ele_pos = np.zeros((128, 3), dtype="float32")
            self.ele_pos[:, 0] = np.arange(128) * f["pitch"]
            self.ele_pos[:, 0] -= np.mean(self.ele_pos[:, 0])
            self.c = 1540.0
            self.tx_ori = np.zeros((256, 3), dtype="float32")
            # self.tx_ori[:, 0] = np.arange(256) * f["pixel_d"]
            self.tx_ori[:, 0] = np.arange(256) * f["pitch"] / 2
            self.tx_ori[:, 0] -= np.mean(self.tx_ori[:, 0])
            self.time_zero = np.zeros((256,), dtype="float32")

            # Make tx_foc a list of length nxmits
            nxmits = self.idata.shape[0]
            self.tx_foc = self.tx_foc[0] * np.ones((nxmits,), dtype="float32")
            self.tx_dir = np.zeros((256, 2), dtype="float32")

            # self.time_zero = 0 * self.time_zero - np.min(self.time_zero)

        else:
            # Get data
            self.idata = np.array(f["channel_data"], dtype="float32")
            self.qdata = np.imag(hilbert(self.idata, axis=-1))
            self.tx_ori = np.array(f["transmit_origin"], dtype="float32").T
            self.tx_ori = self.tx_ori / 2
            self.tx_dir = np.array(f["transmit_direction"], dtype="float32").T
            self.tx_foc = np.array(f["transmit_focus"], dtype="float32")
            self.fc = np.array(f["modulation_frequency"]).item()
            self.fs = np.array(f["sampling_frequency"]).item()
            # self.c = np.array(f["sound_speed"]).item()
            self.time_zero = np.array(f["start_time"], dtype="float32")[0]
            self.fdemod = 0
            self.ele_pos = np.array(f["element_positions"], dtype="float32").T
            self.ele_pos[:, 0] -= np.mean(self.ele_pos[:, 0])
            self.c = 1540.0
            # self.time_zero = np.zeros((256,), dtype="float32")

            # # Make tx_foc a list of length nxmits
            # nxmits = self.idata.shape[0]
            # self.tx_foc = self.tx_foc[0] * np.ones((nxmits,), dtype="float32")
            # self.tx_dir = np.zeros((256, 2), dtype="float32")

            # self.time_zero = 0 * self.time_zero - np.min(self.time_zero)

        # Validate that all information is properly included
        super().validate()


def get_filelist(data_type="all"):
    # Get the filenames of the focused data
    if data_type == "phantom":
        filelist = {"JHU": [19, 20, 21, 22, 23]}
    elif data_type == "invivo":
        filelist = {"OSL": [8, 9, 11, 12, 13, 14], "JHU": [1, 2]}
    elif data_type == "all":
        filelist = {"OSL": [8, 9, 11, 12, 13, 14], "JHU": [1, 2, 19, 20, 21, 22, 23]}
    else:
        filelist = {"OSL": [8, 9, 11, 12, 13, 14], "JHU": [1, 2, 19, 20, 21, 22, 23]}
    return filelist
