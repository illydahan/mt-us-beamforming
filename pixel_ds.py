import sys
import os
import math
import random
import json
from PIL import Image
sys.path.append('../cubdl')

import torch
import torch.nn.functional as func
import torchvision.transforms as vision_transforms
import torchvision.transforms.functional as F
import numpy as np
import matplotlib.pyplot as plt

from cubdl.PixelGrid import make_pixel_grid
from datasets.PWDataLoaders import load_data
from positional_encodings.torch_encodings import PositionalEncodingPermute2D
from cubdl.das_torch import DAS_PW
from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image
from copy import deepcopy

torch.manual_seed(123)
eps = 1e-20

def pre_proc_iq(iq):

    bimg = 20*torch.log10(torch.sqrt(torch.sum(iq ** 2, dim = 0)) + eps)
    bimg = bimg - torch.max(bimg)
    
    bimg = bimg.clamp(-60)
    bimg = (bimg + 60) / 60
    
    return bimg 
    
class IQImage(Dataset):
    def __init__(self, main_data_dir, imgs_dir, cached_data_dir = None, n_t=256, index_mode = False, resize_transform=None) -> None:
        super().__init__()
        self.main_data_dir = main_data_dir
        self.imgs_dir = imgs_dir
        
        self.transforms = vision_transforms.Compose([vision_transforms.Grayscale(),
                                                     vision_transforms.ToTensor()])
        
        self.resize_transform = resize_transform
        self.iq_transform = None
        self.cached_data_dir = cached_data_dir
        self.n_t = n_t

        self._init_valid_files()
        
        if index_mode:
            if os.path.exists(os.path.join(self.main_data_dir, 'idx.json')):
                import json
                with open(os.path.join(self.main_data_dir, 'idx.json'), 'r') as f:
                    self.indexes = json.load(f)
            
            else:
                self._index_data()
        else:
            self.indexes = os.listdir(cached_data_dir)
            
        self.index_mode = index_mode
        
        
    def __len__(self):
        if self.index_mode:
            return len(self.indexes)
        else:
            return len(os.listdir(self.cached_data_dir))

    def _init_valid_files(self):
        all_files = os.listdir(self.main_data_dir)
        self.valid_files = all_files
        
    def _index_data(self):
        indexes = []
        print("Indexing data...")
        for fn in tqdm(os.listdir(self.cached_data_dir), total=len(os.listdir(self.cached_data_dir))):
            data = torch.load(os.path.join(self.cached_data_dir, fn))
            n_len = data.shape[-1]
            
            indexes_per_file = math.ceil(n_len / self.n_t)
            
            for i in range(indexes_per_file):
                indexes.append((i, fn))
            
        self.indexes = indexes
        
    def get_indexes_from_rois(self, rois, n_t):
        def make_divisable(num, n_t = n_t):
            if num % n_t == 0:
                return num
            
            return num + n_t - num % n_t
        
        row_idxs = []
        col_idxs = []
        
        for roi in rois:
            x0, y0, x_end, y_end = roi
            
            x0 = max(x0, n_t)
            y0 = max(y0, n_t)
            x_end = max(x_end, n_t)
            y_end = max(y_end, n_t)
            x0, y0, x_end, y_end = make_divisable(x0), make_divisable(y0), make_divisable(x_end), make_divisable(y_end)
            
            
    
            row_idxs += list(range(y0, y_end, n_t))
            col_idxs += list(range(x0, x_end, n_t))
                    
        return row_idxs, col_idxs
            
            
    
    def __getitem__(self, index, return_fn = False):
        
        if self.index_mode:
            offset, fn = self.indexes[index]
        else:
            fn = self.indexes[index]
        
        im_fn = os.path.splitext(fn)[0][:-3] + '.png'

        cached_file_path = os.path.join(self.cached_data_dir, fn)
        tof_corrected = torch.load(cached_file_path)
        
        im = self.transforms(Image.open(os.path.join(self.imgs_dir, im_fn)))
        
        if self.index_mode:
            
            tof_corrected = tof_corrected[..., offset*self.n_t: (offset + 1) * self.n_t]
            to_pad = 0
            if tof_corrected.shape[1] != self.n_t:
                to_pad = self.n_t - tof_corrected.shape[1]
                tof_corrected = func.pad(tof_corrected, (0, to_pad, 0, 0), "constant", 0).view(2, -1, self.n_t)

            
            
            patch_dim = int(self.n_t ** 0.5)
            
            im_roi = im.view(1, -1)[:, offset*self.n_t: (offset + 1) * self.n_t]

            if to_pad != 0:
                im_roi = func.pad(im_roi, (0, to_pad, 0, 0), "constant", 0)
            
            im = im_roi.view(1, patch_dim, patch_dim)

        else:
            h,w= im.shape[1:]
            i, q= tof_corrected[:128], tof_corrected[128:]
            tof_corrected = torch.stack([i.sum(dim=0), q.sum(dim=0)]).view(-1, h, w)
        
        if self.resize_transform is not None:
            tof_corrected = self.resize_transform(tof_corrected)
            im = self.resize_transform(im)

        return tof_corrected, im
    
    def _calc_mean_std(self):
        print("Calculating data mean and std...")
        psum    = torch.tensor([0.0] * 256)
        psum_sq = torch.tensor([0.0] * 256)
        
        from tqdm import tqdm
        for i in tqdm(range(len(self))):
            iq, im = self.__getitem__(i)
            psum    += iq.sum(axis        = [1, 2])
            psum_sq += (iq ** 2).sum(axis = [1, 2])
        
        im_h, im_w = self.im_shape
        count = len(self) * im_h * im_w

        total_mean = psum / count
        total_var  = (psum_sq / count) - (total_mean ** 2)
        total_std  = torch.sqrt(total_var)
        

        self.iq_transform = vision_transforms.Normalize(mean = total_mean, std=total_std)
        torch.save(total_mean, 'datasets/rf_stats/mean.pt')
        torch.save(total_std, 'datasets/rf_stats/std.pt')

class MultiTaskIQ2ImBase(IQImage):
    def __init__(self, main_data_dir, 
                 imgs_dir, 
                 cached_data_dir,
                 files,
                 resize_transform=None, 
                 task_id=None,
                 augm = False, 
                 n_t = 8,
                 sub_sample_rate = 1) -> None:
        super(MultiTaskIQ2ImBase, self).__init__(main_data_dir, imgs_dir, cached_data_dir, False, resize_transform=resize_transform)
        self.task_id = task_id
        
        self.n_t = n_t        
        self.all_files = os.listdir(cached_data_dir)
        self.augm = augm
        
        if len(files) > 0:
            all_files = list(
                filter(lambda fn: os.path.splitext(fn)[0] in files, self.all_files)
            )
            self.all_files = all_files
        else:
            all_files = self.all_files
            
        self.tof_cache = {}
        self.multi_tof_cache = {}
        self.imgs_cache = {}
        
        self.total_samples = 0
        self.files_index = []

        with open(os.path.join(self.main_data_dir,'norm.json'), 'r') as fp:
            self.norm = json.load(fp)
            
        with open(os.path.join(self.main_data_dir,'mean_std_norm.json'), 'r') as fp:
            self.mean_std_norm = json.load(fp)
            
        with open(os.path.join(self.main_data_dir,'max_val.json'), 'r') as fp:
            self.max_val = json.load(fp)   
            
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, index):
        fn, row_idx, col_idx = self.files_index[index]
        
        gap = self.n_t
        tof_data = self.tof_cache[fn][row_idx - gap : row_idx + gap,
                                      col_idx - gap : col_idx + gap].permute(2, -1, 0, 1).flatten(start_dim = -2)
        
        im_data = self.multi_tof_cache[fn][:,
                                      row_idx - gap : row_idx + gap,
                                      col_idx - gap : col_idx + gap]    
        
        return tof_data, im_data, 0
        
class SubSampleIQ2IM(MultiTaskIQ2ImBase):
    def __init__(self, main_data_dir, imgs_dir, cached_data_dir, files, resize_transform=None, task_id=None, augm=False, n_t=8, sub_sample_rate=1) -> None:
        super().__init__(main_data_dir, imgs_dir, cached_data_dir, files, resize_transform, task_id, augm, n_t, sub_sample_rate)
        
        print("Indexing dataset...")
        for fn in tqdm(self.all_files, total=len(self.all_files)):
            raw_fn = os.path.splitext(fn)[0]
            im = Image.open(os.path.join(self.imgs_dir, raw_fn + '.png')).convert('L')
            w,h = im.size
            im_tensor = F.to_tensor(im)
            
            self.tof_cache[fn] = torch.load(os.path.join(self.cached_data_dir, fn )).cpu()
            n_elem = self.tof_cache[fn].shape[-1]

            self.tof_cache[fn][..., torch.arange(1, n_elem, 2)] = self.tof_cache[fn][..., torch.arange(0, n_elem, 2)] 
            
            multi_tof = torch.load(os.path.join(os.path.join(self.main_data_dir, 'rf_multi'), fn )).cpu().permute(2, 0,1)


            self.multi_tof_cache[fn] = multi_tof
            self.imgs_cache[fn] = im_tensor
        
                                
            for row_idx in range(n_t, h - n_t + 1, n_t):
                for col_idx in range(n_t, w - n_t + 1, n_t * sub_sample_rate):
                    self.files_index.append((fn, row_idx, col_idx))
                    self.total_samples += 1  
        
    def __getitem__(self, index):
        return super().__getitem__(index)
            
class DespackleIQ2IM(IQImage):
    def __init__(self, main_data_dir, 
                 imgs_dir, 
                 cached_data_dir,
                 files,
                 resize_transform=None, 
                 task_id=None,
                 augm = False, 
                 n_t = 8,
                 sub_sample_rate = 1,
                 train=True) -> None:
        super(DespackleIQ2IM, self).__init__(main_data_dir, imgs_dir, cached_data_dir, False, resize_transform=resize_transform)
        self.task_id = task_id
        
        self.n_t = n_t        
        self.all_files = os.listdir(cached_data_dir)
        self.augm = augm
        
        if len(files) > 0:
            all_files = list(
                filter(lambda fn: os.path.splitext(fn)[0] in files, self.all_files)
            )
            self.all_files = all_files
        else:
            all_files = self.all_files
            
        self.tof_cache = {}
        self.multi_tof_cache = {}
        self.imgs_cache = {}
        
        self.total_samples = 0
        self.files_index = []
        print("Indexing dataset...")
        
        self.original_multi_tof = {}
        
        if train:
            roi = {
                os.path.splitext(fn): [] for fn in self.all_files
            }

        for fn in tqdm(all_files, total=len(all_files)):
            raw_fn = os.path.splitext(fn)[0]
            im = Image.open(os.path.join(self.imgs_dir, raw_fn + '.png')).convert('L')
            w,h = im.size
            im_tensor = F.to_tensor(im)
            
            self.tof_cache[fn] = torch.load(os.path.join(self.cached_data_dir, fn )).cpu()
            multi_tof = torch.load(os.path.join(os.path.join(self.main_data_dir, 'rf_multi'), fn )).cpu().permute(2, 0,1)
            
            self.original_multi_tof[fn] = multi_tof
            rescale_im = im_tensor * 60 - 60
            
            
            log_env_denoised = rescale_im 
            despackle_env = 10 ** (log_env_denoised / 20)
            multi_tof_i, multi_tof_j = multi_tof[0].cpu(), multi_tof[1].cpu()
            

            
            i_j_ratio = multi_tof_i / multi_tof_j
            despackle_j = torch.sqrt((despackle_env[0] ** 2) / (i_j_ratio ** 2 + 1))
            despackle_i = despackle_j * i_j_ratio
            
            despackle_tof = torch.stack([despackle_i, despackle_j])

            std_ratio = despackle_tof.std() / multi_tof.std()
            despackle_tof = ((despackle_tof - 0) / std_ratio)
            self.multi_tof_cache[fn] = despackle_tof
            self.imgs_cache[fn] = im_tensor
            
            
            if not train:
                for row_idx in range(n_t, h - n_t + 1, n_t):
                    for col_idx in range(n_t, w - n_t + 1, n_t * sub_sample_rate):
                        self.files_index.append((fn, row_idx, col_idx))
                        self.total_samples += 1
            else:
                if raw_fn not in roi:
                    continue
                
                if roi[raw_fn] == []:
                    for row_idx in range(n_t, h - n_t + 1, n_t):
                        for col_idx in range(n_t, w - n_t + 1, n_t * sub_sample_rate):
                            self.files_index.append((fn, row_idx, col_idx))
                            self.total_samples += 1
                else:
                    row_idxs, col_idxs = self.get_indexes_from_rois(roi[raw_fn], n_t)
                    for row_idx in row_idxs:
                        for col_idx in col_idxs:
                            self.files_index.append((fn, row_idx, col_idx))
                            self.total_samples += 1
                
   
        with open(os.path.join(self.main_data_dir,'norm.json'), 'r') as fp:
            self.norm = json.load(fp)
            
        with open(os.path.join(self.main_data_dir,'mean_std_norm.json'), 'r') as fp:
            self.mean_std_norm = json.load(fp)
            
        with open(os.path.join(self.main_data_dir,'max_val.json'), 'r') as fp:
            self.max_val = json.load(fp)        
    
    def __getitem__(self, index):
        fn, row_idx, col_idx = self.files_index[index]
        
        gap = self.n_t
        tof_data = self.tof_cache[fn][row_idx - gap : row_idx + gap,
                                      col_idx - gap : col_idx + gap].permute(2, -1, 0, 1).flatten(start_dim = -2)
        
        
        curr_multi_iq = self.multi_tof_cache[fn]
        im_data = curr_multi_iq[:, row_idx - gap : row_idx + gap, col_idx - gap : col_idx + gap] 
        

        tof_data = tof_data
        
        return tof_data, im_data, 0
        
    def __len__(self):
        return self.total_samples
    
class MTIQ2Image(IQImage):
    def __init__(self, main_data_dir, 
                 imgs_dir, 
                 cached_data_dir,
                 files,
                 resize_transform=None, 
                 task_id=None,
                 augm = False, 
                 n_t = 8, 
                 sub_sample_rate = 1,
                 train=True,
                 step = None,
                 multi_angle = False,
                 n_angle_mode = False) -> None:

        super(MTIQ2Image, self).__init__(main_data_dir, imgs_dir, cached_data_dir, False, resize_transform=resize_transform)
        self.task_id = task_id
        
        self.n_t = n_t        
        self.all_files = os.listdir(cached_data_dir)
        self.augm = augm
        self.n_angle_mode = n_angle_mode
        
        self.pe = PositionalEncodingPermute2D(2)
        if len(files) > 0:
            all_files = list(
                filter(lambda fn: os.path.splitext(fn)[0] in files, self.all_files)
            )
            self.all_files = all_files
        else:
            all_files = self.all_files
            
        if train:
            roi = {
                'carotid_cross_expe_dataset_iq': [],
                'carotid_long_expe_dataset_iq': [],
                'INS001': [],
                'INS002': [],
                'INS003': [],
                'INS005': [],
                'INS007': [],
                'INS009': [],
                'INS010': [],
                'INS011': [],
                'INS012': [],
                'INS013': [],
                'INS017': [],
                'INS018': [],
                'INS022': [],
                'INS023': [],
                'INS024': [],
                'INS025': [],
                'INS026': [],
                'MYO006': [],
                'OSL002': [],
                'OSL003': [],
                'OSL004': [],
                'OSL005': [],
                'OSL006': [],
                'resolution_distorsion_expe_dataset_iq': [],
                'resolution_distorsion_simu_dataset_iq': [],
                'UFL003': [],
            }
        else:
            roi = {}
        
        gap = self.n_t
        
        if gap == 0:
            gap_0 = 1
            gap_1 = 0
            step = 1
        else:
            gap_0 = gap_1 = gap
        
        if step is None:
            step = self.n_t
        
        self.multi_angle = multi_angle
        self.gap_0 = gap_0
        self.gap_1 = gap_1
                
        self.tof_cache = {}
        self.multi_tof_cache = {}
        self.imgs_cache = {}
        
        self.total_samples = 0
        self.files_index = []
    
        
        print("Indexing dataset...")
        
        self.norm_tof = {}
        self.norm_im = {}
        for fn in tqdm(all_files, total=len(all_files)):
            raw_fn = os.path.splitext(fn)[0]
            if raw_fn + '.png' not in os.listdir(self.imgs_dir):
                continue
            
            im = Image.open(os.path.join(self.imgs_dir, raw_fn + '.png')).convert('L')
            w,h = im.size
            im_tensor = F.to_tensor(im)
             
            self.tof_cache[fn] = torch.load(os.path.join(self.main_data_dir, 'rf_split', fn)).cpu()
            pe_tof_cache = self.tof_cache[fn].permute(2, -1, 0, 1).flatten(start_dim = -2).unsqueeze(0)
            
            self.tof_cache[fn] = pe_tof_cache.squeeze().reshape(2, -1, h, w).permute(-2, -1, 0, 1)
            
            if not multi_angle and not n_angle_mode:
                multi_tof = torch.load(os.path.join(os.path.join(self.main_data_dir, 'rf_multi'), fn )).cpu().permute(2, 0,1)
            else:
                multi_tof = torch.load(os.path.join(os.path.join(self.main_data_dir, 'rf_angle_dim'), fn )).cpu()


                
            self.multi_tof_cache[fn] = multi_tof
            self.imgs_cache[fn] = im_tensor
            

            self.norm_tof[fn] = (
                                    self.tof_cache[fn].mean(dim=(0,1,3)).view(2, 1, 1, 1),
                                    self.tof_cache[fn].std(dim=(0,1,3)).view(2, 1, 1, 1) 
                                )
            
            if not multi_angle and not n_angle_mode:
                self.norm_im[fn] = (
                                        self.multi_tof_cache[fn].mean(dim=(1,2)).view(2,1,1),
                                        self.multi_tof_cache[fn].std(dim=(1,2)).view(2,1,1)
                                    )
            else:
                self.norm_im[fn] = (
                                        self.multi_tof_cache[fn].mean(dim=(1, 2,3)).view(2, 1, 1, 1),
                                        self.multi_tof_cache[fn].std(dim=(1, 2,3)).view(2, 1, 1, 1)
                                    )
            
            if self.n_angle_mode:
                self.tof_cache[fn] = deepcopy(multi_tof)
                self.tof_cache[fn] = (self.tof_cache[fn] - self.norm_im[fn][0]) / (self.norm_im[fn][1] + eps)

            
            
            if not train:
                for row_idx in range(n_t, h - n_t + 1, step):
                    for col_idx in range(n_t, w - n_t + 1, step * sub_sample_rate):
                        self.files_index.append((fn, row_idx, col_idx))
                        self.total_samples += 1
            else:
                if raw_fn not in roi:
                    continue
                
                if roi[raw_fn] == []:
                    for row_idx in range(n_t, h - n_t + 1, step):
                        for col_idx in range(n_t, w - n_t + 1, step * sub_sample_rate):
                            self.files_index.append((fn, row_idx, col_idx))
                            self.total_samples += 1
                else:
                    row_idxs, col_idxs = self.get_indexes_from_rois(roi[raw_fn], n_t)
                    for row_idx in row_idxs:
                        for col_idx in col_idxs:
                            self.files_index.append((fn, row_idx, col_idx))
                            self.total_samples += 1
            
            
    
    def __getitem__(self, index):
        fn, row_idx, col_idx = self.files_index[index]
        
        
        norm_tof = self.norm_tof[fn]
        norm_im =  self.norm_im[fn]

        
        angles = torch.tensor([0, 37, 74])
        tof_data = self.tof_cache[fn][:, angles, row_idx - self.gap_0 : row_idx + self.gap_1, 
                                            col_idx - self.gap_0 : col_idx + self.gap_1].squeeze()
        
            
        if not self.n_angle_mode:
            tof_data = self.tof_cache[fn][row_idx - self.gap_0 : row_idx + self.gap_1, 
                                            col_idx - self.gap_0 : col_idx + self.gap_1].squeeze()
            
            tof_data = tof_data.permute(-2, -1, 0, 1)
            
            im_data = self.multi_tof_cache[fn][:,
                                      row_idx -  self.gap_0 : row_idx + self.gap_1,
                                      col_idx -  self.gap_0 : col_idx + self.gap_1]
            im_data = (im_data - norm_im[0]) / (norm_im[1] + 1e-6)
            
            tof_data = (tof_data - norm_tof[0]) / (norm_tof[1] + 1e-6)
            return tof_data, im_data
        
            raise Exception("Modify code")

        if self.multi_angle:
            im_data = self.multi_tof_cache[fn][:, :, row_idx - self.gap_0 : row_idx + self.gap_1, 
                                            col_idx - self.gap_0 : col_idx + self.gap_1].squeeze()
            im_data = (im_data - norm_im[0]) / (norm_im[1] + 1e-6)
            
        
            
        else:
            im_data = self.multi_tof_cache[fn][:,
                                      row_idx -  self.gap_0 : row_idx + self.gap_1,
                                      col_idx -  self.gap_0 : col_idx + self.gap_1]
            im_data = (im_data - norm_im[0]) / (norm_im[1] + 1e-6)
        

        return tof_data, im_data
    
    def __len__(self):
        return self.total_samples


class DespackleIQ2IMv2(MTIQ2Image):
    def __init__(self, main_data_dir, imgs_dir, cached_data_dir, files, resize_transform=None, task_id=None, augm=False, n_t=8, sub_sample_rate=1, train=True, step=None, multi_angle=False, n_angle_mode=False) -> None:
        super().__init__(main_data_dir, imgs_dir, cached_data_dir, files, resize_transform, task_id, augm, n_t, sub_sample_rate, train, step, multi_angle, n_angle_mode)
        
        self.original_multi_tof = {}
        self.files_index = []
        self.total_samples = 0
        for fn in self.multi_tof_cache.keys():
            raw_fn = os.path.splitext(fn)[0]
            im = Image.open(os.path.join(self.imgs_dir, raw_fn + '.png')).convert('L')
            w,h = im.size
            im_tensor = F.to_tensor(im)
        
            multi_tof = torch.load(os.path.join(os.path.join(self.main_data_dir, 'rf_multi'), raw_fn + '.pt' )).cpu().permute(2, 0,1)

            self.original_multi_tof[fn] = multi_tof
            rescale_im = im_tensor * 60 - 60

        
            log_env_denoised = rescale_im 
            despackle_env = 10 ** (log_env_denoised / 20)
            multi_tof_i, multi_tof_j = multi_tof[0].cpu(), multi_tof[1].cpu()
        
            i_j_ratio = multi_tof_i / (multi_tof_j + eps)
            despackle_j = torch.sqrt((despackle_env[0] ** 2) / (i_j_ratio ** 2 + 1))
            despackle_i = despackle_j * i_j_ratio
        
            despackle_tof = torch.stack([despackle_i, despackle_j])

            
            m = despackle_tof.mean()
            s = despackle_tof.std()
            
            self.norm_im[raw_fn + '.pt'] = (m, s)
            self.multi_tof_cache[raw_fn + '.pt'] = despackle_tof
            self.imgs_cache[raw_fn + '.pt'] = im_tensor
            
            if train:
                roi = {
                    'INS001': [[172, 238, 220, 280], [28, 340, 110, 480],
                               [110, 4, 211,73], [51, 101, 128, 150]
                               , [160, 310, 290, 380]],
                    'INS003': [[78, 74, 150, 150], [160, 380, 320, 460]],
                    'INS005': [[126, 165, 304, 261], [0,0, 100, 100]],
                    'INS007': [[0, 258, 50, 373], [160, 93, 252, 163],
                               [167, 395, 248, 460]],
                    'INS009': [[60, 166, 279, 373]],
                    'INS010': [],
                    'INS011': [],
                    'INS012': [],
                    'INS013': [[0, 160, 220, 280]],
                    'INS017': [],
                    'INS018': [[60, 194, 322, 370]],
                    'carotid_cross_expe_dataset_iq': [],
                    'carotid_long_expe_dataset_iq': [],
                    'MYO006': []

                }
            
            if step is None: step = n_t // 2
            if not train:
                for row_idx in range(n_t, h - n_t + 1, step):
                    for col_idx in range(n_t, w - n_t + 1, step * sub_sample_rate):
                        self.files_index.append((fn, row_idx, col_idx))
                        self.total_samples += 1
            else:
                if raw_fn not in roi:
                    continue
                
                if roi[raw_fn] == []:
                    for row_idx in range(n_t, h - n_t + 1, step):
                        for col_idx in range(n_t, w - n_t + 1, step * sub_sample_rate):
                            self.files_index.append((fn, row_idx, col_idx))
                            self.total_samples += 1
                else:
                    row_idxs, col_idxs = self.get_indexes_from_rois(roi[raw_fn], n_t)
                    for row_idx in row_idxs:
                        for col_idx in col_idxs:
                            self.files_index.append((fn, row_idx, col_idx))
                            self.total_samples += 1
    
    def __getitem__(self, index):
        fn, row_idx, col_idx = self.files_index[index]
        
        
        norm_tof = self.norm_tof[fn]
        norm_im =  self.norm_im[fn]
        
        tof_data = self.tof_cache[fn][row_idx - self.gap_0 : row_idx + self.gap_1, 
                                        col_idx - self.gap_0 : col_idx + self.gap_1].squeeze()
        
        tof_data = tof_data.permute(-2, -1, 0, 1)
        
        im_data = self.multi_tof_cache[fn][:,
                                  row_idx -  self.gap_0 : row_idx + self.gap_1,
                                  col_idx -  self.gap_0 : col_idx + self.gap_1]

        im_data = (im_data - 0) / (norm_im[1].item() + 1e-6)
        
        tof_data = (tof_data - norm_tof[0]) / (norm_tof[1] + 1e-6)
        return tof_data, im_data    

def calculate_ds_stats(ds):
    running_sum = 0.0
    running_sq_sum = 0.0
    total_pixels = 0.0
    for idx in tqdm(range(len(ds)), total=len(ds)):
        rf, im, task_id = ds.__getitem__(idx)
        rf = pre_proc_iq(rf)
        total_pixels = (rf.shape[0] * rf.shape[1])
        running_sum += rf.mean()

    mean = running_sum / len(ds)
    
    for idx in tqdm(range(len(ds)), total=len(ds)):
        rf, im, task_id = ds.__getitem__(idx)
        rf = pre_proc_iq(rf)
        running_sq_sum += (rf.mean() - mean) ** 2
    
    std = running_sq_sum / len(ds)
    
    return mean, std
        
        
       

