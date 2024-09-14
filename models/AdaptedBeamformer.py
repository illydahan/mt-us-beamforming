from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import lightning.pytorch as pl
import kornia.augmentation as augm
import matplotlib.pyplot as plt
from kornia.losses import MS_SSIMLoss
import os
import sys
sys.path.append("./")
from cubdl.das_torch import DAS_PW
from cubdl.das_torch_v2 import DAS_PW as DAS_PW1
from cubdl.PixelGrid import make_pixel_grid
from datasets.PWDataLoaders import load_data
import numpy as np
from pytorch_msssim import ms_ssim as ms_ssim_score
from pytorch_msssim import ssim
from tqdm import tqdm
from copy import deepcopy


out_act = nn.LeakyReLU(negative_slope=0.02)
ms_ssim = MS_SSIMLoss().cuda()


eps = 1e-6
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Identity(nn.Module):
    def __init__(self) -> None:
        super(Identity, self).__init__()
    
    def forward(self, x):
        return x
    
class ShuffleChannels:
    def __init__(self, p = 0.5):
        self.p = p
    
    def __call__(self, in_):
        tof, multi_tof = in_
        tof_c = tof.shape[1]
        bsz = tof.shape[0]
        if torch.rand(1) > self.p:
            return (tof, multi_tof)
        
        h = w = int(tof.shape[-1])
        n_elem = int(tof.shape[2])
        rnd_grid = torch.randperm(n_elem)
        
        tof = tof.flatten(start_dim = -2)[:, :, rnd_grid]
        
        
        return (tof.view(bsz, tof_c, -1, h, w), multi_tof.view(bsz, 2, h, w))
          
class ShuffleImage:
    def __init__(self, p = 0.5):
        self.p = p
        
    def __call__(self, in_):
        tof, multi_tof = in_
        tof_c = tof.shape[1]
        bsz = tof.shape[0]
        if torch.rand(1) > self.p:
            return (tof, multi_tof)
        
        h = w = int(tof.shape[-1])
        
        rnd_grid = torch.randperm(h*w)
        n_elem = tof.shape[2]
        tof = tof.flatten(start_dim = -2)[..., rnd_grid]
        multi_tof = multi_tof.flatten(start_dim = -2)[..., rnd_grid]
        
        
        return (tof.view(bsz, tof_c, -1, h, w), multi_tof.view(bsz, 2, h, w))

class RandomNoise:
    def __init__(self, p = 0.3):
        self.p = p
        
    def __call__(self, in_):
        tof, multi_tof = in_
        tof_c = tof.shape[1]
        bsz = tof.shape[0]
        factor = (torch.rand(1) * 0.15).item()
        if torch.rand(1) > self.p:
            return (tof, multi_tof)
        
        h = w = int(tof.shape[-1])
        
        rnd_grid = torch.randperm(h*w)
        n_elem = tof.shape[2]
        tof += (factor * torch.rand_like(tof))
        tof = tof.view(bsz, 2, -1, h, w)
        multi_tof = multi_tof.view(bsz, 2, 75, h, w)
        
        multi_tof[:,:, 37] = tof.sum(dim = 2)
        
        
        
        return (tof, multi_tof)

class VerticalFlip():
    def __init__(self, p):
        self.p = p
        
    def __call__(self, in_) -> Any:
        tof, multi_tof = in_
        bsz = tof.shape[0]
        tof_c = tof.shape[1]
        if torch.rand(1) > self.p:
            return (tof, multi_tof)
        
        h = w = int(tof.shape[-1])
        
        tof = tof.flip(-2)
        multi_tof = multi_tof.flip(-2)
        return (tof.view(bsz, tof_c, -1, h, w), multi_tof.view(bsz, 2, h, w))
        
class HorizontalFlip():
    def __init__(self, p):
        self.p = p
        
    def __call__(self, in_) -> Any:
        tof, multi_tof = in_
        bsz = tof.shape[0]
        tof_c = tof.shape[1]
        if torch.rand(1) > self.p:
            return (tof, multi_tof)
        
        h = w = int(tof.shape[-1])

        tof = tof.flip(-1)
        multi_tof = multi_tof.flip(-1)
        return (tof.view(bsz, tof_c, -1, h, w), multi_tof.view(bsz, 2, h, w))

augm = torchvision.transforms.Compose([
    ShuffleChannels(),
    VerticalFlip(p = 0.3),
    HorizontalFlip(p = 0.3)

])


def us_ssim_loss(x, y):
    x_mean = x.mean(dim = [1, 2, 3])
    x_std = x.std(dim = [1, 2, 3])
    
    y_mean = y.mean(dim = [1, 2, 3])
    y_std = y.std(dim = [1, 2, 3])
    
    B,n,H,W = x.shape
    
    # Step 1: Flatten the (1, H, W) dimensions
    x_flat = x.view(B, -1)  # Shape: (B, H * W)
    y_flat = y.view(B, -1)  # Shape: (B, H * W)

    # Step 2: Center the tensors by subtracting the mean
    x_centered = x_flat - x_mean.unsqueeze(1)  # Shape: (B, H * W)
    y_centered = y_flat - y_mean.unsqueeze(1)  # Shape: (B, H * W)
    cov_xy = torch.sum(x_centered * y_centered, dim=1) / (H * W - 1)  # Shape: (B,)
    
    ssim_est = ((2*x_mean*y_mean) * (2*cov_xy)) / ((x_mean ** 2 + y_mean ** 2 + 1e-6) * (x_std ** 2 + y_std ** 2 +1e-6))
    return ssim_est.mean()

def calc_loss(y, y_hat, mask = None):
    
    with torch.no_grad():
        gt_env = torch.sqrt(y[:, 0] ** 2 + y[:, 1] ** 2)
        log_y = torch.log10(gt_env + eps).view_as(y_hat)
    
    ssim_loss =  (1 - us_ssim_loss(y_hat.unsqueeze(1), log_y.unsqueeze(1)) **2)
    pixel_loss = torch.nn.functional.mse_loss(y_hat, log_y)
    
    loss = 10 * (pixel_loss) + (ssim_loss)
    return loss



class ConvBlock(nn.Module):
    def __init__(self, in_dim, 
                     out_dim, 
                    output=False, 
                    stride = (1, 1),
                    kernel_size = (3, 1),
                    padding = (1, 0)) -> None:
        super().__init__()
        
        if not output:
            self.block = nn.Sequential(
                nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=kernel_size, bias=False, padding=padding, stride=stride),
                nn.GELU(),
                nn.Conv2d(in_channels=out_dim, out_channels=out_dim, kernel_size=kernel_size, bias=False, padding=padding, stride=stride),
                nn.GELU(),
            )
        else:
            self.block = nn.Sequential(
                nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=(1, 1), bias=False),
                nn.GELU(),
            )
    
    def forward(self, x):
        return self.block(x)

class ConvModel(nn.Module):
    def __init__(self, out_channels = 2) -> None:
        super(ConvModel, self).__init__()
        
        self.block1 = ConvBlock(2, 64, stride=(1,1))
            
        self.block2 = ConvBlock(64, 128, stride=(1,1))
            
        self.block3 = ConvBlock(128, 256, stride=(1,1))
                        
        self.block4 = ConvBlock(256, 512)
        
        self.block5 = ConvBlock(512, 256)
        self.block6 = ConvBlock(256, 32)
        self.block7 = ConvBlock(32, out_channels, output = True)
        
    
    def forward(self, x):
        out1 = self.block1(x)

        out1 = self.block2(out1)
        
        out1 = self.block3(out1)

        out1 = self.block4(out1)
        
        out1 = self.block5(out1)
        out1 = self.block6(out1)
        out1 = self.block7(out1)
        return out1

class ConvModelModified(ConvModel):
    def __init__(self, out_channels = 2, in_channels = 2) -> None:
        super(ConvModelModified, self).__init__()
        
        self.block1 = ConvBlock(in_channels, 64, kernel_size=(3,1), padding = (1,0))
        self.pool1 = nn.MaxPool2d((2,1))

        self.block2 = ConvBlock(64, 128, kernel_size=(3,1), padding = (1,0))
        self.pool2 = nn.MaxPool2d((2,1))

        self.block3 = ConvBlock(128, 256, kernel_size=(3,1), padding = (1,0))
        self.pool3 = nn.MaxPool2d((2,1))

        self.block4 = ConvBlock(256, 128, kernel_size=(3,1), padding = (1,0))

        self.block5 = ConvBlock(128, 64, kernel_size=(3,1), padding = (1,0))
        self.block6 = ConvBlock(64, out_channels, kernel_size=(3,1), padding = (1,0), output = True)

    
    def forward(self, x):
        out1 = self.block1(x)
        out1 = self.pool1(out1)
        
        out1 = self.block2(out1)
        out1 = self.pool2(out1)
        
        out1 = self.block3(out1)
        out1 = self.pool3(out1)
        
        out1 = self.block4(out1)
        out1 = self.block5(out1)
        out1 = self.block6(out1)
        
        return out1

           
class AdapterNetwork(pl.LightningModule):
    def __init__(self,
                 n_channels, 
                 clearml_task,
                 ckpoint_dirname,
                 angle_wise = False):
        super(AdapterNetwork, self).__init__()
    
        self.ckpoint_path = ckpoint_dirname
        self.angle_wise = angle_wise
        self.clear_ml = clearml_task
        self.model = ConvModelModified(in_channels=n_channels)
        
    def _initialize_weights(self):
        import math
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))

                if m.bias is not None:
                    m.bias.data.zero_()
                    
            elif isinstance(m, nn.BatchNorm2d) and m.affine:
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def remove_bias_recursively(self, module):
        for name, sub_module in module.named_children():
            # Recursively remove bias for sub-modules
            self.remove_bias_recursively(sub_module)

            # Check if the current module has bias
            if isinstance(sub_module, (nn.Conv2d, nn.Conv1d, nn.Linear)):
                sub_module.bias = None
    
    def replace_batchnorm_with_identity(self, module):
        for name, child in module.named_children():
            if isinstance(child, nn.BatchNorm2d) or isinstance(child, nn.BatchNorm1d):
                setattr(module, name, nn.Identity())
            else:
                self.replace_batchnorm_with_identity(child)
            
    def forward(self, x, return_iq=False):

        p_z = int(x.shape[-1] ** 0.5)
        out = self.model(x)
        
        if return_iq:
            return out
        if not self.angle_wise:
            out = F.relu(out.mean(2).view(x.shape[0], 2, p_z, p_z))
            env_pred = torch.sqrt(torch.pow(out[:, 0], 2) + torch.pow(out[:, 1], 2) + eps)
            log_y_hat = torch.log10(env_pred + eps)
   
            return log_y_hat, out
        
        else:
             out = F.relu(out.view(x.shape[0], 2, 75, p_z * p_z))
             return out

    def forward_step(self, batch, is_validation = False):

        x, y = batch
        if not is_validation:
            x,y = augm((x,y))
            
        y_hat = self.forward(x.flatten(start_dim = -2))
        
        if not self.angle_wise:
            return x, (y, None), y_hat
        else:
            return x, (y, None), y_hat
                 
    def training_step(self, batch, batch_idx):
        x,y,y_hat = self.forward_step(batch)
        env_hat = y_hat[1]
        y_hat = y_hat[0]
        
        loss = calc_loss(y[0], y_hat, None)


        self.train_dict['loss'] += loss.item() * x.shape[0]
        self.train_dict['items'] += x.shape[0]
        
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x,y,y_hat = self.forward_step(batch, is_validation=True)
        y_hat = y_hat[0]
   
        loss = calc_loss(y[0], y_hat, None)
        
        
        self.val_dict['loss'] += loss.item() * x.shape[0]
        self.val_dict['items'] += x.shape[0]
    
        self.log('val_loss', loss)
        return loss
    
    def on_train_epoch_end(self) -> None:
        mean_loss = self.train_dict['loss'] / self.train_dict['items']
        self.clear_ml.get_logger().report_scalar(
            title='Train loss', 
            value = mean_loss,
             series = 'loss',
            iteration = self.current_epoch
        )
        return super().on_train_epoch_end()
    
    def on_train_epoch_start(self):
        self.train_dict = {
            'loss': 0.0,
            'items': 0,
        }
        return super().on_train_epoch_start()
    
    def _generate_validation_images(self):
        data_file_names = ["INS004.hdf5", "INS021.hdf5", "MYO003.hdf5"]
        
        avg_ssim = 0
        avg_l1  = 0
        das1_ssim = 0
        das1_items = 0
        
        gt_imgs = []
        das_1_imgs = []
        gen_imgs = []
        loss_items = 0
        
        for data_fn in data_file_names:
            data_source = data_fn[:3]
            acq = int(data_fn[3:6])
            P, xlims, zlims = load_data(data_source, acq)
            
            wvln = P.c / P.fc
            dx = wvln / 3
            dz = dx

            angle_idx = int(np.where(P.angles == 0)[0][0])
            grid = make_pixel_grid(xlims, zlims, dx, dz)

            iqdata = (P.idata, P.qdata)
            dasN = DAS_PW(P, grid)
            idasN, qdasN, toff_n_angles = dasN((P.idata, P.qdata), angle_wise = True)
            n_angles = toff_n_angles.shape[1]
            h,w = idasN.shape[:2]
            toff_n_angles = toff_n_angles.reshape((2, n_angles, h, w))
            das1 = DAS_PW1(P, grid, angle_idx)
            maxval = np.maximum(np.abs(P.idata).max(), np.abs(P.qdata).max())
            P.idata /= maxval
            P.qdata /= maxval
            idas1, qdas1, toff_corr, _ = das1((P.idata, P.qdata))
            idas1, qdas1 = idas1.detach().cpu().numpy(), qdas1.detach().cpu().numpy()

            das1_tof_original = deepcopy(toff_corr).view(h, w, 2, -1)
            # generate das image
            idasN, qdasN = idasN.detach().cpu().numpy(), qdasN.detach().cpu().numpy()
            iq = idasN + 1j * qdasN
            bimg = 20 * np.log10(np.abs(iq) + eps)
            bimg -= np.amax(bimg)

            h,w = grid.shape[:2]


            
            bimg_tensor = torch.from_numpy(bimg)
            bimg_tensor = torch.clamp(bimg_tensor, -60)
            bimg_tensor = (bimg_tensor + 60) /60
            gt_imgs.append(bimg_tensor)
            

            angles = torch.tensor([0, n_angles // 2, n_angles - 1])
            toff_corr = toff_n_angles[:, angles]
            toff_corr = toff_corr.permute(-2, -1, 0, 1)
            
            bimg1 = toff_corr.clone().sum(dim=-1)
            bimg1 = 20 * torch.log10(torch.abs(bimg1[..., 0] + 1j * bimg1[..., 1]) + eps)
            bimg1 = bimg1 - bimg1.max()
            bimg1 = bimg1.clip(-60)
            bimg1 = (bimg1 + 60) / 60
            das_1_imgs.append(bimg1)
            
            del das1, dasN
            if w > 160 and h> 160:
                das1_loss = ms_ssim_score(bimg1.view(1, 1, h, w), bimg_tensor.detach().view(1, 1, h, w), data_range=1.0)
    
                das1_ssim += das1_loss
                das1_items += 1
                
            nn_im = self.beamform_iamge_patch(das1_tof_original, p_z=8)
            gen_imgs.append(nn_im)
            
            if w < 160 or h < 160:
                continue
    
            curr_loss = ms_ssim_score(torch.from_numpy(nn_im).view(1, 1, h, w), bimg_tensor.detach().view(1, 1, h, w), data_range=1.0)
            avg_ssim += curr_loss.item()
            avg_l1 += F.l1_loss(torch.from_numpy(nn_im).view(1, 1, h, w),  bimg_tensor.detach().view(1, 1, h, w)).item()
            loss_items += 1
        
        avg_l1 = avg_l1 / loss_items
        avg_ssim = avg_ssim / loss_items
        
        return gen_imgs, das_1_imgs, gt_imgs, avg_l1, avg_ssim
    
    def on_validation_epoch_end(self) -> None:
        mean_loss = self.val_dict['loss'] / self.val_dict['items']
        
        self.lr_schedulers().step(mean_loss)
        
        self.clear_ml.get_logger().report_scalar(
            title='validation loss', 
            value = mean_loss,
            series = 'loss',
            iteration = self.current_epoch
        )
        
        if mean_loss < self.best_val_loss:
            self.best_val_loss = mean_loss
            
        ckpoint_path = self.ckpoint_path
        
        if not os.path.exists(ckpoint_path):
            os.makedirs(ckpoint_path)
            
        torch.save(
            self.model.state_dict(), os.path.join(ckpoint_path, 
                                                         f'{self.current_epoch}.pt')
        )
        
        # generate test image here...
        
        if self.current_epoch == 0:
           return super().on_validation_epoch_end()
        
        gen_imgs, das_1_imgs, gt_imgs, avg_l1, avg_ssim = self._generate_validation_images()
        
        
        self.clear_ml.get_logger().report_scalar(
            title='L1 test set', 
            value = avg_l1,
            series = 'loss',
            iteration = self.current_epoch
        )
        
        self.clear_ml.get_logger().report_scalar(
            title='SSIM test set', 
            value = avg_ssim,
            series = 'loss',
            iteration = self.current_epoch
        )
        
        for fig_idx in range(len(gen_imgs)):
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
            ax1.imshow(das_1_imgs[fig_idx], cmap='gray', vmin = 0, vmax = 1)
            ax2.imshow(gen_imgs[fig_idx], cmap='gray', vmin = 0, vmax = 1)
            ax3.imshow(gt_imgs[fig_idx], cmap='gray', vmin = 0, vmax = 1)
            
            
            self.clear_ml.get_logger().report_matplotlib_figure(
                title=f'Reconstruction Images {fig_idx}', 
                figure=fig,
                series = 'Image plot',
                iteration = self.current_epoch
            )
            
        
        plt.close('all')
        
        return super().on_validation_epoch_end()
    
    def on_validation_epoch_start(self):
        self.val_dict = {
            'loss': 0.0,
            'items': 0,
            'in_images': [],
            'out_images': [],
            'gt_images': [],
        }
        return super().on_validation_epoch_start()
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer, "min", factor = 0.5, min_lr = 1e-5, patience=3),
                "monitor": "val_loss",
                "frequency": 1
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            },
        }
    
    def _initialize_weights(self):
        import math
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d) and m.affine:
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
    
    
    def _get_batch_index(self, tof_corrected, patch_dim = 8):
        h,w = tof_corrected.shape[:2]
        batch_index = []
        for row_idx in range(patch_dim, h - patch_dim):
            for col_idx in range(patch_dim, w - patch_dim):
                batch_index.append((row_idx, col_idx))
                              
        return batch_index
    
    def _get_batch(self, tof_corrected, batch_idx, indexed_batch, batch_size, patch_dim = 8):

        batch = []
        rows_cols = indexed_batch[batch_idx : batch_idx + batch_size]
        for batch_sample_idx in range(len(rows_cols)):
            r,c = rows_cols[batch_sample_idx]
            batch.append(
                tof_corrected[r - patch_dim: r + patch_dim,
                              c - 1: c ]
            )

        return torch.stack(batch)

    @torch.no_grad()
    def beamform_iamge_patch(self, toff_corr, p_z = 8, eval=False, task_id = None):
        h,w, iq_dim, n_elem = toff_corr.shape
        
        
        tof_clone = toff_corr.clone().cpu()
        tof_mean, tof_std = tof_clone[:, :, :2].mean(dim=(0,1,3)), tof_clone[:, :, :2].std(dim=(0,1,3))
        gap = p_z // 2
        tof_clone = torch.nn.functional.pad(tof_clone, (0, 0, 0, 0, p_z, p_z, p_z, p_z))
        
        tof_clone_copy = tof_clone.clone()
        tof_clone = (tof_clone[:, :, :2] - tof_mean.view(1,1,2,1)) / tof_std.view(1,1,2,1)
        if tof_clone.shape[3] > 2:
            tof_clone = torch.cat([tof_clone, tof_clone_copy[:, :, 2:]], dim=2)
        
        padded_h, padded_w = tof_clone.shape[:2]
        beamforming_im = torch.zeros((padded_h, padded_w), dtype = torch.float32)
            
        indexed_batch = self._get_batch_index(tof_clone, patch_dim=p_z)
        batch_size = 96
        
        for row_idx in tqdm(range(p_z, padded_h - p_z + 1, 2*p_z)):
            for col_idx in range(p_z, padded_w - p_z + 1, 2*p_z):
                patch = tof_clone[row_idx - p_z: row_idx + p_z, 
                                  col_idx - p_z: col_idx + p_z].permute(2, -1, 0, 1).flatten(start_dim = -2).cuda()
                
                pred = self(patch.cuda().unsqueeze(0))[0].squeeze()
                
                if self.angle_wise:
                    pred = pred.sum(dim = 1)
                    pred = 20 * torch.log10(torch.sqrt(pred[0] ** 2 + pred[1] ** 2) + eps).view((16, 16))

                beamforming_im[row_idx - p_z: row_idx + p_z, 
                                  col_idx - p_z: col_idx + p_z] = pred


        beamforming_im = 20 * beamforming_im[p_z:-p_z, p_z: -p_z]

        if eval:
            beamforming_im = beamforming_im - beamforming_im.max()
            beamforming_im = 10 ** (beamforming_im / 20)
            
            return beamforming_im.cpu().detach().numpy()
        
        beamforming_im = beamforming_im.detach().cpu()
        beamforming_im = beamforming_im - beamforming_im.max()
        beamforming_im = beamforming_im.clamp(-60)
        beamforming_im = (beamforming_im + 60) / 60
        return beamforming_im.clamp(0, 1).cpu().detach().view(h,w).numpy()
    
    @torch.no_grad()
    def beamform_image(self, toff_corr, p_z = 16):
        h,w, iq_dim, n_elem = toff_corr.shape
        tof_clone = toff_corr.clone().cpu()
        
        tof_clone = torch.nn.functional.pad(tof_clone, (0, 0, 0, 0, p_z, p_z, p_z, p_z))
        beamforming_im = torch.zeros((h*w, 2), dtype = torch.float32)
        
            
        indexed_batch = self._get_batch_index(tof_clone, patch_dim=p_z)
        batch_size = 96
        
        for batch_idx in tqdm(range(0, len(indexed_batch), batch_size)):
            batch_ = self._get_batch(tof_clone, batch_idx=batch_idx, 
                                     indexed_batch = indexed_batch,
                                     batch_size=batch_size,
                                     patch_dim=p_z)
            

            batch_ = batch_.permute(0, -2, -1, 1, 2).flatten(start_dim = -2).cuda()
            pred = self(batch_.cuda())[0].squeeze()
            beamforming_im[batch_idx: batch_idx + batch_size] = pred


        beamforming_im = beamforming_im.detach().cpu()
        beamforming_im = torch.sqrt(beamforming_im[:, 0] ** 2 + beamforming_im[:, 1] ** 2)
        beamforming_im = 20*torch.log10(beamforming_im + eps)
        beamforming_im = beamforming_im - beamforming_im.max()
        beamforming_im = beamforming_im.clamp(-60)
        beamforming_im = (beamforming_im + 60) / 60
        return beamforming_im.clamp(0, 1).cpu().detach().view(h,w).numpy()


if __name__ == "__main__":
    pass