import torch
import lightning.pytorch as pl
from models.AdaptedBeamformer import AdapterNetwork
from torch.utils.data.dataloader import DataLoader
from pixel_ds import MTIQ2Image, PretrainingUSBF, DespackleIQ2IMv2
import torchvision
from clearml import Task


auto_connect_frameworks={
    'matplotlib': False, 'tensorflow': False, 'tensorboard': False, 'pytorch': False,
    'xgboost': False, 'scikit': False, 'fastai': False, 'lightgbm': False,
    'hydra': False, 'detect_repository': False, 'tfdefines': False, 'joblib': False,
    'megengine': False, 'jsonargparse': False, 'catboost': False
}

model_task_name = 'despackle_base'
clearml_task = Task.init(project_name="mt-us-imageformation", task_name=model_task_name,
                         auto_connect_frameworks=auto_connect_frameworks)

MAX_EPOCHS = 100
BATCH_SIZE = 16

adapter_network = AdapterNetwork(n_channels=2, 
                                 clearml_task = clearml_task,
                                 ckpoint_dirname=model_task_name,
                                 angle_wise=False)

transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.Grayscale(),
     torchvision.transforms.ToTensor(),
])

# train_dataset = MTIQ2Image(r'E:\us-data\train', 
#                         imgs_dir = r'E:\us-data\train\imgs', 
#                         cached_data_dir=r'E:\us-data\train\rf_angle_dim',
#                         #files= ["INS002"],
#                         files = [],
#                         n_t=8, 
#                         sub_sample_rate=1,
#                         multi_angle=False,
#                         n_angle_mode=False)

# test_dataset = MTIQ2Image(r'E:\us-data\val', 
#                      imgs_dir = r'E:\us-data\val\imgs', 
#                      cached_data_dir=r'E:\us-data\val\rf_angle_dim',
#                      #files= ["INS014"],
#                      files = [],
#                      n_t=8,
#                      train=False,
#                      multi_angle=False,
#                      n_angle_mode=False)

train_dataset = DespackleIQ2IMv2(r'E:\us-data\train', 
                        imgs_dir = r'E:\us-data\train_speckle\imgs_new_v2', 
                        cached_data_dir=r'E:\us-data\train\rf_angle_dim',
                        #files= ["INS002"],
                        files = [],
                        n_t=8, 
                        sub_sample_rate=1,
                        multi_angle=False,
                        n_angle_mode=False)

test_dataset = DespackleIQ2IMv2(r'E:\us-data\val', 
                     imgs_dir = r'E:\us-data\val_speckle_images', 
                     cached_data_dir=r'E:\us-data\val\rf_angle_dim',
                     #files= ["INS014"],
                     files = [],
                     n_t=8,
                     train=False,
                     multi_angle=False,
                     n_angle_mode=False)


train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)


trainer = pl.Trainer(max_epochs=MAX_EPOCHS,
                     min_epochs=MAX_EPOCHS - 1,
                     accelerator="gpu", 
                     devices=1,
                     detect_anomaly=False)


trainer.fit(model=adapter_network, 
            train_dataloaders=train_loader, 
            val_dataloaders=val_loader)