import os
import sys
import shutil
import socket
from datetime import datetime

import math
import numpy as np
import sklearn.metrics as metrics

import torch
from torch.utils.data import DataLoader

import yaml

import warnings
warnings.filterwarnings('ignore')


# -- IMPORT CUSTOM PATHS  ----------------------------------------------------------- #

# IMPORTS PATH TO THE PROJECT
current_model_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
pycharm_projects_path = os.path.dirname(os.path.dirname(current_model_path))
# IMPORTS PATH TO OTHER PYCHARM PROJECTS
sys.path.append(current_model_path)
sys.path.append(pycharm_projects_path)

from model.arvc_PointNet_bs import PointNetDenseCls
from arvc_Utils.Datasets import PLYDataset
from arvc_Utils.Metrics import validation_metrics
from arvc_Utils.Config import Config as cfg

# ----------------------------------------------------------------------------------- #



def train(_device, _data_loader, _model, _loss_fn, _optimizer):
    print('-'*50 + '\n' + 'TRAINING' + '\n' + '-'*50)
    
    loss_lst = []
    current_clouds = 0

    _model.train()
    for batch, (data, label, _) in enumerate(_data_loader):
        data = data.to(_device, dtype=torch.float32)
        label = label.to(_device, dtype=torch.float32)
        
        # Evaluate model
        pred, m3x3, m64x64 = _model(data.transpose(1, 2))

        # Activation function
        m = torch.nn.Sigmoid()
        pred = m(pred)

        # Calulate loss and backpropagate
        avg_train_loss_ = _loss_fn(pred, label)
        _optimizer.zero_grad()
        avg_train_loss_.backward()
        _optimizer.step()

        # Save loss for plotting
        loss_lst.append(avg_train_loss_.item())

        # Print training progress
        current_clouds += data.size(0)
        if batch % 10 == 0 or data.size(0) < _data_loader.batch_size:  # print every (% X) batches
            print(f' - [Batch: {current_clouds}/{len(_data_loader.dataset)}],'
                  f' / Train Loss: {avg_train_loss_:.4f}')

    return loss_lst


def valid(_device, dataloader_, _model, _loss_fn):
    print('-'*50 + '\n' + 'VALIDATION' + '\n' + '-'*50)

    _model.eval()
    f1_lst, pre_lst, rec_lst, loss_lst, conf_m_lst, trshld_lst = [], [], [], [], [], []
    current_clouds = 0

    with torch.no_grad():
        for batch, (data, label, _) in enumerate(dataloader_):
            data, label = data.to(_device, dtype=torch.float32), label.to(_device, dtype=torch.float32)
            pred, m3x3, m64x64 = _model(data.transpose(1, 2))
            m = torch.nn.Sigmoid()
            pred = m(pred)

            avg_loss = _loss_fn(pred, label)
            loss_lst.append(avg_loss.item())

            trshld, pred_fix, avg_f1, avg_pre, avg_rec, conf_m = validation_metrics(label, pred)
            trshld_lst.append(trshld)
            f1_lst.append(avg_f1)
            pre_lst.append(avg_pre)
            rec_lst.append(avg_rec)
            conf_m_lst.append(conf_m)

            current_clouds += data.size(0)

            if batch % 10 == 0 or data.size()[0] < dataloader_.batch_size:  # print every 10 batches
                print(f'[Batch: {current_clouds}/{len(dataloader_.dataset)}]'
                      f'  [Avg Loss: {avg_loss:.4f}]'
                      f'  [Avg F1 score: {avg_f1:.4f}]'
                      f'  [Avg Precision score: {avg_pre:.4f}]'
                      f'  [Avg Recall score: {avg_rec:.4f}]')

    return loss_lst, f1_lst, pre_lst, rec_lst, conf_m_lst, trshld_lst


if __name__ == '__main__':

    # Files = os.listdir(os.path.join(current_model_path, 'config'))
    Files = ['config_xyz_0.yaml']
    for configFile in Files:
        start_time = datetime.now()

        # --------------------------------------------------------------------------------------------#
        # GET CONFIGURATION PARAMETERS
        CONFIG_FILE = configFile
        config_file_abs_path = os.path.join(current_model_path, 'config', CONFIG_FILE)
        config = cfg(config_file_abs_path)


        # --------------------------------------------------------------------------------------------#
        # CHANGE PATH DEPENDING ON MACHINE
        machine_name = socket.gethostname()
        prefix_path = ''
        if machine_name == 'arvc-fran':
            prefix_path = '/media/arvc/data/datasets'
        else:
            prefix_path = '/home/arvc/Fran/data/datasets'

        TRAIN_DATA = os.path.join(prefix_path, config.train.train_dir.__str__())
        VALID_DATA = os.path.join(prefix_path, config.train.train_dir.__str__())


        # --------------------------------------------------------------------------------------------#
        # CREATE A FOLDER TO SAVE TRAINING
        OUT_DIR = os.path.join(current_model_path, config.train.output_dir.__str__())
        folder_name = datetime.today().strftime('%y%m%d%H%M%S')
        OUT_DIR = os.path.join(OUT_DIR, folder_name)
        if not os.path.exists(OUT_DIR):
            os.makedirs(OUT_DIR)

        shutil.copyfile(config_file_abs_path, os.path.join(OUT_DIR, 'config.yaml'))


        # -------------------------------------------------------------------------------------------- #
        # INSTANCE DATASET
        train_dataset = PLYDataset(_mode='train',
                                   _root_dir=TRAIN_DATA,
                                   _coord_idx=config.train.coord_idx,
                                   _feat_idx=config.train.feat_idx, 
                                   _label_idx=config.train.label_idx,
                                   _normalize=config.train.normalize,
                                   _binary=config.train.binary, 
                                   _add_range=config.train.add_range,   
                                   _compute_weights=config.train.compute_weights)

        if config.train.use_valid_data:
            valid_dataset = PLYDataset(_mode='train',
                                       _root_dir=TRAIN_DATA,
                                       _coord_idx=config.train.coord_idx,
                                       _feat_idx=config.train.feat_idx, 
                                       _label_idx=config.train.label_idx,
                                       _normalize=config.train.normalize,
                                       _binary=config.train.binary, 
                                       _add_range=config.train.add_range,   
                                       _compute_weights=config.train.compute_weights)  
        else:
            # SPLIT VALIDATION AND TRAIN
            train_size = math.floor(len(train_dataset) * config.train.train_split)
            val_size = len(train_dataset) - train_size
            train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size],
                                                               generator=torch.Generator().manual_seed(74))

        # INSTANCE DATALOADERS
        train_dataloader = DataLoader(train_dataset, batch_size=config.train.batch_size, num_workers=10,
                                      shuffle=True, pin_memory=True, drop_last=False)
        valid_dataloader = DataLoader(valid_dataset, batch_size=config.train.batch_size, num_workers=10,
                                      shuffle=True, pin_memory=True, drop_last=False)

        # ------------------------------------------------------------------------------------------------------------ #
        # SELECT MODEL
        if torch.cuda.is_available():
            device = torch.device(config.train.device)
        else:
            device = torch.device("cpu")

        model = PointNetDenseCls(k=config.train.output_classes,
                                 n_feat=len(config.train.feat_idx) + 1 if config.train.add_range else 0,
                                 device=device).to(device)
        loss_fn = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.train.init_lr)

        # ------------------------------------------------------------------------------------------------------------ #
        # --- TRAIN LOOP --------------------------------------------------------------------------------------------- #
        print('TRAINING ON: ', device)
        epoch_timeout_count = 0

        best_val = 1 if config.train.termination_criteria == "loss" else 0:

        f1, precision, recall, conf_matrix, train_loss, valid_loss, threshold = [], [], [], [], [], [], []

        for epoch in range(config.train.epochs):
            print(f"EPOCH: {epoch} {'-' * 50}")
            epoch_start_time = datetime.now()

            train_results = train(_device=device, data_loader=train_dataloader,_model_=model, loss_fn=loss_fn,_optimizer=optimizer)

            valid_results = valid(_device=device,_data_loader=valid_dataloader,_model_=model, loss_fn=loss_fn,_optimizer=optimizer)

            # GET RESULTS
            train_loss.append(train_results)

            valid_loss.append(valid_results[0])
            f1.append(valid_results[1])
            precision.append(valid_results[2])
            recall.append(valid_results[3])
            conf_matrix.append(valid_results[4])
            threshold.append(valid_results[5])

            print('-'*50 + '\n' + 'DURATION:' + '\n' + '-'*50)
            epoch_end_time = datetime.now()
            print('Epoch Duration: {}'.format(epoch_end_time-epoch_start_time))

            # SAVE MODEL AND TEMINATION CRITERIA
            if config.train.termination_criteria == "loss":
                last_val = np.mean(valid_results[0])
                if last_val < best_val:
                    torch.save(model.state_dict(), OUT_DIR + f'/best_model.pth')
                    best_val = last_val
                    epoch_timeout_count = 0
                elif epoch_timeout_count < config.train.epoch_timeout:
                    epoch_timeout_count += 1
                else:
                    break
            elif config.train.termination_criteria == "precision":
                last_val = np.mean(valid_results[2])
                if last_val > best_val:
                    torch.save(model.state_dict(), OUT_DIR + f'/best_model.pth')
                    best_val = last_val
                    epoch_timeout_count = 0
                elif epoch_timeout_count < config.train.epoch_timeout:
                    epoch_timeout_count += 1
                else:
                    break
            elif config.train.termination_criteria == "f1_score":
                last_val = np.mean(valid_results[1])
                if last_val > best_val:
                    torch.save(model.state_dict(), OUT_DIR + f'/best_model.pth')
                    best_val = last_val
                    epoch_timeout_count = 0
                elif epoch_timeout_count < config.train.epoch_timeout:
                    epoch_timeout_count += 1
                else:
                    break
            else:
                print("WRONG TERMINATION CRITERIA")
                exit()

        # SAVE RESULTS
        np.save(OUT_DIR + f'/train_loss', np.array(train_loss))
        np.save(OUT_DIR + f'/valid_loss', np.array(valid_loss))
        np.save(OUT_DIR + f'/f1_score', np.array(f1))
        np.save(OUT_DIR + f'/precision', np.array(precision))
        np.save(OUT_DIR + f'/recall', np.array(recall))
        np.save(OUT_DIR + f'/conf_matrix', np.array(conf_matrix))
        np.save(OUT_DIR + f'/threshold', np.array(threshold))

        end_time = datetime.now()
        print('Total Training Duration: {}'.format(end_time-start_time))
        print("Training Done!")
