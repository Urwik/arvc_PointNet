import os
import sys
import shutil
import socket
from datetime import datetime

import math
import numpy as np
import sklearn.metrics as metrics

import torch
from torchsummary import summary
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


class Trainer():

    def __init__(self, config_obj_):
        self.configuration = config_obj_ #type: cfg
        self.train_abs_path: str
        self.valid_abs_path: str
        self.output_dir: str
        self.best_value: float
        self.last_value: float
        self.activation_fn = self.configuration.train.activation_fn
        self.device = self.configuration.train.device

        self.epoch_timeout_count = 0

        self.setup()


    def setup(self):

        self.make_outputdir()
        self.save_config_file()
        self.instantiate_dataset()
        self.instantiate_dataloader()
        self.set_device()
        self.set_model()
        self.set_optimizer()



    def make_outputdir(self):
        OUT_DIR = os.path.join(current_model_path, self.configuration.train.output_dir.__str__())

        folder_name = datetime.today().strftime('%y%m%d%H%M%S')
        OUT_DIR = os.path.join(OUT_DIR, folder_name)

        if not os.path.exists(OUT_DIR):
            os.makedirs(OUT_DIR)

        self.output_dir = OUT_DIR


    def save_config_file(self):
        shutil.copyfile(self.configuration.config_path, os.path.join(self.output_dir, 'config.yaml'))


    def prefix_path(self):

        machine_name = socket.gethostname()
        prefix_path = ''
        if machine_name == 'arvc-fran':
            prefix_path = '/media/arvc/data/datasets'
        else:
            prefix_path = '/home/arvc/Fran/data/datasets'

        return prefix_path


    def instantiate_dataset(self):

        self.train_abs_path = os.path.join(self.prefix_path(), self.configuration.train.train_dir.__str__())
        self.valid_abs_path = os.path.join(self.prefix_path(), self.configuration.train.valid_dir.__str__())
        
        self.train_dataset = PLYDataset(_mode='train',
                                   _root_dir= self.train_abs_path,
                                   _coord_idx= self.configuration.train.coord_idx,
                                   _feat_idx=self.configuration.train.feat_idx, 
                                   _label_idx=self.configuration.train.label_idx,
                                   _normalize=self.configuration.train.normalize,
                                   _binary=self.configuration.train.binary, 
                                   _add_range=self.configuration.train.add_range,   
                                   _compute_weights=self.configuration.train.compute_weights)

        if self.configuration.train.use_valid_data:
            self.valid_dataset = PLYDataset(_mode='train',
                                       _root_dir=self.valid_abs_path,
                                       _coord_idx=self.configuration.train.coord_idx,
                                       _feat_idx=self.configuration.train.feat_idx, 
                                       _label_idx=self.configuration.train.label_idx,
                                       _normalize=self.configuration.train.normalize,
                                       _binary=self.configuration.train.binary, 
                                       _add_range=self.configuration.train.add_range,   
                                       _compute_weights=self.configuration.train.compute_weights)  
        else:
            # SPLIT VALIDATION AND TRAIN
            train_size = math.floor(len(self.train_dataset) * self.configuration.train.train_split)
            val_size = len(self.train_dataset) - train_size
            self.train_dataset, self.valid_dataset = torch.utils.data.random_split(self.train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(74))


    def instantiate_dataloader(self):
        # INSTANCE DATALOADERS
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.configuration.train.batch_size, num_workers=10,
                                      shuffle=True, pin_memory=True, drop_last=False)
        self.valid_dataloader = DataLoader(self.valid_dataset, batch_size=self.configuration.train.batch_size, num_workers=10,
                                      shuffle=True, pin_memory=True, drop_last=False)


    def set_device(self):
        if torch.cuda.is_available():
            self.device = torch.device(self.configuration.train.device)
        else:
            self.device = torch.device("cpu")


    def set_model(self):
        add_len = 1 if self.configuration.train.add_range else 0
        feat_len = len(self.configuration.train.feat_idx) + add_len

        self.model = PointNetDenseCls(  k=self.configuration.train.output_classes,
                                        n_feat=feat_len,
                                        device=self.configuration.train.device.__str__()).to(self.configuration.train.device)
        

        pytorch_total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('-' * 50)
        print(self.model.__class__.__name__)
        print(f"N Parameters: {pytorch_total_params}\n\n")


    def set_optimizer(self):
        if self.configuration.train.optimizer == "adam":
            return torch.optim.Adam(self.model.parameters(), lr=self.configuration.train.init_lr)
        elif self.configuration.train.optimizer == "sgd":
            return torch.optim.SGD(self.model.parameters(), lr=self.configuration.train.init_lr)
        else:
            print("WRONG OPTIMIZER")
            exit()



    def save_results(self):
        # SAVE RESULTS
        np.save(self.output_dir + f'/train_loss', np.array(train_loss))
        np.save(self.output_dir + f'/valid_loss', np.array(valid_loss))
        np.save(self.output_dir + f'/f1_score', np.array(f1))
        np.save(self.output_dir + f'/precision', np.array(precision))
        np.save(self.output_dir + f'/recall', np.array(recall))
        np.save(self.output_dir + f'/conf_matrix', np.array(conf_matrix))
        np.save(self.output_dir + f'/threshold', np.array(threshold))

    
    def append_results(self):
        # APPEND RESULTS
        train_loss.append(np.mean(train_results[0]))
        valid_loss.append(np.mean(valid_results[0]))
        f1.append(np.mean(valid_results[1]))
        precision.append(np.mean(valid_results[2]))
        recall.append(np.mean(valid_results[3]))
        conf_matrix.append(np.mean(valid_results[4]))
        threshold.append(np.mean(valid_results[5]))


    def termination_criteria(self):
        if self.epoch_timeout_count > self.configuration.train.epoch_timeout:
            return True
        else:
            return False


    def improved_results(self):
        # LOSS
        if self.configuration.train.termination_criteria == "loss":
            self.last_value = np.mean(valid_results[0])

            if self.last_value < self.best_value:
                return True
            else:
                return False

        # PRECISION    
        elif self.configuration.train.termination_criteria == "precision":
            self.last_value = np.mean(valid_results[2])
            
            if self.last_value > self.best_value:
                return True
            else:
                return False

        # F1 SCORE    
        elif self.configuration.train.termination_criteria == "f1_score":
            self.last_value = np.mean(valid_results[1])
            
            if self.last_value > self.best_value:
                return True
            else:
                return False
            
        else:
            print("WRONG TERMINATION CRITERIA")
            exit()


    def update_saved_model(self):

        if self.improved_results():
            torch.save(self.model.state_dict(), self.output_dir + f'/best_model.pth')
            self.best_value = self.last_value
            self.epoch_timeout_count = 0
            return True
        else:
            self.epoch_timeout_count += 1
            return False

       
    def train(self):
        print('-'*50 + '\n' + 'TRAINING' + '\n' + '-'*50)
        current_clouds = 0
        
        self.model.train()

        for batch, (_, data, label, _) in enumerate(self.train_dataloader):
            self.set_optimizer().zero_grad()

            data = data.to(self.device, dtype=torch.float32)
            label = label.to(self.device, dtype=torch.float32)
           
            # Evaluate model
            pred, m3x3, m64x64 = self.model(data.transpose(1, 2))

            pred = self.activation_fn(pred)

            # Calulate loss and backpropagate
            avg_train_loss_ = self.configuration.train.loss_fn(pred, label)
            avg_train_loss_.backward()
            self.set_optimizer().step()

            # Print training progress
            current_clouds += data.size(0)
            if batch % 10 == 0 or data.size(0) < self.train_dataloader.batch_size:  # print every (% X) batches
                print(f' - [Batch: {current_clouds}/{len(self.train_dataloader.dataset)}],'
                    f' / Train Loss: {avg_train_loss_:.4f}')


    def valid(self):
        print('-'*50 + '\n' + 'VALIDATION' + '\n' + '-'*50)
        current_clouds = 0

        self.model.eval()

        f1_lst, pre_lst, rec_lst, loss_lst, conf_m_lst, trshld_lst = [], [], [], [], [], []

        with torch.no_grad():
            for batch, (_, data, label, _) in enumerate(self.valid_dataloader):
                data = data.to(self.device, dtype=torch.float32)
                label = label.to(self.device, dtype=torch.float32)
                
                pred, m3x3, m64x64 = self.model(data.transpose(1, 2))

                pred = self.activation_fn(pred)

                avg_loss = self.configuration.train.loss_fn(pred, label)
                loss_lst.append(avg_loss.item())

                trshld, pred_fix, avg_f1, avg_pre, avg_rec, conf_m = validation_metrics(label, pred)


                trshld_lst.append(trshld)
                f1_lst.append(avg_f1)
                pre_lst.append(avg_pre)
                rec_lst.append(avg_rec)
                conf_m_lst.append(conf_m)

                current_clouds += data.size(0)

                if batch % 10 == 0 or data.size()[0] < self.valid_dataloader.batch_size:  # print every 10 batches
                    print(f'[Batch: {current_clouds}/{len(self.valid_dataloader.dataset)}]'
                        f'  [Avg Loss: {avg_loss:.4f}]'
                        f'  [Avg F1 score: {avg_f1:.4f}]'
                        f'  [Avg Precision score: {avg_pre:.4f}]'
                        f'  [Avg Recall score: {avg_rec:.4f}]')

        return loss_lst, f1_lst, pre_lst, rec_lst, conf_m_lst, trshld_lst



class Results:
    def __init__(self):
        self.f1 = []
        self.precision = []
        self.recall = []
        self.conf_matrix = []
        self.train_loss = []
        self.valid_loss = []
        self.threshold = []

    def add_train(self, results):
        self.f1.append(results[0])
        self.precision.append(results[1])
        self.recall.append(results[2])
        self.conf_matrix.append(results[3])
        self.train_loss.append(results[4])
        self.valid_loss.append(results[5])
        self.threshold.append(results[6])

    def add_valid(self, results):
        print(results)



def get_config(_config_file):
    config_file_abs_path = os.path.join(current_model_path, 'config', _config_file)
    config = cfg(config_file_abs_path)

    return config



if __name__ == '__main__':

    # Files = os.listdir(os.path.join(current_model_path, 'config'))
    Files = ['config_new_version.yaml']
    for configFile in Files:
        training_start_time = datetime.now()

        config = get_config(configFile)

        trainer = Trainer(config)

        global_results = Results()

        # ------------------------------------------------------------------------------------------------------------ #
        # --- TRAIN LOOP --------------------------------------------------------------------------------------------- #
        print('TRAINING ON: ', trainer.device.__str__())

        trainer.best_value = 1 if config.train.termination_criteria == "loss" else 0


        for epoch in range(config.train.epochs):
            print(f"EPOCH: {epoch} {'-' * 50}")
            epoch_start_time = datetime.now()

            # update_lr(trainer.optimizer, epoch, config.train.lr_decay, config.train.init_lr)

            train_results = trainer.train() 
            valid_results = trainer.valid()

            global_results.add_train(train_results)
            global_results.add_valid(valid_results)


            print('-'*50 + '\n' + 'DURATION:' + '\n' + '-'*50)
            epoch_end_time = datetime.now()
            print('Epoch Duration: {}'.format(epoch_end_time-epoch_start_time))


        training_end_time = datetime.now()
        print('Total Training Duration: {}'.format(training_end_time-training_start_time))
        print("Training Done!")
