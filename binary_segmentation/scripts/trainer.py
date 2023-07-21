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

from arvc_Utils.Utils import bcolors
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
        self.valid_results = Results()
        self.global_valid_results = Results()

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
        self.set_scheduler()


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

        self.train_dataset.dataset_size = 40

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
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.configuration.train.init_lr)
        elif self.configuration.train.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.configuration.train.init_lr)
        else:
            print("WRONG OPTIMIZER")
            exit()


    def set_scheduler(self):
        if self.configuration.train.lr_scheduler == "step":
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=2, gamma=0.1, verbose=False)

        elif self.configuration.train.lr_scheduler == "plateau":
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=5, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=False)
        
        else:   
            self.lr_scheduler = None


    def save_global_results(self):
        # SAVE RESULTS
        np.save(self.output_dir + f'/valid_loss', np.array(self.global_valid_results.loss))
        np.save(self.output_dir + f'/f1_score', np.array(self.global_valid_results.f1))
        np.save(self.output_dir + f'/precision', np.array(self.global_valid_results.precision))
        np.save(self.output_dir + f'/recall', np.array(self.global_valid_results.recall))
        np.save(self.output_dir + f'/conf_matrix', np.array(self.global_valid_results.conf_matrix))
        np.save(self.output_dir + f'/threshold', np.array(self.global_valid_results.threshold))

    

    def add_to_global_results(self):
        self.global_valid_results.loss.append(self.valid_avg_loss)
        self.global_valid_results.f1.append(self.valid_avg_f1)
        self.global_valid_results.precision.append(self.valid_avg_precision)
        self.global_valid_results.recall.append(self.valid_avg_recall)
        self.global_valid_results.conf_matrix.append(self.valid_avg_cm)
        self.global_valid_results.threshold.append(self.valid_avg_threshold)


    def compute_mean_valid_results(self):
        self.valid_avg_loss = float(np.mean(self.valid_results.loss))
        self.valid_avg_f1 = float(np.mean(self.valid_results.f1))
        self.valid_avg_precision = float(np.mean(self.valid_results.precision))
        self.valid_avg_recall = float(np.mean(self.valid_results.recall))
        self.valid_avg_threshold = float(np.mean(self.valid_results.threshold))
        self.valid_avg_cm = np.mean(self.valid_results.conf_matrix, axis=0)


    def termination_criteria(self):
        if self.epoch_timeout_count > self.configuration.train.epoch_timeout:
            return True
        else:
            return False


    def improved_results(self):
        # LOSS
        if self.configuration.train.termination_criteria == "loss":
            self.last_value = self.valid_avg_loss

            if self.last_value < self.best_value:
                return True
            else:
                return False

        # PRECISION    
        elif self.configuration.train.termination_criteria == "precision":
            self.last_value = self.valid_avg_precision
            
            if self.last_value > self.best_value:
                return True
            else:
                return False

        # F1 SCORE    
        elif self.configuration.train.termination_criteria == "f1_score":
            self.last_value = self.valid_avg_f1
            
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
            print('-'*50)
            print(f"{bcolors.GREEN} NEW MODEL SAVED {bcolors.ENDC} WITH PRECISION: {bcolors.GREEN}{self.valid_avg_precision:.4f}{bcolors.ENDC}")
            return True
        else:
            self.epoch_timeout_count += 1
            return False

       
    def train(self):
        print('-'*50 + '\n' + 'TRAINING' + '\n' + '-'*50)
        current_clouds = 0
        
        self.model.train()

        for batch, (_, features, label, _) in enumerate(self.train_dataloader):
            self.optimizer.zero_grad()

            features = features.to(self.device, dtype=torch.float32)
            label = label.to(self.device, dtype=torch.float32)
           
            # Evaluate model
            pred, m3x3, m64x64 = self.model(features.transpose(1, 2))

            pred = self.activation_fn(pred)

            # Calulate loss and backpropagate
            avg_train_loss_ = self.configuration.train.loss_fn(pred, label)
            avg_train_loss_.backward()
            self.optimizer.step()

            # Print training progress
            current_clouds += features.size(0)
            if batch % 1 == 0 or features.size(0) < self.train_dataloader.batch_size:  # print every (% X) batches
                print(f' - [Batch: {current_clouds}/{len(self.train_dataloader.dataset)}],'
                    f' / Train Loss: {avg_train_loss_:.4f}')


    def valid(self):
        print('-'*50 + '\n' + 'VALIDATION' + '\n' + '-'*50)
        self.valid_results = Results()
        current_clouds = 0

        self.model.eval()
        with torch.no_grad():
            for batch, (_, features, label, _) in enumerate(self.valid_dataloader):
                features = features.to(self.device, dtype=torch.float32)
                label = label.to(self.device, dtype=torch.float32)

                pred, m3x3, m64x64 = self.model(features.transpose(1, 2))

                pred = self.activation_fn(pred)

                avg_loss = self.configuration.train.loss_fn(pred, label)
                self.valid_results.loss.append(avg_loss.item())

                trshld, pred_fix, avg_f1, avg_pre, avg_rec, conf_m = validation_metrics(label, pred)

                self.valid_results.threshold.append(trshld)
                self.valid_results.f1.append(avg_f1)
                self.valid_results.precision.append(avg_pre)
                self.valid_results.recall.append(avg_rec)
                self.valid_results.conf_matrix.append(conf_m)
                self.valid_results.pred_fix.append(pred_fix)

                current_clouds += features.size(0)

                if batch % 10 == 0 or features.size()[0] < self.valid_dataloader.batch_size:  # print every 10 batches
                    print(f'[Batch: {current_clouds}/{len(self.valid_dataloader.dataset)}]'
                        f'  [Avg Loss: {avg_loss:.4f}]'
                        f'  [Avg F1 score: {avg_f1:.4f}]'
                        f'  [Avg Precision score: {avg_pre:.4f}]'
                        f'  [Avg Recall score: {avg_rec:.4f}]')

        self.compute_mean_valid_results()
        self.add_to_global_results()

    def get_learning_rate(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']


    def update_learning_rate_custom(self):
        prev_lr = self.get_learning_rate()
        for g in self.optimizer.param_groups:
            g['lr'] = ( prev_lr / 10)

        # self.scheduler.step()

    def update_learning_rate(self):
        if self.configuration.train.lr_scheduler == "step":
            self.lr_scheduler.step()

        elif self.configuration.train.lr_scheduler == "plateau":
            self.lr_scheduler.step(self.valid_avg_loss)


class Results:
    def __init__(self):
        self.f1 = []
        self.precision = []
        self.recall = []
        self.conf_matrix = []
        self.loss = []
        self.threshold = []
        self.pred_fix = []




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
            print(f"{bcolors.GREEN} EPOCH: {epoch} {bcolors.ENDC}| LR: {trainer.optimizer.param_groups[0]['lr']} {'-' * 30}") 
            epoch_start_time = datetime.now()


            trainer.train() 
            trainer.valid()
            trainer.update_saved_model()
            
            if trainer.termination_criteria():
                break

            trainer.update_learning_rate()

            print('-'*50); print(f'EPOCH DURATION: {bcolors.ORANGE} {datetime.now()-epoch_start_time}{bcolors.ENDC}'); print('-'*50); print('\n')


        trainer.save_global_results()
        print('\n'); print('-'*50);print(f'{bcolors.GREEN}TRAINING DONE!{bcolors.ENDC}') 
        print(f'TOTAL TRAINING DURATION: {bcolors.ORANGE} {datetime.now()-training_start_time}{bcolors.ENDC}'); print('-'*50); print('\n')

