import numpy as np
import pandas as pd
import csv
import os
import socket
from torch.utils.data import DataLoader
import torch
import sklearn.metrics as metrics
import sys
import yaml
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# IMPORTS PATH TO THE PROJECT
current_model_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
pycharm_projects_path = os.path.dirname(os.path.dirname(current_model_path))
# IMPORTS PATH TO OTHER PYCHARM PROJECTS
sys.path.append(current_model_path)
sys.path.append(pycharm_projects_path)

from model.arvc_PointNet_bs import PointNetDenseCls
from arvc_Utils.Datasets import PLYDataset
from arvc_Utils.datasetTransforms import np2ply


def test(device_, dataloader_, model_, loss_fn_):
    # TEST
    model_.eval()
    f1_lst, pre_lst, rec_lst, loss_lst, conf_m_lst, files_lst = [], [], [], [], [], []
    current_clouds = 0

    with torch.no_grad():
        for batch, (data, label, filename_) in enumerate(tqdm(dataloader_)):
            data, label = data.to(device_, dtype=torch.float32), label.to(device_, dtype=torch.float32)
            pred, m3x3, m64x64 = model_(data.transpose(1, 2))
            m = torch.nn.Sigmoid()
            pred_prob = m(pred)
            pred_lbl = np.where(pred_prob >= THRESHOLD, 1, 0).astype(int)
            files_lst.append(filename_)

            if SAVE_PRED_CLOUDS:
                save_pred_as_ply(data, pred_lbl, PRED_CLOUDS_DIR, filename_)

    return files_lst


def save_pred_as_ply(data_, pred_fix_, output_dir_, filename_):
    data_ = data_.detach().cpu().numpy()
    batch_size = np.size(data_, 0)
    n_points = np.size(data_, 1)

    feat_xyzlabel = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('label', 'u4')]

    for i in range(batch_size):
        xyz = data_[i][:, [0,1,2]]
        actual_pred = pred_fix_[i].reshape(n_points, 1)
        cloud = np.hstack((xyz, actual_pred))
        filename = filename_[0]
        np2ply(cloud, output_dir_, filename, features=feat_xyzlabel, binary=True)


if __name__ == '__main__':
    start_time = datetime.now()

    # --------------------------------------------------------------------------------------------#
    # REMOVE MODELS THAT ARE ALREADY EXPORTED
    models_list = os.listdir(os.path.join(current_model_path, 'saved_models'))

    # models_list = ['2302251201']

    for MODEL_DIR in models_list:
        print(f'Testing Model: {MODEL_DIR}')

        MODEL_PATH = os.path.join(current_model_path, 'saved_models', MODEL_DIR)

        config_file_abs_path = os.path.join(MODEL_PATH, 'config.yaml')
        with open(config_file_abs_path) as file:
            config = yaml.safe_load(file)

        # DATASET
        TEST_DIR= "ARVCOUSTER/ply_xyznormal"
        FEATURES= config["train"]["FEATURES"]

        if FEATURES == [0,1,2,4,5,6]:
            FEATURES = [0,1,2,3,4,5]
        elif FEATURES == [0,1,2,7]:
            FEATURES = [0,1,2,6]

        LABELS= []
        NORMALIZE= config["train"]["NORMALIZE"]
        BINARY= False
        DEVICE= config["test"]["DEVICE"]
        BATCH_SIZE= 1
        OUTPUT_CLASSES= config["train"]["OUTPUT_CLASSES"]
        SAVE_PRED_CLOUDS= True

        if "ADD_RANGE" in config["train"]:
            ADD_RANGE = config["train"]["ADD_RANGE"]
            ADD_LEN = 1
        else:
            ADD_RANGE = False
            ADD_LEN = 0

        # --------------------------------------------------------------------------------------------#
        # CHANGE PATH DEPENDING ON MACHINE
        machine_name = socket.gethostname()
        if machine_name == 'arvc-Desktop':
            TEST_DATA = os.path.join('/media/arvc/data/datasets', TEST_DIR)
        else:
            TEST_DATA = os.path.join('/home/arvc/Fran/data/datasets', TEST_DIR)

        # --------------------------------------------------------------------------------------------#
        # INSTANCE DATASET
        dataset = PLYDataset(root_dir = TEST_DATA,
                             features= FEATURES,
                             labels = LABELS,
                             normalize = NORMALIZE,
                             binary = BINARY,
                             compute_weights=False,
                             add_range_= ADD_RANGE)

        if "ADD_RANGE" in config["train"]:
            ADD_RANGE = config["train"]["ADD_RANGE"]
            ADD_LEN = 1
        else:
            ADD_RANGE = False
            ADD_LEN = 0

        # INSTANCE DATALOADER
        test_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, drop_last=True)

        # SELECT DEVICE TO WORK WITH
        if torch.cuda.is_available():
            device = torch.device(DEVICE)
        else:
            device = torch.device("cpu")

        model = PointNetDenseCls(k=OUTPUT_CLASSES,
                                 n_feat=len(FEATURES) + ADD_LEN,
                                 device=device).to(device)

        loss_fn = torch.nn.BCELoss()

        # MAKE DIR WHERE TO SAVE THE CLOUDS
        if SAVE_PRED_CLOUDS:
            PRED_CLOUDS_DIR = os.path.join(MODEL_PATH, "test_ouster_real")
            if not os.path.exists(PRED_CLOUDS_DIR):
                os.makedirs(PRED_CLOUDS_DIR)

        # LOAD TRAINED MODEL
        model.load_state_dict(torch.load(os.path.join(MODEL_PATH, 'best_model.pth'), map_location=device))
        threshold = np.load(MODEL_PATH + f'/threshold.npy')
        THRESHOLD = np.mean(threshold[-1])

        print('-'*50)
        print('TESTING ON: ', device)
        results = test(device_=device,
                       dataloader_=test_dataloader,
                       model_=model,
                       loss_fn_=loss_fn)

        print("Done!")
