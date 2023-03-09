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
import warnings
warnings.filterwarnings('ignore')

# IMPORTS PATH TO THE PROJECT
current_model_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
pycharm_projects_path = os.path.dirname(os.path.dirname(current_model_path))
# IMPORTS PATH TO OTHER PYCHARM PROJECTS
sys.path.append(current_model_path)
sys.path.append(pycharm_projects_path)

from model.arvc_PointNet_bs import PointNetDenseCls
from arvc_Utils.Datasets import vis_Test_Dataset
from arvc_Utils.datasetTransforms import np2ply


def test(device_, dataloader_, model_, loss_fn_):
    # TEST
    model_.eval()
    f1_lst, pre_lst, rec_lst, loss_lst, conf_m_lst, files_lst = [], [], [], [], [], []
    current_clouds = 0

    with torch.no_grad():
        for batch, (data, label, filename_) in enumerate(dataloader_):
            data, label = data.to(device_, dtype=torch.float32), label.to(device_, dtype=torch.float32)
            pred, m3x3, m64x64 = model_(data.transpose(1, 2))
            m = torch.nn.Sigmoid()
            pred = m(pred)

            avg_loss = loss_fn_(pred, label)
            loss_lst.append(avg_loss.item())
            files_lst.append(filename_)

            pred_fix, avg_f1, avg_pre, avg_rec, conf_m = compute_metrics(label, pred)
            f1_lst.append(avg_f1)
            pre_lst.append(avg_pre)
            rec_lst.append(avg_rec)
            conf_m_lst.append(conf_m)

            if SAVE_PRED_CLOUDS:
                save_pred_as_ply(data, pred_fix, PRED_CLOUDS_DIR, filename_)

            current_clouds += data.size(0)

            if batch % 1 == 0 or data.size()[0] < dataloader_.batch_size:  # print every 10 batches
                print(f'  [Batch: {current_clouds}/{len(dataloader_.dataset)}],'
                      f'  [File: {str(filename_)}],'
                      f'  [F1 score: {avg_f1:.4f}],'
                      f'  [Precision score: {avg_pre:.4f}],'
                      f'  [Recall score: {avg_rec:.4f}]')

    return loss_lst, f1_lst, pre_lst, rec_lst, conf_m_lst, files_lst


def compute_metrics(label_, pred_):

    pred = pred_.cpu().numpy()
    label = label_.cpu().numpy().astype(int)
    trshld = THRESHOLD
    pred = np.where(pred > trshld, 1, 0).astype(int)

    f1_score_list = []
    precision_list = []
    recall_list = []
    tn_list = []
    fp_list = []
    fn_list = []
    tp_list = []

    batch_size = np.size(pred, 0)
    for i in range(batch_size):
        tmp_labl = label[i]
        tmp_pred = pred[i]

        f1_score_ = metrics.f1_score(tmp_labl, tmp_pred, average='binary')
        precision_ = metrics.precision_score(tmp_labl, tmp_pred, average='binary')
        recall_ = metrics.recall_score(tmp_labl, tmp_pred, average='binary')
        tn_, fp_, fn_, tp_ = metrics.confusion_matrix(tmp_labl, tmp_pred, labels=[0,1]).ravel()

        tn_list.append(tn_)
        fp_list.append(fp_)
        fn_list.append(fn_)
        tp_list.append(tp_)

        f1_score_list.append(f1_score_)
        precision_list.append(precision_)
        recall_list.append(recall_)

    avg_f1_score = np.mean(np.array(f1_score_list))
    avg_precision = np.mean(np.array(precision_list))
    avg_recall = np.mean(np.array(recall_list))
    avg_tn = np.mean(np.array(tn_list))
    avg_fp = np.mean(np.array(fp_list))
    avg_fn = np.mean(np.array(fn_list))
    avg_tp = np.mean(np.array(tp_list))

    return pred, avg_f1_score, avg_precision, avg_recall, (avg_tn, avg_fp, avg_fn, avg_tp)


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


def get_extend_clouds():
    csv_file = os.path.join(MODEL_PATH, 'representative_clouds.csv')
    df = pd.read_csv(csv_file)
    extended_clouds_ = df.iloc[0].tolist()
    extended_clouds_ = np.unique(extended_clouds_).tolist()

    return extended_clouds_


if __name__ == '__main__':
    start_time = datetime.now()

    # --------------------------------------------------------------------------------------------#
    # REMOVE MODELS THAT ARE ALREADY EXPORTED
    models_list = os.listdir(os.path.join(current_model_path, 'saved_models'))

    # models_list = ['2302241531']

    for MODEL_DIR in models_list:
        print(f'-'*50)
        print(f'Testing Model: {MODEL_DIR}')

        MODEL_PATH = os.path.join(current_model_path, 'saved_models', MODEL_DIR)

        config_file_abs_path = os.path.join(MODEL_PATH, 'config.yaml')
        with open(config_file_abs_path) as file:
            config = yaml.safe_load(file)

        # DATASET
        TEST_DIR= "ARVCTRUSS/test/ply_xyzlabelnormal" #config["test"]["TEST_DIR"]
        FEATURES= config["train"]["FEATURES"]
        LABELS= config["train"]["LABELS"]
        NORMALIZE= config["train"]["NORMALIZE"]
        BINARY= config["train"]["BINARY"]
        DEVICE= config["test"]["DEVICE"]
        BATCH_SIZE= config["test"]["BATCH_SIZE"]
        OUTPUT_CLASSES= config["train"]["OUTPUT_CLASSES"]
        SAVE_PRED_CLOUDS= True # config["test"]["SAVE_PRED_CLOUDS"]

        if "ADD_RANGE" in config["train"]:
            ADD_RANGE = config["train"]["ADD_RANGE"]
            ADD_LEN = 1
        else:
            ADD_RANGE = False
            ADD_LEN = 0
        # ------------------------
        # --------------------------------------------------------------------------------------------#
        # CHANGE PATH DEPENDING ON MACHINE
        machine_name = socket.gethostname()
        if machine_name == 'arvc-Desktop':
            TEST_DATA = os.path.join('/media/arvc/data/datasets', TEST_DIR)
            VIS_DATA = os.path.join('/media/arvc/data/datasets', 'ARVCTRUSS/test_visualization/ply_xyzlabelnormal')
        else:
            TEST_DATA = os.path.join('/home/arvc/Fran/data/datasets', TEST_DIR)
            VIS_DATA = os.path.join('/home/arvc/Fran/data/datasets', 'ARVCTRUSS/test_visualization/ply_xyzlabelnormal')

        extended_clouds = get_extend_clouds()
        # --------------------------------------------------------------------------------------------#
        # INSTANCE DATASET
        dataset = vis_Test_Dataset(root_dir = TEST_DATA,
                                   common_clouds_dir = VIS_DATA,
                                   extend_clouds = extended_clouds,
                                   features= FEATURES,
                                   labels = LABELS,
                                   normalize = NORMALIZE,
                                   binary = BINARY,
                                   compute_weights=False,
                                   add_range_= ADD_RANGE)

        # INSTANCE DATALOADER
        test_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, drop_last=False)

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
            PRED_CLOUDS_DIR = os.path.join(MODEL_PATH, "vis_clouds")
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

        f1_score = np.array(results[1])
        precision = np.array(results[2])
        recall = np.array(results[3])
        confusion_matrix_list = np.array(results[4])
        mean_cf = np.mean(confusion_matrix_list, axis=0)
        median_cf = np.median(confusion_matrix_list, axis=0)
        mean_tp, mean_fp, mean_tn, mean_fn = mean_cf[3], mean_cf[1], mean_cf[0], mean_cf[2]
        med_tp, med_fp, med_tn, med_fn = median_cf[3], median_cf[1], median_cf[0], median_cf[2]
        files_list = results[5]

        print('\n\n')
        print(f'Threshold: {THRESHOLD}')
        print(f'[Mean F1_score:  {np.mean(f1_score)}] [Median F1_score:  {np.median(f1_score)}]')
        print(f'[Mean Precision: {np.mean(precision)}] [Median Precision: {np.median(precision)}]')
        print(f'[Mean Recall:    {np.mean(recall)}] [Median Recall:    {np.median(recall)}]')
        print(f'[Mean TP: {mean_tp}, FP: {mean_fp}, TN: {mean_tn}, FN: {mean_fn}] '
              f'[Median TP: {med_tp}, FP: {med_fp}, TN: {med_tn}, FN: {med_fn}]')
        print("Done!")