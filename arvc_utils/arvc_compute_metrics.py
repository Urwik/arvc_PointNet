import matplotlib.pyplot as plt
import numpy as np
import os
import socket
import sklearn.metrics as metrics
from plyfile import PlyData, PlyElement
from datetime import datetime

def compute_metrics(label_, pred_):

    tmp_labl = label_
    tmp_pred = pred_

    f1_score = metrics.f1_score(tmp_labl, tmp_pred)
    precision = metrics.precision_score(tmp_labl, tmp_pred)
    recall = metrics.recall_score(tmp_labl, tmp_pred)
    tn, fp, fn, tp = metrics.confusion_matrix(tmp_labl, tmp_pred).ravel()

    return pred_, f1_score, precision, recall, (tn, fp, fn, tp)


def ply2np(path_to_file):
    plydata = PlyData.read(path_to_file)
    data = plydata.elements[0].data
    # nm.memmap to np.ndarray
    data = np.array(list(map(list, data)))
    array = data[:,3]
    return array

if __name__ == '__main__':

    GT_DIR = ''
    PRED_DIR = ''
    # CHANGE PATH DEPENDING ON MACHINE
    machine_name = socket.gethostname()
    if machine_name == 'arvc-Desktop':
        PRED_DIR = os.path.abspath('//pred_clouds/20.12.2022')
        GT_DIR = os.path.abspath('/media/arvc/data/datasets/ARVC_GZF/test/ply/bin_class')
    else:
        ROOT_DIR = os.path.abspath('/home/arvc/Fran/data/datasets/ARVC_GZF/train/ply/bin_class')


    ground_truth = os.listdir(GT_DIR)
    prediction = os.listdir(PRED_DIR)


    gt_list_sorted = sorted(ground_truth)
    pred_list_sorted = sorted(prediction)

    f1_score_list = []
    precision_list = []
    recall_list =  []
    conf_m = []

    for i in range(len(gt_list_sorted)):
        print(i)
        if i == 999:
            break
        if not i == 716:
            gt_cloud_path = os.path.join(GT_DIR, gt_list_sorted[i])
            pred_cloud_path = os.path.join(PRED_DIR, pred_list_sorted[i])

            gt_cloud = ply2np(gt_cloud_path)
            pred_cloud = ply2np(pred_cloud_path)

            my_metrics = compute_metrics(gt_cloud, pred_cloud)

            f1_score_list.append(my_metrics[1])
            precision_list.append(my_metrics[2])
            recall_list.append(my_metrics[3])
            conf_m.append(my_metrics[4])

    print(f'Mean Precision: {np.mean(np.array(precision_list)):.4f}')
    print(f'Mean Recall: {np.mean(np.array(recall_list)):.4f}')
    print(f'Mean Confusion Matrix: {np.mean(np.array(conf_m), axis=0)}')


    print("Plot Done!")
