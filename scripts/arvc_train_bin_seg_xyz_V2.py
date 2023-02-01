import numpy as np
import os
import shutil
import math
from torch.utils.data import DataLoader
import torch
import socket
import sys
import sklearn.metrics as metrics
import yaml
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# IMPORTS PATH TO THE PROJECT
current_project_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
pycharm_projects_path = os.path.dirname(current_project_path)
# IMPORTS PATH TO OTHER PYCHARM PROJECTS
sys.path.append(current_project_path)
sys.path.append(pycharm_projects_path)

from models.arvc_PointNet_bin_seg import PointNetDenseCls
from arvc_Utils.Datasets import PLYDataset

def train(device_, train_loader_, model_, loss_fn_, optimizer_):
    loss_lst = []
    current_clouds = 0

    # TRAINING
    print('-' * 50)
    print('TRAINING')
    print('-'*50)
    model_.train()
    for batch, (data, label, _) in enumerate(train_loader_):
        data, label = data.to(device_, dtype=torch.float32), label.to(device_, dtype=torch.float32)
        pred, m3x3, m64x64 = model_(data.transpose(1, 2))
        m = torch.nn.Sigmoid()
        pred = m(pred)

        avg_train_loss_ = loss_fn_(pred, label)
        loss_lst.append(avg_train_loss_.item())

        optimizer_.zero_grad()
        avg_train_loss_.backward()
        optimizer_.step()

        current_clouds += data.size(0)

        if batch % 1 == 0 or data.size(0) < train_loader_.batch_size:  # print every (% X) batches
            print(f' - [Batch: {current_clouds}/{len(train_loader_.dataset)}],'
                  f' / Train Loss: {avg_train_loss_:.4f}')

    return loss_lst


def valid(device_, dataloader_, model_, loss_fn_):

    # VALIDATION
    print('-' * 50)
    print('VALIDATION')
    print('-'*50)
    model_.eval()
    f1_lst, pre_lst, rec_lst, loss_lst, conf_m_lst, trshld_lst = [], [], [], [], [], []
    current_clouds = 0

    with torch.no_grad():
        for batch, (data, label, _) in enumerate(dataloader_):
            data, label = data.to(device_, dtype=torch.float32), label.to(device_, dtype=torch.float32)
            pred, m3x3, m64x64 = model_(data.transpose(1, 2))
            m = torch.nn.Sigmoid()
            pred = m(pred)

            avg_loss = loss_fn_(pred, label)
            loss_lst.append(avg_loss.item())

            trshld, pred_fix, avg_f1, avg_pre, avg_rec, conf_m = compute_metrics(label, pred)
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


def compute_metrics(label_, pred_):

    pred = pred_.cpu().numpy()
    label = label_.cpu().numpy().astype(int)
    trshld = compute_best_threshold(pred, label)
    pred = np.where(pred > trshld, 1, 0).astype(int)

    f1_score_list = []
    precision_list = []
    recall_list =  []
    tn_list = []
    fp_list = []
    fn_list = []
    tp_list = []

    batch_size = np.size(pred, 0)
    for i in range(batch_size):
        tmp_labl = label[i]
        tmp_pred = pred[i]

        f1_score = metrics.f1_score(tmp_labl, tmp_pred, average='binary')
        precision_ = metrics.precision_score(tmp_labl, tmp_pred)
        recall_ = metrics.recall_score(tmp_labl, tmp_pred)
        tn, fp, fn, tp = metrics.confusion_matrix(tmp_labl, tmp_pred, labels=[0,1]).ravel()

        tn_list.append(tn)
        fp_list.append(fp)
        fn_list.append(fn)
        tp_list.append(tp)

        f1_score_list.append(f1_score)
        precision_list.append(precision_)
        recall_list.append(recall_)

    avg_f1_score = np.mean(np.array(f1_score_list))
    avg_precision = np.mean(np.array(precision_list))
    avg_recall = np.mean(np.array(recall_list))
    avg_tn = np.mean(np.array(tn_list))
    avg_fp = np.mean(np.array(fp_list))
    avg_fn = np.mean(np.array(fn_list))
    avg_tp = np.mean(np.array(tp_list))

    return trshld, pred, avg_f1_score, avg_precision, avg_recall, (avg_tn, avg_fp, avg_fn, avg_tp)


def compute_best_threshold(pred_, gt_):
    trshld_per_cloud = []
    method_ = config["THRESHOLD_METHOD"]
    for cloud in range(len(pred_)):
        if method_ == "roc":
            fpr, tpr, thresholds = metrics.roc_curve(gt_[cloud], pred_[cloud])
            gmeans = np.sqrt(tpr * (1 - fpr))
            index = np.argmax(gmeans)
            trshld_per_cloud.append(thresholds[index])

        elif method_ == "pr":
            precision_, recall_, thresholds = metrics.precision_recall_curve(gt_[cloud], pred_[cloud])
            f1_score_ = (2 * precision_ * recall_) / (precision_ + recall_)
            index = np.argmax(f1_score_)
            trshld_per_cloud.append(thresholds[index])

        elif method_ == "tuning":
            thresholds = np.arange(0.0, 1.0, 0.0001)
            f1_score_ = np.zeros(shape=(len(thresholds)))
            for index, elem in enumerate(thresholds):
                prediction_ = np.where(pred_[cloud] > elem, 1, 0).astype(int)
                f1_score_[index] = metrics.f1_score(gt_[cloud], prediction_)

            index = np.argmax(f1_score_)
            trshld_per_cloud.append(thresholds[index])
        else:
            print('Error in the name of the method to use for compute best threshold')

    return sum(trshld_per_cloud)/len(trshld_per_cloud)


if __name__ == '__main__':

    # HYPERPARAMETERS
    start_time = datetime.now()

    CONFIG_FILE = 'train_configuration.yaml'
    config_file_abs_path = os.path.join(current_project_path, 'config', CONFIG_FILE)
    with open(config_file_abs_path) as file:
        config = yaml.safe_load(file)

    # CHANGE PATH DEPENDING ON MACHINE
    output_dir = os.path.join(current_project_path, config["OUTPUT_DIR"])

    machine_name = socket.gethostname()
    if machine_name == 'arvc-Desktop':
        dataset_dir = os.path.join('/media/arvc/data/datasets', config["DATASET_DIR"])
    else:
        dataset_dir = os.path.join('/home/arvc/Fran/data/datasets', config["DATASET_DIR"])

    # CREATE A FOLDER TO SAVE TRAINING
    folder_name = datetime.today().strftime('%y%m%d_%H%M')
    out_dir = os.path.join(output_dir, folder_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    shutil.copyfile(config_file_abs_path, os.path.join(out_dir, 'config.yaml'))
    # SELECT DEVICE TO WORK WITH
    device = torch.device(config["DEVICE"])

    # INSTANCE DATASET
    dataset = PLYDataset(root_dir=dataset_dir,
                         features=config["FEATURES"],
                         labels=config["LABELS"],
                         normalize=config["NORMALIZE"],
                         binary=config["BINARY"],
                         transform=None)

    # INSTANCIE MODEL, LOSS FUNCTION AND OPTIMIZER
    model = PointNetDenseCls(k=config["OUTPUT_CLASSES"],
                             n_feat=len(config["FEATURES"]),
                             device=device).to(device)
    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["LR"])

    # Split validation and train
    train_size = math.floor(len(dataset) * config["TRAIN_SPLIT"])
    val_size = len(dataset) - train_size

    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size],
                                                       generator=torch.Generator().manual_seed(74))

    # INSTANCE DATALOADERS
    train_dataloader = DataLoader(train_set, batch_size=config["BATCH_SIZE"], num_workers=10, shuffle=True, pin_memory=True, drop_last=False)
    val_dataloader = DataLoader(val_set, batch_size=config["BATCH_SIZE"], num_workers=10, shuffle=True, pin_memory=True, drop_last=False)

    # ---------------------------------------------------------------------------------------------------------------- #
    # --- TRAIN LOOP ------------------------------------------------------------------------------------------------- #
    print('TRAINING ON: ', device)
    epoch_timeout_count = 0
    tc = config["TERMINATION_CRITERIA"]
    if tc == "loss":
        best_val = 1
    else:
        best_val = 0

    f1, precision, recall, conf_matrix, train_loss, valid_loss, threshold = [], [], [], [], [], [], []

    for epoch in range(config["EPOCHS"]):
        print(f"EPOCH: {epoch} {'-' * 50}")
        epoch_start_time = datetime.now()

        train_results = train(device_=device,
                              train_loader_=train_dataloader,
                              model_=model,
                              loss_fn_=loss_fn,
                              optimizer_=optimizer)

        valid_results = valid(device_=device,
                              dataloader_=val_dataloader,
                              model_=model,
                              loss_fn_=loss_fn)

        # GET RESULTS
        train_loss.append(train_results)
        valid_loss.append(valid_results[0])
        f1.append(valid_results[1])
        precision.append(valid_results[2])
        recall.append(valid_results[3])
        conf_matrix.append(valid_results[4])
        threshold.append(valid_results[5])

        print('-' * 50)
        print('DURATION:')
        print('-' * 50)
        epoch_end_time = datetime.now()
        print('Epoch Duration: {}'.format(epoch_end_time-epoch_start_time))

        # SAVE MODEL AND TEMINATION CRITERIA
        if tc == "loss":
            last_val = np.mean(valid_results[0])
            if last_val < best_val:
                torch.save(model.state_dict(), out_dir + f'/best_model.pth')
                best_val = last_val
                epoch_timeout_count = 0
            elif epoch_timeout_count < config["EPOCH_TIMEOUT"]:
                epoch_timeout_count += 1
            else:
                break
        elif tc == "precision":
            last_val = np.mean(valid_results[2])
            if last_val > best_val:
                torch.save(model.state_dict(), out_dir + f'/best_model.pth')
                best_val = last_val
                epoch_timeout_count = 0
            elif epoch_timeout_count < config["EPOCH_TIMEOUT"]:
                epoch_timeout_count += 1
            else:
                break
        elif tc == "f1_score":
            last_val = np.mean(valid_results[1])
            if last_val > best_val:
                torch.save(model.state_dict(), out_dir + f'/best_model.pth')
                best_val = last_val
                epoch_timeout_count = 0
            elif epoch_timeout_count < config["EPOCH_TIMEOUT"]:
                epoch_timeout_count += 1
            else:
                break
        else:
            print("WRONG TERMINATION CRITERIA")
            exit()


    # SAVE RESULTS
    np.save(out_dir + f'/train_loss', np.array(train_loss))
    np.save(out_dir + f'/valid_loss', np.array(valid_loss))
    np.save(out_dir + f'/f1_score', np.array(f1))
    np.save(out_dir + f'/precision', np.array(precision))
    np.save(out_dir + f'/recall', np.array(recall))
    np.save(out_dir + f'/conf_matrix', np.array(conf_matrix))
    np.save(out_dir + f'/threshold', np.array(threshold))

    end_time = datetime.now()
    print('Total Training Duration: {}'.format(end_time-start_time))
    print("Training Done!")
