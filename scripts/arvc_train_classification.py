import numpy as np
import os
import math
from torch.utils.data import DataLoader
import torch
import socket
import sys
import sklearn.metrics as metrics
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

machine_name = socket.gethostname()
if machine_name == 'arvc-Desktop':
    sys.path.append('/')
else:
    sys.path.append('/home/arvc/Fran/PycharmProjects/arvc_PointNet')

from model.arvc_PointNet import PointNetCls
from arvc_utils.arvc_dataset import PLYDatasetPlaneCount


def train(device_, train_loader_, model_, loss_fn_, optimizer_):
    loss_lst = []
    current_clouds = 0

    # TRAINING
    print('-' * 50)
    print('TRAINING')
    print('-'*50)
    model_.train()
    for batch, (data, label, files) in enumerate(train_loader_):
        data, label = data.to(device_), label.to(device_)
        data = data.to(torch.float32)
        pred, m3x3, m64x64 = model_(data.transpose(1,2))
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
    acc_lst, f1_lst, pre_lst, rec_lst, loss_lst, conf_m_lst = [], [], [], [], [], []
    current_clouds = 0

    with torch.no_grad():
        for batch, (data, label, _) in enumerate(dataloader_):
            data, label = data.to(device_), label.to(device_)
            data = data.to(torch.float32)
            pred, m3x3, m64x64 = model_(data.transpose(1, 2))

            avg_loss = loss_fn_(pred, label)
            loss_lst.append(avg_loss.item())

            pred_fix, avg_acc, avg_f1, avg_pre, avg_rec, conf_m = compute_metrics(label, pred)
            acc_lst.append(avg_acc)
            f1_lst.append(avg_f1)
            pre_lst.append(avg_pre)
            rec_lst.append(avg_rec)
            conf_m_lst.append(conf_m)

            current_clouds += data.size(0)

            if batch % 10 == 0 or data.size()[0] < dataloader_.batch_size:  # print every 10 batches
                print(f'[Batch: {current_clouds}/{len(dataloader_.dataset)}]'
                      f'  [Avg Loss: {avg_loss:.4f}]'
                      f'  [Avg Accuracy: {avg_acc:.4f}]'
                      f'  [Avg F1 score: {avg_f1:.4f}]'
                      f'  [Avg Precision score: {avg_pre:.4f}]'
                      f'  [Avg Recall score: {avg_rec:.4f}]')

    return loss_lst, acc_lst, f1_lst, pre_lst, rec_lst, conf_m_lst


def compute_metrics(label_, pred_):
    pred_ = torch.argmax(pred_, dim=1) # Get predicted class
    pred = pred_.cpu().numpy().astype(int)
    label = label_.cpu().numpy().astype(int)

    accuracy_ = metrics.accuracy_score(label, pred)
    f1_score = metrics.f1_score(label, pred, average='micro')
    precision_ = metrics.precision_score(label, pred, average='micro')
    recall_ = metrics.recall_score(label, pred, average='micro')
    conf_matrix_ = metrics.confusion_matrix(label, pred, labels=np.arange(0, 50, 1, dtype=int))

    return pred, accuracy_,f1_score, precision_, recall_, conf_matrix_


if __name__ == '__main__':

    # HYPERPARAMETERS
    start_time = datetime.now()
    BATCH_SIZE = 10
    EPOCHS = 3
    INIT_LR = 1e-3

    TRAIN_SPLIT = 0.8
    VAL_SPLIT = 1 - TRAIN_SPLIT

    MAX_K = 50

    # CHANGE PATH DEPENDING ON MACHINE
    machine_name = socket.gethostname()
    if machine_name == 'arvc-Desktop':
        ROOT_DIR = os.path.abspath('/media/arvc/data/datasets/ARVC_GZF/train/ply_xyzlabelnormal_code_test')
        OUTPUT_PATH = os.path.abspath('//model_save/')
    else:
        ROOT_DIR = os.path.abspath('/home/arvc/Fran/data/datasets/ARVC_GZF/train/ply_xyzlabelnormal')
        OUTPUT_PATH = os.path.abspath('/home/arvc/Fran/PycharmProjects/arvc_PointNet/model_save/class_xyz')

    date = datetime.today().strftime('%Y.%m.%d.%H')
    folder_name = date + f'-{BATCH_SIZE}_{EPOCHS}_{INIT_LR}'
    out_dir = os.path.join(OUTPUT_PATH, folder_name)

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # SELECT DEVICE TO WORK WITH
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # INSTANCIE MODEL, LOSS FUNCTION AND OPTIMIZER
    model = PointNetCls(k=MAX_K, n_feat = 3).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=INIT_LR)

    # INSTANCE DATASET
    dataset = PLYDatasetPlaneCount(root_dir = ROOT_DIR,
                                   features = [0,1,2],
                                   labels_file ='planes_count.csv',
                                   normalize =True)

    # Split validation and train
    train_size = math.floor(len(dataset) * TRAIN_SPLIT)
    val_size = len(dataset) - train_size

    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size],
                                                       generator=torch.Generator().manual_seed(74))


    # INSTANCE DATALOADERS
    train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, num_workers=10, shuffle=True, pin_memory=True, drop_last=False)
    val_dataloader = DataLoader(val_set, batch_size=BATCH_SIZE, num_workers=10, shuffle=True, pin_memory=True, drop_last=False)

    # ---------------------------------------------------------------------------------------------------------------- #
    # --- TRAIN LOOP ------------------------------------------------------------------------------------------------- #
    print('TRAINING ON: ', device)
    best_val_loss = 1
    epoch_timeout = 5
    epoch_timeout_count = 0
    accuracy, f1, precision, recall, conf_matrix, train_loss, valid_loss = [], [], [], [], [], [], []

    for epoch in range(EPOCHS):
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
        accuracy.append(valid_results[1])
        f1.append(valid_results[2])
        precision.append(valid_results[3])
        recall.append(valid_results[4])
        conf_matrix.append(valid_results[5])

        epoch_end_time = datetime.now()
        print('-' * 50)
        print('DURATION: {}'.format(epoch_end_time-epoch_start_time))
        print('-' * 50)

        # SAVE MODEL AND TEMINATION CRITERIA
        avg_epoch_val_loss = np.mean(valid_results[0])
        if avg_epoch_val_loss < best_val_loss:
            torch.save(model.state_dict(), out_dir + f'/best_model.pth')
            best_val_loss = avg_epoch_val_loss
            epoch_timeout_count = 0
        elif epoch_timeout_count < epoch_timeout:
            epoch_timeout_count += 1
        else:
            break

    # SAVE RESULTS
    np.save(out_dir + f'/train_loss', np.array(train_loss))
    np.save(out_dir + f'/valid_loss', np.array(valid_loss))
    np.save(out_dir + f'/accuracy', np.array(accuracy))
    np.save(out_dir + f'/f1_score', np.array(f1))
    np.save(out_dir + f'/precision', np.array(precision))
    np.save(out_dir + f'/recall', np.array(recall))
    np.save(out_dir + f'/conf_matrix', np.array(conf_matrix))

    end_time = datetime.now()
    print('Total Training Duration: {}'.format(end_time-start_time))
    print("Training Done!")
