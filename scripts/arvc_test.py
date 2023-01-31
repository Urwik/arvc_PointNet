import numpy as np
import os
import math
from torch.utils.data import DataLoader
import torch

from arvc_utils.arvc_dataset import PLYDataset


def test(device, model, model_file, dataloader, iteration=10):

    # TESTING
    print('-'*50)
    print('TEST')
    model.load_state_dict(torch.load(model_file, map_location=torch.device(device)))

    print('-' * 50)
    print('VALIDATION')
    model.eval()
    correct = 0
    with torch.no_grad():
        for batch, (data, label) in enumerate(dataloader):
            data, label = data.to(device), label.to(device)
            pred, m3x3, m64x64 = model(data.transpose(1,2))
            loss = loss_fn(pred, label)
            test_loss.append(loss.item())
            correct += (pred.argmax(1) == label).type(torch.float).sum().item()

    correct /= len(dataloader.dataset)
    print(f"Validation accuracy: {(100 * correct):>0.1f}%, Validation Loss: {sum(val_loss)/len(dataloader):.3f} \n")



if __name__ == '__main__':

    BATCH_SIZE = 16
    EPOCHS = 100
    INIT_LR = 1e-3

    TRAIN_SPLIT = 0.70
    TEST_SPLIT = 0.15
    VAL_SPLIT = 1 - TRAIN_SPLIT + TEST_SPLIT

    MAX_K = 7

    # ROOT_DIR = os.path.abspath('/home/arvc/Fran/data/ply')
    # OUTPUT_PATH = os.path.abspath('/home/arvc/Fran/PycharmProjects/HKPS/model_save/')

    ROOT_DIR = os.path.abspath('/media/arvc/data/experiments/ouster/simulated/73_pcds/raw/ply')
    OUTPUT_PATH = os.path.abspath('/home/arvc/PycharmProjects/HKPS/model_save/')

    # SELECT DEVICE TO WORK WITH
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Pointnet.PointNet(classes=MAX_K).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=INIT_LR)

    # INSTANCE DATASET
    labels = np.load('cloud_labels.npy')
    dataset = PLYDataset(data_root=ROOT_DIR,
                         _features=(0, 1, 2),
                         _labels=labels,
                         transform=None)

    # Split validation and train
    train_size = math.floor(len(dataset) * TRAIN_SPLIT)
    test_size = math.floor(len(dataset) * TEST_SPLIT)
    val_size = len(dataset) - (train_size + test_size)

    train_set, test_set, val_set = torch.utils.data.random_split(dataset, [train_size, test_size, val_size])

    train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, drop_last=False)
    test_dataloader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, drop_last=False)
    val_dataloader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, drop_last=False)

    print('TRAINING ON: ', device)
    best_accuracy = 0
    for epoch in range(EPOCHS):
        train_loss, val_loss, test_loss = [], [], []

        test(device=device,
              train_loader=train_dataloader,
              val_loader=val_dataloader,
              model=model,
              loss_fn=loss_fn,
              optimizer=optimizer)

    print("Done!")
