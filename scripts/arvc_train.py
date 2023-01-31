import matplotlib.pyplot as plt
import numpy as np
import os
import math
from torch.utils.data import DataLoader
import torch
import socket
import sys

machine_name = socket.gethostname()
if machine_name == 'arvc-Desktop':
    sys.path.append('/')
else:
    sys.path.append('/home/arvc/Fran/PycharmProjects/arvc_PointNet')

from models.PointNet import PointNetDenseCls
from arvc_utils.arvc_dataset import PLYDataset


def train(device, train_loader, val_loader, model, loss_fn, optimizer):
    train_loss, val_loss = [], []
    samples, avg_train_loss = 0, 0

    # TRAINING
    print('TRAINING')
    model.train()
    for batch, (data, label) in enumerate(train_loader, start=1):
        data, label = data.to(device), label.to(device)
        pred, m3x3, m64x64 = model(data.transpose(1, 2))
        label = label.type(torch.long)
        pred = torch.softmax(pred, dim=2)
        pred = pred.transpose(1, 2)
        loss = loss_fn(pred, label)
        train_loss.append(loss.item() * data.size()[0])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        samples += data.size()[0]

        if batch % 2 == 0 or data.size()[0] < train_loader.batch_size:  # print every 2 batches
            avg_train_loss = sum(train_loss) / samples
            print(f' - [Batch: {samples}/{len(train_loader.dataset)}],'
                  f' Train Loss: {avg_train_loss:.4f}')

    # VALIDATION
    print('VALIDATION')
    model.eval()
    samples, correct, avg_val_loss, accuracy = 0, 0, 0, 0
    with torch.no_grad():
        for batch, (data, label) in enumerate(val_loader):
            data, label = data.to(device), label.to(device)
            pred, m3x3, m64x64 = model(data.transpose(1, 2))
            label = label.type(torch.long)
            pred = torch.softmax(pred, dim=2)
            pred = pred.transpose(1, 2)
            loss = loss_fn(pred, label)
            val_loss.append(loss.item() * data.size()[0])
            correct += (pred.argmax(1) == label).type(torch.float).sum().item()
            samples += data.size()[0]

    accuracy = correct / samples
    avg_val_loss = sum(val_loss) / samples
    print(f" - Accuracy: {(100 * accuracy):>0.1f}%, Valid Loss: {avg_val_loss:.4f} \n")

    return accuracy, avg_train_loss, avg_val_loss


def arvc_test(device, dataloader, model, loss_fn):
    test_loss = []
    samples, correct, avg_test_loss = 0, 0, 0
    model.eval()
    with torch.no_grad():
        for batch, (data, label) in enumerate(dataloader):
            data, label = data.to(device), label.to(device)
            pred, m3x3, m64x64 = model(data.transpose(1, 2))
            label = label.type(torch.long)
            pred = torch.softmax(pred, dim=2)
            pred = pred.transpose(1, 2)
            loss = loss_fn(pred, label)
            test_loss.append(loss.item() * data.size()[0])
            correct += (pred.argmax(1) == label).type(torch.float).sum().item()
            samples += data.size()[0]

    accuracy = correct / samples
    avg_test_loss = sum(test_loss) / samples
    print(f"Test accuracy: {(100 * accuracy):>0.1f}%, Test loss: {avg_test_loss:.4f} \n")
    return accuracy, avg_test_loss


if __name__ == '__main__':

    BATCH_SIZE = 16
    EPOCHS = 1
    INIT_LR = 1e-4

    TRAIN_SPLIT = 0.90
    TEST_SPLIT = (1 - TRAIN_SPLIT) / 2
    VAL_SPLIT = (1 - TRAIN_SPLIT) / 2

    MAX_K = 121

    # CHANGE PATH DEPENDING ON MACHINE
    machine_name = socket.gethostname()
    if machine_name == 'arvc-Desktop':
        ROOT_DIR = os.path.abspath('/media/arvc/data/datasets/ARVC_GZF/clouds/ply')
        OUTPUT_PATH = os.path.abspath('//model_save/')
    else:
        ROOT_DIR = os.path.abspath('/home/arvc/Fran/data/datasets/ARVC_GZF/ply')
        OUTPUT_PATH = os.path.abspath('/home/arvc/Fran/PycharmProjects/arvc_PointNet/model_save/')

    # SELECT DEVICE TO WORK WITH
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = PointNetDenseCls(k=MAX_K).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=INIT_LR)

    # INSTANCE DATASET
    # labels = np.load('cloud_labels.npy')

    dataset = PLYDataset(data_root=ROOT_DIR,
                         _features=(0, 1, 2),
                         transform=None)

    # Split validation and train
    train_size = math.floor(len(dataset) * TRAIN_SPLIT)
    test_size = math.floor(len(dataset) * TEST_SPLIT)
    val_size = len(dataset) - (train_size + test_size)

    train_set, test_set, val_set = torch.utils.data.random_split(dataset, [train_size, test_size, val_size],
                                                                 generator=torch.Generator().manual_seed(74))

    train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, drop_last=False)
    test_dataloader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, drop_last=False)
    val_dataloader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, drop_last=False)

    print('TRAINING ON: ', device)
    best_accuracy = 0
    acc, train_loss, val_loss = [], [], []
    for epoch in range(EPOCHS):
        print(f"EPOCH: {epoch} {'-' * 50}")

        results = train(device=device,
                        train_loader=train_dataloader,
                        val_loader=val_dataloader,
                        model=model,
                        loss_fn=loss_fn,
                        optimizer=optimizer)

        accuracy = results[0]
        avg_train_loss = results[1]
        avg_val_loss = results[2]

        acc.append(accuracy)
        train_loss.append(avg_train_loss)
        val_loss.append(avg_val_loss)

        if accuracy > best_accuracy:
            torch.save(model.state_dict(), OUTPUT_PATH + f'/{BATCH_SIZE}_{EPOCHS}_{INIT_LR}_best_model.pth')

    # SAVE RESULTS
    np.save(OUTPUT_PATH + f'/{BATCH_SIZE}_{EPOCHS}_{INIT_LR}_accuracy', np.array(acc))
    np.save(OUTPUT_PATH + f'/{BATCH_SIZE}_{EPOCHS}_{INIT_LR}_train_loss', np.array(train_loss))
    np.save(OUTPUT_PATH + f'/{BATCH_SIZE}_{EPOCHS}_{INIT_LR}_val_loss', np.array(val_loss))

    # print('TESTING ON: ', device)
    # models.load_state_dict(torch.load(OUTPUT_PATH + '/best_model.pth', map_location=device))
    # arvc_test(device=device,
    #      dataloader=test_dataloader,
    #      models=models,
    #      loss_fn=loss_fn)

    # PLOTTING FIGURES
    plt.figure()
    plt.subplot(211)
    plt.title('Training-Validation Loss')
    plt.plot(train_loss, 'b', label='train')
    plt.plot(val_loss, 'r', label='validation')
    plt.legend()
    plt.yticks(np.arange(0, 5, 0.2))
    plt.ylim([0, 5])

    plt.subplot(212)
    plt.title('Accuracy')
    plt.bar(np.arange(0, EPOCHS, 1), acc, color='g')
    plt.ylim([0, 1])
    plt.show()

    print("Done!")
