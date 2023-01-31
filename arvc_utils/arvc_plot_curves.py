import matplotlib.pyplot as plt
import numpy as np
import os
import socket

if __name__ == '__main__':

    ROOT_DIR = os.path.abspath('//model_save/bin_seg_xyzcurv/20.01.2023.09-16_50_0.001_ROC')

    # LOAD RESULTS
    train_loss = np.load(ROOT_DIR + f'/train_loss.npy')
    valid_loss = np.load(ROOT_DIR + f'/valid_loss.npy').flatten()
    f1_score = np.load(ROOT_DIR + f'/f1_score.npy').flatten()
    precision = np.load(ROOT_DIR + f'/precision.npy').flatten()
    recall = np.load(ROOT_DIR + f'/recall.npy').flatten()
    conf_matrix = np.load(ROOT_DIR + f'/conf_matrix.npy').flatten()
    threshold = np.load(ROOT_DIR + f'/threshold.npy')
    avg_thrshld = np.mean(threshold[-1])
    print(f'Average Threshold {avg_thrshld}')
    threshold = np.load(ROOT_DIR + f'/threshold.npy').flatten()


    EPOCHS = train_loss.shape[0]
    train_loss = train_loss.flatten()

    # PLOTTING FIGURES
    plt.figure()
    plt.subplot(311)
    plt.title('Training-Validation Loss')
    plt.plot(np.linspace(0, EPOCHS, train_loss.size), train_loss, 'r', label='train')
    plt.plot(np.linspace(0, EPOCHS, valid_loss.size), valid_loss, 'g', label='validation')
    plt.legend()
    plt.yticks(np.arange(0, 1, 0.1))
    plt.ylim([0, 1])

    plt.subplot(312)
    plt.title('Precision-Recall')
    plt.plot(np.linspace(0, EPOCHS, precision.size), precision, 'b', label='precision')
    plt.plot(np.linspace(0, EPOCHS, recall.size), recall, 'y', label='recall')
    plt.legend()
    plt.yticks(np.arange(0, 1, 0.1))
    plt.ylim([0, 1])

    plt.subplot(313)
    plt.title('Threshold')
    plt.plot(np.linspace(0, EPOCHS, threshold.size), threshold, 'g', label='threshold')

    plt.show()
