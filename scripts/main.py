import numpy as np
import os
from plyfile import PlyData, PlyElement
from sklearn import preprocessing
import pandas as pd

if __name__ == '__main__':
    dataset_dir = "/media/arvc/data/datasets/ARVC_GZF/train/ply_xyzlabelnormal/"
    file = "00500.ply"
    filename = file.split('.')[0]
    path_to_file = os.path.join(dataset_dir, file)
    ply = PlyData.read(path_to_file)
    data = ply["vertex"].data
    # np.memmap to array
    data = np.array(list(map(list, data)))
    labels_file = 'planes_count.csv'
    name = labels_file.split('.')[0]

    labels_path = os.path.join(dataset_dir, labels_file)

    data_frame = pd.read_csv(labels_path, delimiter=',')
    labels_dict = data_frame.set_index('File')['Planes'].to_dict()

    label = labels_dict.get(int(filename))

    a = 12
