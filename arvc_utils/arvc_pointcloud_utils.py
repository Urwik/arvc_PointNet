import os
import numpy as np
from plyfile import PlyData, PlyElement


def txt2ply(root_dir, features, binary=True):
    data_path = os.path.abspath(root_dir)

    for _, entry in enumerate(os.scandir(data_path)):
        filename = os.path.splitext(entry.name)[0]
        cloud = np.loadtxt(entry.path, delimiter=';')
        np2ply(cloud, features, filename, binary)


def np2ply(data_array, out_dir, ply_name, features, binary):

    abs_file_path = os.path.join(out_dir, ply_name + '.ply')

    cloud = list(map(tuple, data_array))
    vertex = np.array(cloud, dtype=features)
    el = PlyElement.describe(vertex, 'vertex')
    if binary:
        PlyData([el]).write(abs_file_path)
    else:
        PlyData([el], text=True).write(abs_file_path)


if __name__ == '__main__':
    output_dir = '/home/arvc/PycharmProjects/HKPS/results/'
    input_array = ''

    # feat_XYZRGBXnYnZn = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), ('xn', 'f4'),
    #             ('yn', 'f4'), ('zn', 'f4'), ('label', 'i4')]

    # feat_XYZRGB = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    feat_XYZI = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('intensity', 'f4')]

    array_cloud = np.load(input_array)

    np2ply(array_cloud, output_dir, 'test_cloud', feat_XYZI, binary=False)




