import numpy as np

labels_normal = np.load('//arvc_utils/labels_normal_06162.npy')
labels_bin = np.load('//arvc_utils/labels_bin_06162.npy')

np.savetxt("original.csv", labels_normal, delimiter = ",")
np.savetxt("bin.csv", labels_bin, delimiter = ",")