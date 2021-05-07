import cv2
import numpy as np
import os
import scipy
import math
import joblib

from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.utils.fixes import parse_version
from sklearn.model_selection import GridSearchCV
import time
from utils import *
from KDEpy import TreeKDE
import sys
sys.setrecursionlimit(1000000)

r0 = 1.38

mask_img = cv2.imread("data/emilymask.png", -1)
pral_img = cv2.imread("data/Diff.png", -1)
tang_img = cv2.imread("data/DiffSpec.png", -1)
norm_img = cv2.imread("data/spec_normal.png", -1)
mask_img = cv2.resize(mask_img, (2160, 3840))
pral_img = cv2.cvtColor(pral_img, cv2.COLOR_BGRA2BGR)
tang_img = cv2.cvtColor(tang_img, cv2.COLOR_BGRA2BGR)

norm_img = norm_img.astype(np.float32)
norm_img = norm_img / (65536.0 / 2.0) - 1.0

x, y, z = [norm_img[:, :, i] for i in range(3)]
cos_halfway = x / np.sqrt(x ** 2 + y ** 2 + z ** 2)
halfway = np.arccos(cos_halfway)
h, w = halfway.shape

singlespecular = pral_img - tang_img
singlespecular = cv2.cvtColor(singlespecular, cv2.COLOR_BGR2GRAY)

c = np.zeros((h, w), dtype=np.float32)

# if False:
#     mask_img = cv2.resize(mask_img, (960, 540))
#     halfway = cv2.resize(halfway, (960, 540))
#     singlespecular = cv2.resize(singlespecular, (960, 540))
#     x = cv2.resize(x, (960, 540))
#     c = cv2.resize(c, (960, 540))

for range_obj in range_list:
    # kde = joblib.load(os.path.join("model", range_obj.name + '.skm'))
    area = range_obj.incolor(mask_img)
    area_halfway = halfway <= np.pi / 4
    area = area * area_halfway
    selected_halfway = halfway[area]
    selected_rho = singlespecular[area]
    selected_x = x[area]
    print("Now running " + range_obj.name, ", with pixs number", len(selected_x))
    kde_data = np.load(os.path.join("model", range_obj.name + '.npy'))
    kde = TreeKDE(kernel='gaussian', bw='silverman').fit(kde_data)

    if len(selected_x) <= 2300:
        sorted_halfway = np.sort(selected_halfway)
        sortarg_halfway = np.argsort(selected_halfway).tolist()

        sorted_p_halfway = kde.evaluate(sorted_halfway)
        p_halfway = sorted_p_halfway[sortarg_halfway]
    else:
        sorted_halfway = np.sort(selected_halfway)
        sortarg_halfway = np.argsort(selected_halfway).tolist()
        sorted_p_halfway = np.array([])
        for part_sorted_halfway in np.array_split(sorted_halfway, sorted_halfway.size / 2500 + 1):
            part_sorted_p_halfway = kde.evaluate(part_sorted_halfway)
            sorted_p_halfway = np.append(sorted_p_halfway, part_sorted_p_halfway)
        p_halfway = sorted_p_halfway[sortarg_halfway]
    c_in_area = selected_rho / r0 / p_halfway * (2 * selected_x - selected_x ** 2)
    print(np.max(c_in_area))
    c[area] = c_in_area

c = c.astype(np.float32)
np.save('data/c.npy', c)
c_min = np.min(c)
c_max = np.max(c)
c = c - c_min
c = c / (c_max - c_min) * 255
cv2.imwrite("data/c.png", c)
