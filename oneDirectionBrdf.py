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

if True:
    mask_img = cv2.resize(mask_img, (960, 540))
    halfway = cv2.resize(halfway, (960, 540))
    singlespecular = cv2.resize(singlespecular, (960, 540))
    x = cv2.resize(x, (960, 540))
    c = cv2.resize(c, (960, 540))

for range_obj in range_list:
    kde = joblib.load(os.path.join("model", range_obj.name + '.skm'))
    print(time.time())
    kde.score_samples(np.array([0.1, 0.2, 0.3, 0.4, 0.5])[:, np.newaxis])
    print(time.time())
    print(range_obj.name)
    area = range_obj.incolor(mask_img)
    area_halfway = halfway <= np.pi / 4
    area = area * area_halfway
    selected_halfway = halfway[area]
    selected_rho = singlespecular[area]
    selected_x = x[area]
    print(len(selected_x))
    p_halfway = kde.score_samples(selected_halfway[:, np.newaxis])
    c_in_area = selected_rho / r0 / p_halfway * (2 * selected_x - selected_x ** 2)
    c[area] = c_in_area

c = c.astype(np.float32)
np.save('data/c.npy', c)
c_min = np.min(c)
c_max = np.max(c)
c = c - c_min
c = c / (c_max - c_min) * 255
cv2.imwrite("data/c.png", c)
