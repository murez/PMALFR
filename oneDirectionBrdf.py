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
from utils import *

r0 = 1.38

mask_img = cv2.imread("data/emilymask.png", -1)
pral_img = cv2.imread("data/Diff.png", -1)
tang_img = cv2.imread("data/DiffSpec.png", -1)
norm_img = cv2.imread("data/spec_normal.png")

pral_img = cv2.cvtColor(pral_img, cv2.COLOR_BGRA2BGR)
tang_img = cv2.cvtColor(tang_img, cv2.COLOR_BGRA2BGR)

norm_img = norm_img.astype(np.float32)
norm_img = norm_img / (65536.0 / 2.0) - 1.0
x, y, z = [norm_img[:, :, i] for i in range(3)]
cos_halfway = x / np.sqrt(x ** 2 + y ** 2 + z ** 2)
halfway = np.arccos(cos_halfway)
h, w = halfway.shape

one_halfway = np.reshape(halfway, (h * w, 1))

for range_obj in range_list:
    kde = joblib.load(os.path.join("model", range_obj.name + '.skm'))
    one_p_halfway = kde.score_samples(one_halfway)
    p_halfway = np.reshape(one_p_halfway, (h, w))
    print(p_halfway)
    print(p_halfway.shape)
