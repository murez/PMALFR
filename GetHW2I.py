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

rgb_img = cv2.imread("data\smallmasked.png")




b, g, r = [rgb_img[:, :, i] for i in range(3)]
remove_img = b < 0

for range_obj in range_list:
    thisarea = range_obj.inrange(rgb_img)
    rgb_img[thisarea] = range_obj.color
    remove_img = remove_img + thisarea

remove_img = remove_img == 0
rgb_img[remove_img] = (0, 0, 0)
big_img = cv2.resize(rgb_img, (8192, 8192), cv2.INTER_NEAREST)

cv2.imwrite("data/save.png", rgb_img)

cv2.imwrite("data/big_mask.png", big_img)

rgb_normal = cv2.imread("data\diffuse_normal_maya.png")
normal_map = rgb_normal.astype(np.float32) / 255 * 2.0 - 1.0
# (1,0,0) . (b,g,r) / (b^2 + g^2 + r^2)
b, g, r = [normal_map[:, :, i] for i in range(3)]
cos_halfv = b / np.sqrt(b ** 2 + g ** 2 + r ** 2)

for i, range_obj in enumerate(range_list):
    area = range_obj.incolor(big_img)
    x = cos_halfv[area]
    x = np.arccos(x)
    area = x <= np.pi/4
    x = x[area]
    x = np.asarray(x)[:, np.newaxis]

    X_plot = np.linspace(0, np.pi / 4, 100)
    grid_param = {'bandwidth': list(np.arange(0.0, 0.01, 0.005))}

    # kde_grid = GridSearchCV(KernelDensity(), grid_param)
    # kde = kde_grid.fit(x).best_estimator_
    # print(kde_grid.best_params_)
    # x = x.reshape(-1, 1)
    kde = KernelDensity(bandwidth=0.05).fit(x)
    # joblib.dump(kde, os.path.join("model", range_obj.name + '.skm'))
    dens = kde.score_samples(X_plot[:, np.newaxis])
    plt.plot(X_plot, np.exp(dens))
    plt.title(range_obj.name)
    plt.show()

normal_map_eye = np.zeros((8192, 8192))
# normal_map_eye[eyearea] = cos_halfv

# img_rgb[eyearea]
