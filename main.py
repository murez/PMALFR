import cv2
import numpy as np
import open3d as o3d

img_cross = cv2.imread("DigitalEmily2_FlashCross/CC_Cross/cam0_Cross_FromCR2_WB.exr"
                       , cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
img_parallel = cv2.imread("DigitalEmily2_FlashParallel/CC_Mixed/cam0_MixedFlash_FromCR2_WB.exr"
                          , cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

img_specular = img_parallel - img_cross

mesh = o3d.io.read_triangle_mesh("Emily_2_1_OBJ/Emily_2_1.obj")
cam_in = o3d.camera.PinholeCameraIntrinsic(
    3482, 5218, 22977.3, 22917.8, 1741, 2609
)
cam_ex = np.array([[0.668428, -0.0098941, -0.743711, -119.556],
                   [0.00205749, 0.999932, -0.0114536, 3.04448],
                   [0.743774, 0.00612572, 0.668403, 103.007],
                   [0, 0, 0, 1]])


cam_cal = o3d.camera.PinholeCameraParameters()
cam_cal.extrinsic = cam_ex
cam_cal.intrinsic = cam_in

eyerange = [(0,20),()]