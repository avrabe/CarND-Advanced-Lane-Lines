import glob

import attr
import cv2
import numpy as np

from haikunator import Haikunator

from lane.image import Image

haikunator = Haikunator()

def calibrate_camera(files):
    """
    Calibrate the camera.
    :param files:   A string containing the pathname to files used for calibration in unix
                    style pattern expansion.
    :return: The object and image points needed to undistort an image.
    """
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.
    # Make a list of calibration images
    images = glob.glob(files)
    # Step through the list and search for chessboard corners
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        # If found, add object points, image points
        if ret is True:
            objpoints.append(objp)
            imgpoints.append(corners)
    return objpoints, imgpoints


@attr.s
class Undistort:
    """
    Undistort the given image based on the calibration information provided.
    """
    calibrate = attr.ib(default=(None, None))
    name = attr.ib(default=haikunator.haikunate())

    def __attrs_post_init__(self):
        self.mtx = None
        self.dist = None
        self.shape = None

    def call(self, image):
        call_objpoints, call_imgpoints = self.calibrate
        if not self.shape or not self.shape == image.image.shape[0:2]:
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(call_objpoints, call_imgpoints,
                                                               image.image.shape[0:2], None, None)
            self.mtx = mtx
            self.dist = dist
            self.shape = image.image.shape[0:2]
        undist = cv2.undistort(image.image, self.mtx, self.dist, None, self.mtx)
        return Image(image=np.copy(undist), color=image.color, name=self.name, meta=image.meta.copy())