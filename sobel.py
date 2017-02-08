# import cv2
# import numpy as np
#
#
#
# # Choose a Sobel kernel size
# ksize = 3 # Choose a larger odd number to smooth gradient measurements
#
# # Apply each of the thresholding functions
# gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(0, 255))
# grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(0, 255))
# mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(0, 255))
# dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0, np.pi/2))
# #Try different combinations and see what you get. For example, here is a selection for pixels where both the x and y gradients meet the threshold criteria, or the gradient magnitude and direction are both within their threshold values.
#
# combined = np.zeros_like(dir_binary)
# combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

import glob
import os.path

import cv2
import numpy as np

from lane import Sequential, Undistort, Magnitude_Sobel_Threshold, FileImage, Parallel, Absolute_Sobel_Threshold, \
    Direction_Sobel_Threshold, Merge_Threshold, ColorChannel

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6 * 9, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d points in real world space
imgpoints = []  # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('camera_cal/calibration*.jpg')

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

model = Sequential()
model.add(Undistort(objpoints=objpoints, imgpoints=imgpoints, name="undistort"))
parallel = Parallel(merge=Merge_Threshold())
parallel.add(Magnitude_Sobel_Threshold(name="mag", sobel_kernel=9, threshold=(30, 100)))
parallel.add(Absolute_Sobel_Threshold(name="gradx", orient='x', threshold=(20, 100)))
parallel.add(Absolute_Sobel_Threshold(name="grady", orient='y', threshold=(20, 100)))
parallel.add(Direction_Sobel_Threshold(name="dir", threshold=(20, 150)))
model.add(parallel)

model = Sequential()
model.add(Undistort(objpoints=objpoints, imgpoints=imgpoints, name="undistort"))
model.add(Absolute_Sobel_Threshold(name="gradx", orient='x', threshold=(20, 100),
                                   color_channel=ColorChannel.VALUE))


fname = os.path.join("test_images", "test1.jpg")
img = FileImage(filename=fname)

foo = model.call(img)
print(foo.name, foo.image.shape)

cv2.imwrite("foo.png", cv2.cvtColor(foo.image, cv2.COLOR_GRAY2BGR))
