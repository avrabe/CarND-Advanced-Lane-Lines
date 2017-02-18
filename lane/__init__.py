import glob
import threading
from enum import Enum
from queue import Queue

import attr
import cv2
import numpy as np
from haikunator import Haikunator

haikunator = Haikunator()


def calibrate_camera(files):
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


class Color(Enum):
    UNKNOWN = 0
    GRAY = 1
    BGR = 2
    RGB = 3
    HLS = 4
    HSV = 5


class ColorChannel(Enum):
    UNKNOWN = 0
    RED = 1
    GREEN = 2
    BLUE = 3
    GRAY = 4
    HUE = 5
    SATURATION = 6
    LIGHTNESS = 7
    VALUE = 8


@attr.s
class ImageChannel:
    color_channel = attr.ib(default=Color.UNKNOWN)
    image = attr.ib(default=None)
    name = attr.ib(default=haikunator.haikunate())
    meta = attr.ib(default={})


@attr.s
class Image:
    color = attr.ib(default=Color.UNKNOWN)
    image = attr.ib(default=None)
    name = attr.ib(default=haikunator.haikunate())
    meta = attr.ib(default={})

    def get_channel(self, channel):
        """
        Extract a specific channel or convert it to it
        :param channel:
        :return:
        """
        if channel in [ColorChannel.GRAY, ColorChannel.RED,
                       ColorChannel.BLUE, ColorChannel.GREEN]:
            img = self.image
            if self.color == Color.BGR:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif self.color == Color.HSV:
                img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
            elif self.color == Color.HLS:
                img = cv2.cvtColor(img, cv2.COLOR_HLS2RGB)

            if self.color == Color.GRAY and channel == ColorChannel.GRAY:
                img = np.copy(self.image)
                return ImageChannel(color_channel=ColorChannel.GRAY, image=img, name=self.name, meta=self.meta.copy())
            elif self.color == Color.RGB and channel == ColorChannel.GRAY:
                img = np.copy(self.image)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                return ImageChannel(color_channel=ColorChannel.GRAY, image=img, name=self.name, meta=self.meta.copy())

            r, g, b = cv2.split(img)
            if channel == ColorChannel.RED:
                return ImageChannel(color_channel=ColorChannel.RED, image=np.copy(r), name=self.name,
                                    meta=self.meta.copy())
            elif channel == ColorChannel.BLUE:
                return ImageChannel(color_channel=ColorChannel.BLUE, image=np.copy(b), name=self.name,
                                    meta=self.meta.copy())
            elif channel == ColorChannel.GREEN:
                return ImageChannel(color_channel=ColorChannel.GREEN, image=np.copy(g), name=self.name,
                                    meta=self.meta.copy())

        if channel in [ColorChannel.HUE, ColorChannel.LIGHTNESS, ColorChannel.SATURATION]:
            img = self.image
            if self.color == Color.BGR:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
            elif self.color == Color.RGB:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
            elif self.color == Color.HSV:
                img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

            h, l, s = cv2.split(img)
            if channel == ColorChannel.HUE:
                return ImageChannel(color_channel=ColorChannel.HUE, image=np.copy(h), name=self.name,
                                    meta=self.meta.copy())
            elif channel == ColorChannel.LIGHTNESS:
                return ImageChannel(color_channel=ColorChannel.LIGHTNESS, image=np.copy(l), name=self.name,
                                    meta=self.meta.copy())
            elif channel == ColorChannel.SATURATION:
                return ImageChannel(color_channel=ColorChannel.SATURATION, image=np.copy(s), name=self.name,
                                    meta=self.meta.copy())

        if channel in [ColorChannel.HUE, ColorChannel.SATURATION, ColorChannel.VALUE]:
            img = self.image
            if self.color == Color.BGR:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            elif self.color == Color.RGB:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            elif self.color == Color.HLS:
                img = cv2.cvtColor(img, cv2.COLOR_HLS2RGB)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

            h, s, v = cv2.split(img)
            if channel == ColorChannel.HUE:
                return ImageChannel(color_channel=ColorChannel.HUE, image=np.copy(h), name=self.name,
                                    meta=self.meta.copy())
            elif channel == ColorChannel.SATURATION:
                return ImageChannel(color_channel=ColorChannel.SATURATION, image=np.copy(s), name=self.name,
                                    meta=self.meta.copy())
            elif channel == ColorChannel.VALUE:
                return ImageChannel(color_channel=ColorChannel.VALUE, image=np.copy(v), name=self.name,
                                    meta=self.meta.copy())

        print(self.color, channel)
        assert 1 == 0


@attr.s
class FileImage(Image):
    filename = attr.ib(default=None)

    def __attrs_post_init__(self):
        assert self.filename is not None
        self._load_image()
        self.name = self.filename

    def _load_image(self):
        """
        Load and resize the image
        """
        img = cv2.imread(self.filename)
        assert img is not None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.image = img
        self.color = Color.RGB


@attr.s
class Sequential:
    name = attr.ib(default=haikunator.haikunate())

    def __attrs_post_init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def call(self, image):
        img = image
        for layer in self.layers:
            # print(img.name, layer.name)
            img = layer.call(img)
            assert isinstance(img, Image) or isinstance(img, ImageChannel)
        return img


class myThread(threading.Thread):
    def __init__(self, threadLock, queue, layer):
        threading.Thread.__init__(self, daemon=True)
        self.threadLock = threadLock
        self.layer = layer
        self.name = layer.name
        self.queue = queue

    def run(self):
        while True:
            (image, img) = self.queue.get()
            ret = self.layer.call(image)
            self.threadLock.acquire()
            img.append(ret)
            self.threadLock.release()
            self.queue.task_done()


@attr.s
class Parallel:
    merge = attr.ib(default=None)
    name = attr.ib(default=haikunator.haikunate())

    def __attrs_post_init__(self):
        self.layers = []
        self.threads = []
        self.threadLock = threading.Lock()

    def add(self, layer):
        self.layers.append(layer)
        queue = Queue()
        thread = myThread(self.threadLock, queue, layer)
        thread.start()
        self.threads.append(thread)

    def call(self, image):
        img = []

        for t in self.threads:
            t.queue.put((image, img))
        for t in self.threads:
            t.queue.join()

        if self.merge:
            # print(self.name, "merge")
            img = self.merge.call(img)
            assert isinstance(img, Image) or isinstance(img, ImageChannel)
        return img


@attr.s
class Undistort:
    calibrate = attr.ib(default=(None, None))
    name = attr.ib(default=haikunator.haikunate())

    def call(self, image):
        call_objpoints, call_imgpoints = self.calibrate
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(call_objpoints, call_imgpoints,
                                                           image.image.shape[0:2], None, None)
        undist = cv2.undistort(image.image, mtx, dist, None, mtx)
        return Image(image=np.copy(undist), color=image.color, name=self.name, meta=image.meta.copy())


@attr.s
class No_Op:
    name = attr.ib(default=haikunator.haikunate())

    def call(self, image):
        if isinstance(image, Image):
            return Image(image=np.copy(image.image), color=image.color, name=self.name, meta=image.meta.copy())
        elif isinstance(image, Image):
            return ImageChannel(image=np.copy(image.image), color_channel=image.color_channel, name=self.name)
        else:
            assert 1 == 0


@attr.s
class GaussianBlur:
    name = attr.ib(default=haikunator.haikunate())
    kernel_size = attr.ib(default=7)

    def call(self, image):
        img = np.copy(image.image)
        img = cv2.GaussianBlur(img, (self.kernel_size, self.kernel_size), 0)
        if isinstance(image, Image):
            return Image(image=img, color=image.color, name=self.name, meta=image.meta.copy())
        elif isinstance(image, Image):
            return ImageChannel(image=img, color_channel=image.color_channel, name=self.name)
        else:
            assert 1 == 0


@attr.s
class DeNoise:
    name = attr.ib(default=haikunator.haikunate())
    kernel_size = attr.ib(default=7)
    h = attr.ib(default=7)

    def call(self, image):
        img = np.copy(image.image)
        if isinstance(image, Image) and not image.color == Color.GRAY:
            img = cv2.fastNlMeansDenoisingColored(img, None, self.h)
            return Image(image=img, color=image.color, name=self.name, meta=image.meta.copy())
        elif isinstance(image, Image) and image.color == Color.GRAY:
            img = cv2.fastNlMeansDenoising(img, None, self.h)
            return Image(image=img, color=image.color, name=self.name, meta=image.meta.copy())
        elif isinstance(image, ImageChannel):
            img = cv2.fastNlMeansDenoising(img, None, self.h)
            return ImageChannel(image=img, color_channel=image.color_channel, name=self.name)
        else:
            assert 1 == 0


@attr.s
class ColorChannel_Threshold:
    threshold = attr.ib(default=(0, 255))
    color_channel = attr.ib(default=ColorChannel.GRAY)
    name = attr.ib(default=haikunator.haikunate())
    binary = attr.ib(default=True)

    def call(self, image):
        value = 1 if self.binary else 255
        img = image.get_channel(self.color_channel).image.astype(np.float)
        out_img = np.zeros_like(img)
        out_img[(img > self.threshold[0]) & (img < self.threshold[1])] = value

        return ImageChannel(image=np.copy(out_img), color_channel=self.color_channel, name=self.name,
                            meta=image.meta.copy())


@attr.s
class Magnitude_Sobel_Threshold:
    threshold = attr.ib(default=(0, 255))
    sobel_kernel = attr.ib(default=3)
    color_channel = attr.ib(default=ColorChannel.GRAY)
    name = attr.ib(default=haikunator.haikunate())
    binary = attr.ib(default=True)

    def call(self, image):
        value = 1 if self.binary else 255
        img = image.get_channel(self.color_channel).image.astype(np.float)
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=self.sobel_kernel)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=self.sobel_kernel)
        abs_sobel = np.sqrt(sobelx * sobelx + sobely * sobely)
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel > self.threshold[0]) & (scaled_sobel < self.threshold[1])] = value
        return ImageChannel(image=np.copy(sxbinary), color_channel=ColorChannel.GRAY, name=self.name,
                            meta=image.meta.copy())


@attr.s
class Absolute_Sobel_Threshold:
    orient = attr.ib(default='x')
    threshold = attr.ib(default=(0, 255))
    sobel_kernel = attr.ib(default=3)
    color_channel = attr.ib(default=ColorChannel.GRAY)
    name = attr.ib(default=haikunator.haikunate())
    binary = attr.ib(default=True)

    def call(self, image):
        value = 1 if self.binary else 255
        img = image.get_channel(self.color_channel).image.astype(np.float)
        if self.orient == "x":
            sobel = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=self.sobel_kernel)
        else:
            sobel = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=self.sobel_kernel)
        abs_sobel = np.absolute(sobel)
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= self.threshold[0]) & (scaled_sobel <= self.threshold[1])] = value
        return ImageChannel(image=np.copy(sxbinary), color_channel=ColorChannel.GRAY, name=self.name,
                            meta=image.meta.copy())


@attr.s
class Direction_Sobel_Threshold:
    threshold = attr.ib(default=(0, 255))
    sobel_kernel = attr.ib(default=3)
    color_channel = attr.ib(default=ColorChannel.GRAY)
    name = attr.ib(default=haikunator.haikunate())
    binary = attr.ib(default=True)

    def call(self, image):
        value = 1 if self.binary else 255
        img = image.get_channel(self.color_channel).image.astype(np.float)
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=self.sobel_kernel)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=self.sobel_kernel)
        abs_sobelx = np.absolute(sobelx)
        abs_sobely = np.absolute(sobely)
        abs_sobelx = np.arctan2(abs_sobely, abs_sobelx)
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(abs_sobelx >= self.threshold[0]) & (abs_sobelx <= self.threshold[1])] = value
        return ImageChannel(image=sxbinary, color_channel=ColorChannel.GRAY, name=self.name, meta=image.meta.copy())


@attr.s
class Merge_Threshold:
    name = attr.ib(default=haikunator.haikunate())
    binary = attr.ib(default=True)
    merge = attr.ib(default="(('gradx' >= 1) & ('grady' >= 1)) | (('mag' >= 1) & ('dir' >= 1))")

    def call(self, images):
        value = 1 if self.binary else 255
        my_merge = self.merge
        d = {}
        last = None
        meta = {}
        for i in range(0, len(images)):
            number = str("'%d'" % i)
            image_name = str("'%s'" % images[i].name)
            parameter = str("d[%s]" % image_name)
            if number in my_merge:
                my_merge = my_merge.replace(number, parameter)
                d[images[i].name] = last = images[i].image
                meta.update(images[i].meta)
            elif image_name in my_merge:
                my_merge = my_merge.replace(image_name, parameter)
                d[images[i].name] = last = images[i].image
                meta.update(images[i].meta)
            else:
                print("The image %s (%d) will be ignored while merging." % (images[i].image, i))
        combined = np.zeros_like(last).astype('uint8')
        # combined[((d['gradx'] >= 1) & (d['grady'] >= 1)) | ((d['mag'] >= 1) & (d['dir'] >= 1))] = value
        combined[eval(my_merge)] = value
        return Image(image=combined, color=Color.GRAY, name=self.name, meta=meta)


@attr.s
class Overlay:
    name = attr.ib(default=haikunator.haikunate())
    base = attr.ib(default=None)

    def call(self, images):
        d = []
        base_image = None
        base_color = None
        meta = {}
        for i in images:
            meta.update(i.meta)
            if i.name == self.base:
                base_image = np.copy(i.image)
                base_color = i.color
            else:
                d.append(i.image)

        for overlay in d:
            bkg = np.copy(overlay)
            for i in range(0, 2):
                nonzero = overlay[:, :, i].nonzero()
                nonzeroy = np.array(nonzero[0])
                nonzerox = np.array(nonzero[1])
                bkg[nonzero] = [255, 255, 255]
            base_image = cv2.addWeighted(base_image, 1.0, bkg, -1.0, 0)
            base_image = cv2.addWeighted(base_image, 1.0, overlay, 1.0, 0)
        if 'lane_lines' in meta.keys():
            side_pos = meta['lane_lines']['side_pos']
            cv2.putText(base_image, side_pos, (50, 50), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
            lane_line_image = meta['lane_lines']['lane_line_image'].image
            resized_image = cv2.resize(lane_line_image, (0, 0), fx=0.3, fy=0.3)

            x_offset = base_image.shape[1] - resized_image.shape[1] - 10
            y_offset = 10
            base_image[y_offset:y_offset + resized_image.shape[0],
            x_offset:x_offset + resized_image.shape[1]] = resized_image

        return Image(image=base_image, color=base_color, name=self.name, meta=meta)


@attr.s
class LaneLines:
    name = attr.ib(default=haikunator.haikunate())
    plot = attr.ib(default=True)
    always_blind_search = attr.ib(default=False)
    max_one_eyed_search = attr.ib(default=2)

    def __attrs_post_init__(self):
        self.left_fit = None
        self.right_fit = None
        self.counter = 0

    def call(self, image):
        binary_warped = image.image if isinstance(image, ImageChannel) else image.get_channel(ColorChannel.GRAY).image
        meta = image.meta.copy()
        if self.always_blind_search or (
                        self.left_fit is None and self.right_fit is None) or self.counter > self.max_one_eyed_search:
            side_pos, out_img, b_warped = self.__blind_search(binary_warped)
            self.counter = 0
        else:
            side_pos, out_img, b_warped = self.__one_eyed_search(binary_warped)
            self.counter += 1
        ret = Image(image=out_img, color=Color.RGB, name=self.name, meta=meta)
        meta[self.name] = {'side_pos': side_pos,
                           'lane_line_image': Image(image=b_warped, color=Color.RGB, name=self.name, meta=meta)}
        return ret

    def __one_eyed_search(self, binary_warped):
        # Assume you now have a new warped binary image 
        # from the next frame of video (also called "binary_warped")
        # It's now much easier to find line pixels!
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 150  # 100
        left_lane_inds = (
            (nonzerox > (
                self.left_fit[0] * (nonzeroy ** 2) + self.left_fit[1] * nonzeroy + self.left_fit[2] - margin)) & (
                nonzerox < (
                    self.left_fit[0] * (nonzeroy ** 2) + self.left_fit[1] * nonzeroy + self.left_fit[2] + margin)))
        right_lane_inds = (
            (nonzerox > (
                self.right_fit[0] * (nonzeroy ** 2) + self.right_fit[1] * nonzeroy + self.right_fit[2] - margin)) & (
                nonzerox < (
                    self.right_fit[0] * (nonzeroy ** 2) + self.right_fit[1] * nonzeroy + self.right_fit[2] + margin)))

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
        out_img = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

        return self.__plot_lane_line(out_img)

    def __blind_search(self, binary_warped):
        # Assuming you have created a warped binary image called "binary_warped"
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[int(binary_warped.shape[0] / 2):, :], axis=0)
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(binary_warped.shape[0] / nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 70
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
                nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
                nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        self.left_fit = np.polyfit(lefty, leftx, 2)
        self.right_fit = np.polyfit(righty, rightx, 2)

        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        return self.__plot_lane_line(out_img)
        # return self.__one_eyed_search(binary_warped)

    def __plot_lane_line(self, binary_warped):
        # Generate x and y values for plotting
        out_img = np.zeros_like(binary_warped)
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        left_fitx = self.left_fit[0] * ploty ** 2 + self.left_fit[1] * ploty + self.left_fit[2]
        right_fitx = self.right_fit[0] * ploty ** 2 + self.right_fit[1] * ploty + self.right_fit[2]

        xm_per_pix = 3.7 / 614
        camera_center = (left_fitx[-1] + right_fitx[-1]) / 2
        center_diff = float((camera_center - binary_warped.shape[1] / 2) * xm_per_pix)
        side_pos = 'left'
        if center_diff <= 0:
            side_pos = 'right'
        side_pos = "%.2fm to the %s (center %.2f %.2f) %.2f" % \
                   (center_diff, side_pos, camera_center, binary_warped.shape[1] / 2, xm_per_pix)
        margin = 10

        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        for i in [binary_warped, out_img]:
            cv2.fillPoly(i, np.int_([left_line_pts]), (0, 255, 255))
            cv2.fillPoly(i, np.int_([right_line_pts]), (0, 255, 255))

        return side_pos, out_img, binary_warped


@attr.s
class Warp:
    """
    bot_width: The percent of the bottom trapezoid height
    mid_width: The percent of the middle trapezoid height
    height_pct: The percent for the trapezoid height
    bottom_trim: Percent from top to bottom to crop the car hood
    """
    name = attr.ib(default=haikunator.haikunate())
    bot_width = attr.ib(default=.76)
    mid_width = attr.ib(default=.08)
    height_pct = attr.ib(default=0.62)
    bottom_trim = attr.ib(default=.935)
    offset = attr.ib(default=.25)

    def call(self, image):
        img = np.copy(image.image)
        if isinstance(image, ImageChannel) or image.color == Color.GRAY:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            color = Color.RGB
        else:
            color = image.color
        shape = imshape = img.shape
        h, w = img.shape[0:2]
        color4 = [40, 40, 50]

        # src = np.array([[(100, imshape[0]), (imshape[1] / 2 - 75, imshape[0] * 0.65),
        #                      (imshape[1] / 2 + 75, imshape[0] * 0.65), (imshape[1] - 100, imshape[0])]],
        #                    dtype=np.float32)

        src = np.array([[w * (.5 - self.mid_width / 2), h * self.height_pct],
                        [w * (.5 + self.mid_width / 2), h * self.height_pct],
                        [w * (.5 + self.bot_width / 2), h * self.bottom_trim],
                        [w * (.5 - self.bot_width / 2), h * self.bottom_trim]],
                       dtype=np.float32)
        # cv2.line(img, (int(vertices[0][0][0]), int(vertices[0][0][1])),
        #         (int(vertices[0][1][0]), int(vertices[0][1][1])), color4, 3)
        # cv2.line(img, (int(vertices[0][1][0]), int(vertices[0][1][1])),
        #         (int(vertices[0][2][0]), int(vertices[0][2][1])), color4, 3)
        # cv2.line(img, (int(vertices[0][2][0]), int(vertices[0][2][1])),
        #         (int(vertices[0][3][0]), int(vertices[0][3][1])), color4, 3)
        # tl = [[0, h]]
        # tr = [[0, 0]]
        # br = [[w, 0]]
        # bl = [[w, h]]
        # dst = np.float32([tl, tr, br, bl])
        call_offset = w * self.offset
        dst = np.array([[call_offset, 0],
                        [w - call_offset, 0],
                        [w - call_offset, h],
                        [call_offset, h]],
                       dtype=np.float32)
        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)
        meta = image.meta.copy()
        meta[self.name] = Minv
        # e) use cv2.warpPerspective() to warp your image to a top-down view
        img = cv2.warpPerspective(img, M, (w, h))
        return Image(image=img, color=color, name=self.name, meta=meta)


@attr.s
class Unwarp:
    name = attr.ib(default=haikunator.haikunate())
    minv = attr.ib(default="warp")

    def call(self, image):
        img = np.copy(image.image)
        h, w = img.shape[0:2]
        Minv = image.meta[self.minv]
        img = cv2.warpPerspective(img, Minv, (w, h))
        return Image(image=img, color=image.color, name=self.name, meta=image.meta.copy())
