from enum import Enum

import attr
import cv2
import numpy as np
from haikunator import Haikunator

haikunator = Haikunator()


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


@attr.s
class Image:
    color = attr.ib(default=Color.UNKNOWN)
    image = attr.ib(default=None)
    name = attr.ib(default=haikunator.haikunate())

    def get_channel(self, channel):
        if self.color == Color.GRAY and channel == ColorChannel.GRAY:
            img = np.copy(self.image)
            return ImageChannel(color_channel=ColorChannel.GRAY, image=img, name=self.name)
        elif self.color == Color.RGB and channel == ColorChannel.GRAY:
            img = np.copy(self.image)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            return ImageChannel(color_channel=ColorChannel.GRAY, image=img, name=self.name)
        elif self.color == Color.BGR and channel == ColorChannel.GRAY:
            img = np.copy(self.image)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return ImageChannel(color_channel=ColorChannel.GRAY, image=img, name=self.name)
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
            print(img.name, layer.name)
            img = layer.call(img)
            assert isinstance(img, Image) or isinstance(img, ImageChannel)
        return img


@attr.s
class Parallel:
    merge = attr.ib(default=None)
    name = attr.ib(default=haikunator.haikunate())

    def __attrs_post_init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def call(self, image):
        img = []
        for layer in self.layers:
            print(self.name, image.name, layer.name)
            img.append(layer.call(image))
        if self.merge:
            print(self.name, "merge")
            img = self.merge.call(img)
            assert isinstance(img, Image) or isinstance(img, ImageChannel)
        return img


@attr.s
class Undistort:
    objpoints = attr.ib()
    imgpoints = attr.ib()
    name = attr.ib(default=haikunator.haikunate())

    def call(self, image):
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints,
                                                           image.image.shape[0:2], None, None)
        undist = cv2.undistort(image.image, mtx, dist, None, mtx)
        return Image(image=np.copy(undist), color=image.color, name=self.name)


@attr.s
class Magnitude_Sobel_Threshold():
    threshold = attr.ib(default=(0, 255))
    sobel_kernel = attr.ib(default=3)
    color_channel = attr.ib(default=ColorChannel.GRAY)
    name = attr.ib(default=haikunator.haikunate())

    def call(self, image):
        img = image.get_channel(self.color_channel).image.astype(np.float)
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=self.sobel_kernel)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=self.sobel_kernel)
        abs_sobel = np.sqrt(sobelx * sobelx + sobely * sobely)
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel > self.threshold[0]) & (scaled_sobel < self.threshold[1])] = 255
        return ImageChannel(image=np.copy(sxbinary), color_channel=ColorChannel.GRAY, name=self.name)


@attr.s
class Absolute_Sobel_Threshold():
    orient = attr.ib(default='x')
    threshold = attr.ib(default=(0, 255))
    sobel_kernel = attr.ib(default=3)
    color_channel = attr.ib(default=ColorChannel.GRAY)
    name = attr.ib(default=haikunator.haikunate())

    def call(self, image):
        img = image.get_channel(self.color_channel).image.astype(np.float)
        if self.orient == "x":
            sobel = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=self.sobel_kernel)
        else:
            sobel = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=self.sobel_kernel)
        abs_sobel = np.absolute(sobel)
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= self.threshold[0]) & (scaled_sobel <= self.threshold[1])] = 255
        print(sxbinary)
        return ImageChannel(image=np.copy(sxbinary), color_channel=ColorChannel.GRAY, name=self.name)


@attr.s
class Direction_Sobel_Threshold():
    threshold = attr.ib(default=(0, 255))
    sobel_kernel = attr.ib(default=3)
    color_channel = attr.ib(default=ColorChannel.GRAY)
    name = attr.ib(default=haikunator.haikunate())

    def call(self, image):
        img = image.get_channel(self.color_channel).image.astype(np.float)
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=self.sobel_kernel)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=self.sobel_kernel)
        abs_sobelx = np.absolute(sobelx)
        abs_sobely = np.absolute(sobely)
        abs_sobelx = np.arctan2(abs_sobely, abs_sobelx)
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(abs_sobelx >= self.threshold[0]) & (abs_sobelx <= self.threshold[1])] = 255
        return ImageChannel(image=sxbinary, color_channel=ColorChannel.GRAY, name=self.name)


@attr.s
class Merge_Threshold:
    name = attr.ib(default=haikunator.haikunate())

    def call(self, images):
        d = {}
        for i in images:
            d[i.name] = i.image

        combined = np.zeros_like(d['dir'])
        combined[((d['gradx'] >= 1) & (d['grady'] >= 1)) | ((d['mag'] >= 1) & (d['dir'] >= 1))] = 255
        return Image(image=combined, color=Color.GRAY, name=self.name)


def flip(self):
    """
    Flip the image
    """
    try:
        if self.image:
            self.image = cv2.flip(self.image, 1)
    except:
        print("Failure on flip. Reset image.")
        self.image = None


def adjust_brightness(self):
    """
    Randomly adjust the brightness
    """
    if self.image:
        hsv = cv2.cvtColor(self.image, cv2.COLOR_RGB2HSV)  # convert it to hsv

        h, s, v = cv2.split(hsv)
        v += np.clip(v + random.randint(-5, 15), 0, 255).astype('uint8')
        final_hsv = cv2.merge((h, s, v))

        image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)
        self.image = image


def normalize_image(self):
    """
    Normalize the image
    """
    if self.image:
        r, g, b = cv2.split(self.image)
        x = r.copy()
        r = cv2.normalize(r, x)
        g = cv2.normalize(g, x)
        b = cv2.normalize(b, x)
        self.image = cv2.merge((r, g, b))
