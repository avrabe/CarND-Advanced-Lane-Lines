import attr
import cv2
import numpy as np
from haikunator import Haikunator

from lane.image import ColorChannel, ImageChannel, Image, Color

haikunator = Haikunator()


@attr.s
class ColorChannel_Threshold:
    """
    Create a threshold image channel based on a specific color channel
    :param threshold: A tuple holding the threshold the values should lay inbetween.
    :param color_channel: The color channel to perform the thresholding.
    :param name: The name of the layer.
    :param binary: Create a binary or a regular image channel image.
    """
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
    """
    Create a threshold image channel based on the magnitude sobel.
    :param threshold: A tuple holding the threshold the values should lay inbetween.
    :param sobel_kernel: The size of the sobel kernel.
    :param color_channel: The color channel to perform the thresholding.
    :param name: The name of the layer.
    :param binary: Create a binary or a regular image channel image.
    """
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
    """
    Create a threshold image channel based on the absolute sobel.
    :param orient: The orientation of the sobel. 'x' for X, otherwise Y.
    :param threshold: A tuple holding the threshold the values should lay inbetween.
    :param sobel_kernel: The size of the sobel kernel.
    :param color_channel: The color channel to perform the thresholding.
    :param name: The name of the layer.
    :param binary: Create a binary or a regular image channel image.
    """
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
    """
    Create a threshold image channel based on the direction sobel.
    :param threshold: A tuple holding the threshold the values should lay inbetween.
    :param sobel_kernel: The size of the sobel kernel.
    :param color_channel: The color channel to perform the thresholding.
    :param name: The name of the layer.
    :param binary: Create a binary or a regular image channel image.
    """
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
    """
    Merge several thresholds together into a new image channel.
    :param merge: A string defining the merge algorithm..
    :param name: The name of the layer.
    :param binary: Create a binary or a regular image channel image.
    """
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
        combined[eval(my_merge)] = value
        return Image(image=combined, color=Color.GRAY, name=self.name, meta=meta)
