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
    LUV = 6  # TODO colorchannel conversion
    YUV = 7  # TODO colorchannel conversion
    YCrCb = 8  # TODO colochannel conversion


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
    """
    The representation of an image channel
    """
    color_channel = attr.ib(default=Color.UNKNOWN)
    image = attr.ib(default=None)
    name = attr.ib(default=haikunator.haikunate())
    meta = attr.ib(default={})


@attr.s
class Image:
    """
    The representation of an image.
    """
    color = attr.ib(default=Color.UNKNOWN)
    image = attr.ib(default=None)
    name = attr.ib(default=haikunator.haikunate())
    meta = attr.ib(default={})

    def get_image(self, color):
        if self.color == color:
            return self
        _conv = {}
        if self.color == Color.GRAY:
            _conv = {
                Color.RGB: cv2.COLOR_GRAY2RGB,
                Color.BGR: cv2.COLOR_GRAY2BGR,
                Color.HLS: cv2.COLOR_GRAY2HLS,
                Color.HSV: cv2.COLOR_GRAY2HSV,
                Color.LUV: cv2.COLOR_GRAY2LUV,
                Color.YUV: cv2.COLOR_GRAY2YUV,
                Color.YCrCb: cv2.COLOR_GRAY2YCrCb
            }
        elif self.color == Color.RGB:
            _conv = {
                Color.GRAY: cv2.COLOR_RGB2GRAY,
                Color.BGR: cv2.COLOR_RGB2BGR,
                Color.HLS: cv2.COLOR_RGB2HLS,
                Color.HSV: cv2.COLOR_RGB2HSV,
                Color.LUV: cv2.COLOR_RGB2LUV,
                Color.YUV: cv2.COLOR_RGB2YUV,
                Color.YCrCb: cv2.COLOR_RGB2YCrCb
            }
        elif self.color == Color.BGR:
            _conv = {
                Color.GRAY: cv2.COLOR_BGR2GRAY,
                Color.RGB: cv2.COLOR_BGR2RGB,
                Color.HLS: cv2.COLOR_BGR2HLS,
                Color.HSV: cv2.COLOR_BGR2HSV,
                Color.LUV: cv2.COLOR_BGR2LUV,
                Color.YUV: cv2.COLOR_BGR2YUV,
                Color.YCrCb: cv2.COLOR_BGR2YCrCb
            }
        if _conv:
            img = cv2.cvtColor(np.copy(self.image), _conv[color])
            return Image(color=color, image=img, name=self.name, meta=self.meta.copy())
        assert 1 == 0

    def get_channel(self, channel_color):
        """
        Extract a specific channel or convert it to it.
        :param channel_color: The color of the channel to extract
        :return: An ImageChannel with the extracted color channel.
        """
        if channel_color in [ColorChannel.GRAY, ColorChannel.RED,
                             ColorChannel.BLUE, ColorChannel.GREEN]:
            img = self.image
            if self.color == Color.BGR:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif self.color == Color.HSV:
                img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
            elif self.color == Color.HLS:
                img = cv2.cvtColor(img, cv2.COLOR_HLS2RGB)

            if self.color == Color.GRAY and channel_color == ColorChannel.GRAY:
                img = np.copy(self.image)
                return ImageChannel(color_channel=ColorChannel.GRAY, image=img, name=self.name, meta=self.meta.copy())
            elif self.color == Color.RGB and channel_color == ColorChannel.GRAY:
                img = np.copy(self.image)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                return ImageChannel(color_channel=ColorChannel.GRAY, image=img, name=self.name, meta=self.meta.copy())

            r, g, b = cv2.split(img)
            if channel_color == ColorChannel.RED:
                return ImageChannel(color_channel=ColorChannel.RED, image=np.copy(r), name=self.name,
                                    meta=self.meta.copy())
            elif channel_color == ColorChannel.BLUE:
                return ImageChannel(color_channel=ColorChannel.BLUE, image=np.copy(b), name=self.name,
                                    meta=self.meta.copy())
            elif channel_color == ColorChannel.GREEN:
                return ImageChannel(color_channel=ColorChannel.GREEN, image=np.copy(g), name=self.name,
                                    meta=self.meta.copy())

        if channel_color in [ColorChannel.HUE, ColorChannel.LIGHTNESS, ColorChannel.SATURATION]:
            img = self.image
            if self.color == Color.BGR:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
            elif self.color == Color.RGB:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
            elif self.color == Color.HSV:
                img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

            h, l, s = cv2.split(img)
            if channel_color == ColorChannel.HUE:
                return ImageChannel(color_channel=ColorChannel.HUE, image=np.copy(h), name=self.name,
                                    meta=self.meta.copy())
            elif channel_color == ColorChannel.LIGHTNESS:
                return ImageChannel(color_channel=ColorChannel.LIGHTNESS, image=np.copy(l), name=self.name,
                                    meta=self.meta.copy())
            elif channel_color == ColorChannel.SATURATION:
                return ImageChannel(color_channel=ColorChannel.SATURATION, image=np.copy(s), name=self.name,
                                    meta=self.meta.copy())

        if channel_color in [ColorChannel.HUE, ColorChannel.SATURATION, ColorChannel.VALUE]:
            img = self.image
            if self.color == Color.BGR:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            elif self.color == Color.RGB:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            elif self.color == Color.HLS:
                img = cv2.cvtColor(img, cv2.COLOR_HLS2RGB)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

            h, s, v = cv2.split(img)
            if channel_color == ColorChannel.HUE:
                return ImageChannel(color_channel=ColorChannel.HUE, image=np.copy(h), name=self.name,
                                    meta=self.meta.copy())
            elif channel_color == ColorChannel.SATURATION:
                return ImageChannel(color_channel=ColorChannel.SATURATION, image=np.copy(s), name=self.name,
                                    meta=self.meta.copy())
            elif channel_color == ColorChannel.VALUE:
                return ImageChannel(color_channel=ColorChannel.VALUE, image=np.copy(v), name=self.name,
                                    meta=self.meta.copy())

        print(self.color, channel_color)
        assert 1 == 0

    def crop(self, startx=0, starty=0, endx=0, endy=0):
        """
        Flip the image
        """
        img = np.copy(self.image)
        img = img[starty: endy, startx: endx]
        return Image(color=self.color, image=img, name=self.name, meta=self.meta.copy())

    def scale(self, dsize=(64, 64)):
        """
        Flip the image
        """
        img = np.copy(self.image)
        img = cv2.resize(img, dsize)
        return Image(color=self.color, image=img, name=self.name, meta=self.meta.copy())

    def flip(self):
        """
        Flip the image
        """
        img = cv2.flip(np.copy(self.image), 1)

        return Image(color=self.color, image=img, name=self.name, meta=self.meta.copy())


@attr.s
class FileImage(Image):
    """
    A special Image class which uses a file as input.
    """
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
class Warp:
    """
    Perform a perspective transformation.
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
    """
    Perform the reverse transformation of the "Warp" Layer.
    """
    name = attr.ib(default=haikunator.haikunate())
    minv = attr.ib(default="warp")

    def call(self, image):
        img = np.copy(image.image)
        h, w = img.shape[0:2]
        Minv = image.meta[self.minv]
        img = cv2.warpPerspective(img, Minv, (w, h))
        return Image(image=img, color=image.color, name=self.name, meta=image.meta.copy())


def imwrite(image, output_image):
    """
    Write an image or image channel to a file
    :param image: The Image or ImageChannel
    :param output_image: The filename to write to.
    """
    if isinstance(image, ImageChannel) or image.color == Color.GRAY:
        cv2.imwrite(output_image, cv2.cvtColor(image.image, cv2.COLOR_GRAY2BGR))
    else:
        cv2.imwrite(output_image, cv2.cvtColor(image.image, cv2.COLOR_RGB2BGR))
