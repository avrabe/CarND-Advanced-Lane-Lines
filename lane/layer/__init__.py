import threading
from queue import Queue

import attr
import numpy as np
from haikunator import Haikunator

from lane.image import Image, ImageChannel

haikunator = Haikunator()


@attr.s
class Sequential:
    """
    A special layer which sequentially will process the image or image channel
    from one layer to the other.
    """
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


class ParallelThread(threading.Thread):
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
        thread = ParallelThread(self.threadLock, queue, layer)
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
class No_Op:
    name = attr.ib(default=haikunator.haikunate())

    def call(self, image):
        if isinstance(image, Image):
            return Image(image=np.copy(image.image), color=image.color, name=self.name, meta=image.meta.copy())
        elif isinstance(image, Image):
            return ImageChannel(image=np.copy(image.image), color_channel=image.color_channel, name=self.name)
        else:
            assert 1 == 0
