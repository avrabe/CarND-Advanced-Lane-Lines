import glob
import os.path

import click
import cv2
import sys
from  moviepy.video.io.VideoFileClip import VideoFileClip

from lane import Sequential, Undistort, FileImage, Parallel, Merge_Threshold, Image, Color, ColorChannel, \
    ColorChannel_Threshold, \
    ImageChannel, Absolute_Sobel_Threshold, Warp, LaneLines, No_Op, Unwarp, Overlay, calibrate_camera, GaussianBlur, \
    DeNoise


def binary_threshold():
    color_parallel = Parallel(merge=Merge_Threshold(merge="(('v' >= 1) & ('s' >= 1))", binary=True, name="color"),
                              name="color_threshold")
    color_parallel.add(ColorChannel_Threshold(name="v", color_channel=ColorChannel.VALUE, threshold=(150, 255)))
    color_parallel.add(ColorChannel_Threshold(name="s", color_channel=ColorChannel.SATURATION, threshold=(100, 255)))
    parallel = Parallel(merge=Merge_Threshold(merge="(('gradx' >= 1) & ('grady' >= 1) | ('color' >= 1))", binary=False),
                        name="thresholds")
    parallel.add(color_parallel)
    parallel.add(Absolute_Sobel_Threshold(name="gradx", orient='x', threshold=(50, 255)))
    parallel.add(Absolute_Sobel_Threshold(name="grady", orient='y', threshold=(25, 255)))
    return parallel


def undistort_model():
    model = Sequential()
    model.add(Undistort(calibrate=calibrate_camera('camera_cal/calibration*.jpg'), name="undistort"))
    return model


def full_model():
    model = undistort_model()
    threshold = Sequential()
    #threshold.add(DeNoise())
    threshold.add(GaussianBlur())
    threshold.add(binary_threshold())
    #threshold.add(DeNoise())
    threshold.add(Warp(name="warp", height_pct=.64, bot_width=.50, mid_width=.08))
    threshold.add(LaneLines(name="lane_lines", always_blind_search=False, max_one_eyed_search=1))
    threshold.add(Unwarp(name="unwarp", minv="warp"))
    ll_parallel = Parallel(merge=Overlay(base="undistort"))
    ll_parallel.add(No_Op(name="undistort"))
    ll_parallel.add(threshold)
    model.add(ll_parallel)
    return model


def threshold_model():
    model = undistort_model()
    model.add(GaussianBlur())
    model.add(binary_threshold())
    return model

def warp_threshold_model():
    model = threshold_model()
    model.add(Warp(name="warp", height_pct=.64, bot_width=.60, mid_width=.1))
    return model


models = {
    'full': full_model,
    'undistort': undistort_model,
    'undistort_threshold': threshold_model,
    'undistort_threshold_warp': warp_threshold_model
}


video_model = None

def process_video_image(image):
    img = Image(image=image, color=Color.RGB)
    result = video_model.call(img).image
    return result



@click.group(chain=True)
def cli():
    pass


@cli.command()
@click.option('-o', '--output', default=".",
              help='The output directory',
              type=click.Path(exists=True))
@click.option('-i', '--input', multiple=True,
              help='The input image(s)')
@click.option('-m', '--model', default='full', type=click.Choice(models.keys()))
def videos(output, input, model):
    global video_model
    video_model = models[model]()

    videos = []
    for i in input:
        videos.extend(glob.glob(i))

    for idx, fname in enumerate(videos):
        output_filename = "output_%d_%s" % (idx, os.path.basename(fname))
        output_video = os.path.join(output, output_filename)
        clip1 = VideoFileClip(fname)
        white_clip = clip1.fl_image(process_video_image)  # NOTE: this function expects color images!!
        white_clip.write_videofile(output_video, audio=False)
        print("Processed video %s to %s" % (fname, output_filename))

    sys.exit()


@cli.command()
@click.option('-o', '--output', default=".",
              help='The output directory',
              type=click.Path(exists=True))
@click.option('-i', '--input', multiple=True,
              help='The input image(s)')
@click.option('-m', '--model', default='full', type=click.Choice(models.keys()))
def images(output, input, model):
    model = models[model]()

    images = []
    for i in input:
        images.extend(glob.glob(i))

    for idx, fname in enumerate(images):
        output_filename = "output_%d_%s" % (idx, os.path.basename(fname))
        output_image = os.path.join(output, output_filename)
        img = FileImage(filename=fname)
        foo = model.call(img)
        if isinstance(foo, ImageChannel) or foo.color == Color.GRAY:
            cv2.imwrite(output_image, cv2.cvtColor(foo.image, cv2.COLOR_GRAY2BGR))
        else:
            cv2.imwrite(output_image, cv2.cvtColor(foo.image, cv2.COLOR_RGB2BGR))
        print("Processed image %s to %s" % (fname, output_filename))
    sys.exit()


if __name__ == '__main__':
    cli()
