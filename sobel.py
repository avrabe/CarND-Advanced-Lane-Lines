import glob
import os.path
import sys

import click
from moviepy.video.io.VideoFileClip import VideoFileClip

from lane import LaneLines, Overlay
from lane.camera import calibrate_camera, Undistort
from lane.image import FileImage, GaussianBlur, Warp, Unwarp, Image, imwrite, Color, ColorChannel
from lane.layer import Sequential, Parallel, No_Op
from lane.threshold import ColorChannel_Threshold, Absolute_Sobel_Threshold, Merge_Threshold


def binary_threshold(smooth=True):
    """"
    :return The Threshold model
    """
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


def undistort_model(smooth=True):
    """"
    :return The Undistort model
    """
    model = Sequential()
    model.add(Undistort(calibrate=calibrate_camera('camera_cal/calibration*.jpg'), name="undistort"))
    return model


def full_model(smooth=True):
    """"
    :return The full pipeline to detect lane lines as model
    """
    model = undistort_model()
    threshold = Sequential()
    # threshold.add(DeNoise())
    threshold.add(GaussianBlur())
    threshold.add(binary_threshold())
    # threshold.add(DeNoise())
    threshold.add(Warp(name="warp", height_pct=.64, bot_width=.50, mid_width=.08))
    threshold.add(LaneLines(name="lane_lines", always_blind_search=False, max_one_eyed_search=10, smooth=smooth))
    threshold.add(Unwarp(name="unwarp", minv="warp"))
    ll_parallel = Parallel(merge=Overlay(base="undistort"))
    ll_parallel.add(No_Op(name="undistort"))
    ll_parallel.add(threshold)
    model.add(ll_parallel)
    return model


def undistort_threshold(smooth=True):
    """"
    :return A partial model to process undistortion and thresholding.
    """
    model = undistort_model()
    model.add(GaussianBlur())
    model.add(binary_threshold())
    return model


def undistort_threshold_warp(smooth=True):
    """"
    :return A partial model to process undistortion, thresholding and warping.
    """
    model = undistort_threshold()
    model.add(Warp(name="warp", height_pct=.64, bot_width=.60, mid_width=.1))
    return model


def undistort_threshold_warp_lanelines(smooth=True):
    """"
    :return A partial model to process undistortion, thresholding, warping and detection of lane lines.
    """
    model = undistort_threshold_warp(smooth)
    model.add(LaneLines(name="lane_lines", always_blind_search=False, max_one_eyed_search=10, smooth=smooth,
                        return_binary_warped=True))
    return model

def undistort_threshold_warp_lanelines_unwarp(smooth=True):
    """"
    :return A partial model to process undistortion, thresholding, warping, detection of lane lines and unwarping.
    """
    model = undistort_threshold_warp_lanelines(smooth)
    model.add(Unwarp(name="unwarp", minv="warp"))
    return model

models = {
    'full': full_model,
    'undistort': undistort_model,
    'undistort_threshold': undistort_threshold,
    'undistort_threshold_warp': undistort_threshold_warp,
    'undistort_threshold_warp_lanelines': undistort_threshold_warp_lanelines,
    'undistort_threshold_warp_lanelines_unwarp': undistort_threshold_warp_lanelines_unwarp
}

video_model = None


def process_video_image(image):
    """
    Helper for the video pipeline.
    :param image: The unprocessed image from the video.
    :return: The resulting image
    """
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
        output_filename = "output_%s_%d_%s" % (model, idx, os.path.basename(fname))
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
    image_model = models[model](smooth=False)

    images = []
    for i in input:
        images.extend(glob.glob(i))

    for idx, fname in enumerate(images):
        output_filename = "output_%s_%d_%s" % (model, idx, os.path.basename(fname))
        output_image = os.path.join(output, output_filename)
        img = FileImage(filename=fname)
        foo = image_model.call(img)
        imwrite(foo, output_image)
        print("Processed image %s to %s" % (fname, output_filename))
    sys.exit()


if __name__ == '__main__':
    cli()
