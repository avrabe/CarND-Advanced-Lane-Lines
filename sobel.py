import glob
import os.path

import click
import cv2
from  moviepy.video.io.VideoFileClip import VideoFileClip

from lane import Sequential, Undistort, FileImage, Parallel, Merge_Threshold, Image, Color, ColorChannel, \
    ColorChannel_Threshold, \
    ImageChannel, Absolute_Sobel_Threshold, Warp, LaneLines, No_Op, Unwarp, Overlay, calibrate_camera

model = Sequential()
model.add(Undistort(calibrate=calibrate_camera('camera_cal/calibration*.jpg'), name="undistort"))

ll_parallel = Parallel(merge=Overlay(base="undistort"))

color_parallel = Parallel(merge=Merge_Threshold(merge="(('v' >= 1) & ('s' >= 1))", binary=True, name="color"),
                          name="color_threshold")
color_parallel.add(ColorChannel_Threshold(name="v", color_channel=ColorChannel.VALUE, threshold=(50, 255)))
color_parallel.add(ColorChannel_Threshold(name="s", color_channel=ColorChannel.SATURATION, threshold=(100, 255)))

parallel = Parallel(merge=Merge_Threshold(merge="(('gradx' >= 1) & ('grady' >= 1) | ('color' >= 1))", binary=False),
                    name="thresholds")
parallel.add(color_parallel)
parallel.add(Absolute_Sobel_Threshold(name="gradx", orient='x', threshold=(12, 255)))
parallel.add(Absolute_Sobel_Threshold(name="grady", orient='y', threshold=(25, 255)))

threshold = Sequential()
threshold.add(parallel)
threshold.add(Warp(name="warp", height_pct=.64))
threshold.add(LaneLines(name="lane_lines", always_blind_search=False))
threshold.add(Unwarp(name="warp", minv="warp"))

model.add(ll_parallel)
ll_parallel.add(No_Op(name="undistort"))
ll_parallel.add(threshold)


def process_video_image(image):
    img = Image(image=image, color=Color.RGB)
    result = model.call(img).image
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
def videos(output, input):
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


@cli.command()
@click.option('-o', '--output', default=".",
              help='The output directory',
              type=click.Path(exists=True))
@click.option('-i', '--input', multiple=True,
              help='The input image(s)')
def images(output, input):
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


if __name__ == '__main__':
    cli()
