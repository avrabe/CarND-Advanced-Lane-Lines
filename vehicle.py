"""
Based on sample code form Udactiy SDND class: Vehicle Detection and Tracking - HOG Classify
"""
import collections
import glob
import os
import sys
import threading
import time
from itertools import product
from queue import Queue

import click
import cv2
import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip
from scipy.ndimage.measurements import label
from skimage.feature import hog
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from lane.image import FileImage, Color, Image
from sobel import models


def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                     vis=False, feature_vec=True):
    """
    Exract the hog feature out of the image
    :param img: The inout image
    :param orient: The number of orientations of the hog
    :param pix_per_cell: The amount of pix for a cell
    :param cell_per_block: The amount of cells per block
    :param vis: Shall a visual image be returned?
    :param feature_vec: Shall the feature vector be calculated
    :return: The feature vector and the output image (if vis is set to True)
    """
    # Call with two outputs if vis==True
    if vis:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec)
        ##features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
        ##                          cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
        ##                          visualise=True, feature_vector=feature_vec)
        ##print(hog_image.shape, np.max(hog_image))
        ##hog_image *=255
        ##hog_image = cv2.cvtColor(hog_image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        ##cv2.imwrite("hog_image.png", hog_image)
        return features


# Define a function to compute binned color features
def bin_spatial(img, size=(32, 32)):
    """
    Create features based on the color features of an image
    :param img: The input image
    :param size: The size of the returned feature (It's a resized image)
    :return: The feature vector
    """
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel()
    # Return the feature vector
    return features


# Define a function to compute color histogram features
def color_hist(img, nbins=32):
    """
    Calculate the features based on the color histogram
    :param img: The input image
    :param nbins: The number of bins to put the histogram into
    :return: The calculated feature vector
    """
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


def files_iter(imgs):
    """
    An iterator returning Images and flipped images
    :param imgs: A list containing filenames as list
    :return: The image and flipped image
    """
    for _file in imgs:
        image = FileImage(filename=_file)
        yield image
        yield image.flip()


# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, cspace=Color.RGB, hog_channel=0,
                     orient=None, pix_per_cell=None, cell_per_block=None):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for image in files_iter(imgs):
        # Read in each one by one
        # image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'

        # Call get_hog_features() with vis=False, feature_vec=True
        f = features_features_features(image, cspace=cspace, hog_channel=hog_channel,
                                       orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block)
        features.append(f)
    # Return list of feature vectors
    return features


class HogThread(threading.Thread):
    def __init__(self, queue):
        threading.Thread.__init__(self, daemon=True)
        self.queue = queue
        self.features = None

    def run(self):
        while True:
            (feature_image, channel, orient, pix_per_cell, cell_per_block) = self.queue.get()
            self.features = get_hog_features(feature_image[:, :, channel],
                                             orient, pix_per_cell, cell_per_block,
                                             vis=False, feature_vec=True)
            self.queue.task_done()


threads = []
for i in [0, 1, 2]:
    queue = Queue()
    thread = HogThread(queue)
    thread.start()
    threads.append(thread)


def features_features_features(image=None, cspace=Color.YCrCb, orient=10,
                               pix_per_cell=8, cell_per_block=2, hog_channel="ALL",
                               spatial_size=(16, 16),
                               hist_bins=16):
    feature_image = image.get_image(color=cspace).image
    my_range = range(feature_image.shape[2]) if hog_channel == 'ALL' else range(hog_channel, hog_channel + 1)
    hog_features = []

    for channel in my_range:
        # threads[channel].queue.put((feature_image, channel, orient, pix_per_cell, cell_per_block))

        hog_features.append(get_hog_features(feature_image[:, :, channel],
                                             orient, pix_per_cell, cell_per_block,
                                             vis=False, feature_vec=True))
    # for t in threads:
    #    t.queue.join()
    #    hog_features.append(t.features)
    hog_features = np.ravel(hog_features) if len(hog_features) > 1 else hog_features[0]
    f = hog_features
    # Append the new feature vector to the features list
    bin_features = bin_spatial(feature_image, size=spatial_size)
    f = np.append(bin_features, f)

    # Apply color_hist() to get color histogram features
    hist_features = color_hist(feature_image, nbins=hist_bins)
    # Append the new feature vector to the features list
    f = np.append(hist_features, hog_features)
    return f


# Define a function that takes an image,
# start and stop positions in both x and y,
# window size (x and y dimensions),
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                 xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0] * (xy_overlap[0]))
    ny_buffer = np.int(xy_window[1] * (xy_overlap[1]))
    nx_windows = np.int((xspan - nx_buffer) / nx_pix_per_step)
    ny_windows = np.int((yspan - ny_buffer) / ny_pix_per_step)
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs * nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys * ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list


def predict_vehicle(model='model.pkl', image=None,
                    colorspace=Color.YCrCb, hog_channel="ALL"
                    ):
    svc = joblib.load(model)
    X_scaler = joblib.load("scaler.pkl")
    r = []

    bin_features = []
    wl = []
    t = time.time()
    ## counter=6
    ## boo=image.get_image(Color.BGR).image
    for window in range(64, 144, 24):
        # print("##### window %d" % window)
        # Reduce the sample size because HOG features are slow to compute
        # The quiz evaluator times out after 13s of CPU time

        window_list = slide_window(image.image, xy_window=(window, window), xy_overlap=(0.2, 0.2),
                                   y_start_stop=[400, 656])
        for ((startx, starty), (endx, endy)) in window_list:
            i = image.crop(startx=startx, starty=starty,
                           endx=endx, endy=endy).scale((64, 64))
            bf = features_features_features(i, cspace=colorspace, hog_channel=hog_channel)
            bin_features.append(bf)
        t2 = time.time()
        print(round(t2 - t, 2), 'Seconds to extract HOG features...')
        if not bin_features:
            continue
        wl.extend(window_list)
        ## boo = draw_boxes(boo, window_list, thick=counter, color=(0,0,window*2))
        ## counter -= 2
    ## cv2.imwrite("search_windows.png", boo)

    t = time.time()
    # Create an array stack of feature vectors
    X = np.vstack((bin_features)).astype(np.float64)
    # Fit a per-column scaler
    # X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)
    # Define the labels vector
    # Split up data into randomized training and test sets

    # Check the prediction time for a single sample
    n_predict = len(bin_features)
    p = svc.predict(scaled_X)
    print('My SVC predicts: ', np.count_nonzero(p))
    t2 = time.time()
    print(round(t2 - t, 5), 'Seconds to predict', n_predict, 'labels with SVC')
    for i in range(len(p)):
        if p[i] > 0:
            r.append(wl[i])

    return r


def create_model(colorspace=Color.YCrCb, hog_channel="ALL",
                 cars='train/vehicles/*/*.png', notcars='train/non-vehicles/*/*.png',
                 output='model.pkl',
                 orient=None, pix_per_cell=None, cell_per_block=None):
    # Divide up into cars and notcars
    cars = glob.glob(cars)
    notcars = glob.glob(notcars)

    # Reduce the sample size because HOG features are slow to compute
    # The quiz evaluator times out after 13s of CPU time
    sample_size = len(cars)
    cars = cars[0:sample_size]
    notcars = notcars[0:sample_size]

    t = time.time()
    car_features = extract_features(cars, cspace=colorspace, hog_channel=hog_channel,
                                    orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block)
    notcar_features = extract_features(notcars, cspace=colorspace, hog_channel=hog_channel,
                                       orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block)
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to extract HOG features...')
    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)
    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)
    print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC
    svc = LinearSVC()
    # Check the training time for the SVC
    t = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC

    print(
        "%s %s: Test Accuracy of SVC = %.6f" % (colorspace.name, str(hog_channel), round(svc.score(X_test, y_test), 4)))
    # Check the prediction time for a single sample
    t = time.time()
    n_predict = 10
    print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
    print('For these', n_predict, 'labels: ', y_test[0:n_predict])
    t2 = time.time()
    print(round(t2 - t, 5), 'Seconds to predict', n_predict, 'labels with SVC')
    joblib.dump(svc, output)
    joblib.dump(X_scaler, "scaler.pkl")


### TODO: Tweak these parameters and see how the results change.
# colorspaces = [Color.RGB, Color.HSV, Color.LUV, Color.YUV, Color.YCrCb]
# hog_channels = [0, 1, 2, "ALL"]
# for colorspace, hog_channel in product(colorspaces, hog_channels):
#    print("--- Run with %s %s ---" % (colorspace.name, str(hog_channel)))
#    method_name(colorspace=colorspace, hog_channel=hog_channel)


@click.group(chain=True)
def cli():
    pass


@cli.command()
@click.option('-o', '--output', default="model.pkl",
              help='The output file',
              type=click.Path(exists=False))
@click.option('--cars', default='train/vehicles/*/*.png',
              help='The input image(s)')
@click.option('--notcars', default='train/non-vehicles/*/*.png',
              help='The input image(s)')
def model(output, cars, notcars):
    colorspaces = [Color.YCrCb]
    hog_channels = ["ALL"]
    orients = [10]  # range(8, 12)
    pix_per_cells = [8]  # range(6, 10)
    cell_per_blocks = [2]  # range(1, 4)
    for colorspace, hog_channel, orient, pix_per_cell, cell_per_block in product(colorspaces, hog_channels, orients,
                                                                                 pix_per_cells, cell_per_blocks):
        print("--- Run with %s %s %d %d %d ---" % (
            colorspace.name, str(hog_channel), orient, pix_per_cell, cell_per_block))
        create_model(output=output, cars=cars, notcars=notcars, colorspace=colorspace, hog_channel=hog_channel,
                     orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block)
    sys.exit()


def add_heat(heatmap, bbox_list, factor=1):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += int(5 * factor)

    # Return updated heatmap
    return heatmap  # Iterate through list of bboxes


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


# Here is your draw_boxes function from the previous exercise
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
    # Return the image
    return img


@cli.command()
@click.option('-m', '--model', default="model.pkl",
              help='The model',
              type=click.Path(exists=True))
@click.option('-o', '--output', default=".",
              help='The output directory',
              type=click.Path(exists=True))
@click.option('-i', '--input', multiple=True,
              help='The input image(s)')
def predict(model, output, input):
    images = []
    for i in input:
        images.extend(glob.glob(i))

    for idx, fname in enumerate(images):
        print(fname)
        output_filename = "output_%d_%s" % (idx, os.path.basename(fname))
        output_image = os.path.join(output, output_filename)
        img = FileImage(filename=fname)

        colorspaces = [Color.YCrCb]
        hog_channels = ["ALL"]
        for colorspace, hog_channel in product(colorspaces, hog_channels):
            foo = predict_image(colorspace, hog_channel, img)

            print("Processed image %s to %s" % (fname, output_filename))
        foo = cv2.cvtColor(foo,cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_image, foo)

    sys.exit()


previous_predicted_vehicles = collections.deque(maxlen=5)


def predict_image(colorspace, hog_channel, img, use_previous_heatmap=False, lane_image=None):
    heat = np.zeros_like(img.image[:, :, 0]).astype(np.float)
    w = predict_vehicle(image=img, colorspace=colorspace, hog_channel=hog_channel)
    # Add heat to each box in box list
    if use_previous_heatmap:
        if len(previous_predicted_vehicles) == 5:
            heat = add_heat(heat, previous_predicted_vehicles[0], factor=0.05)
            heat = add_heat(heat, previous_predicted_vehicles[1], factor=0.1)
            heat = add_heat(heat, previous_predicted_vehicles[2], factor=0.3)
            heat = add_heat(heat, previous_predicted_vehicles[3], factor=0.5)
            heat = add_heat(heat, previous_predicted_vehicles[4], factor=0.7)
        previous_predicted_vehicles.append(w)
    heat = add_heat(heat, w, factor=1)
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, 8)
    # Visualize the heatmap when displaying
    labels = label(heat)
    heatmap = np.clip(heat, 0, 255).astype(np.uint8)
    # foo = image_model.call(img)
    # foo = draw_boxes(img.get_image(Color.BGR).image, w)
    if lane_image:
        image = lane_image
    else:
        image = img.get_image(Color.RGB)
    foo = draw_labeled_bboxes(image.image, labels)
    # heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_GRAY2RGB)

    foo = cv2.addWeighted(foo, 1.0, heatmap, 1.0, 0)
    foo = cv2.addWeighted(foo, 1.0, heatmap, 1.0, 0)
    foo = cv2.addWeighted(foo, 1.0, heatmap, 1.0, 0)
    foo = cv2.addWeighted(foo, 1.0, heatmap, 1.0, 0)
    foo = cv2.addWeighted(foo, 1.0, heatmap, 1.0, 0)
    cv2.putText(foo, "%d" % np.max(heat), (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))

    # foo = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)
    return foo


def process_video_image(image):
    """
    Helper for the video pipeline.
    :param image: The unprocessed image from the video.
    :return: The resulting image
    """
    img = Image(image=image, color=Color.RGB)
    result = video_model.call(img).image
    lane_img = Image(image=result, color=Color.RGB)
    result = predict_image(Color.YCrCb, "ALL", img, use_previous_heatmap=True, lane_image=lane_img)
    return result


@cli.command()
@click.option('-o', '--output', default=".",
              help='The output directory',
              type=click.Path(exists=True))
@click.option('-i', '--input', multiple=True,
              help='The input image(s)')
def videos(output, input):
    global video_model
    video_model = models['full']()

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


if __name__ == '__main__':
    cli()
