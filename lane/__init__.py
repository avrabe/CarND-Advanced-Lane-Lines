import attr
import cv2
import numpy as np
from haikunator import Haikunator

from lane.image import Color, ColorChannel, ImageChannel, Image

haikunator = Haikunator()


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
            base_image = cv2.addWeighted(base_image, 1.0, bkg, -.70, 0)
            base_image = cv2.addWeighted(base_image, 1.0, overlay, 1.0, 0)
        if 'lane_lines' in meta.keys():
            side_pos = meta['lane_lines']['side_pos']
            y0, dy = 50, 50
            for i, line in enumerate(side_pos.split('\n')):
                y = y0 + i * dy
                cv2.putText(base_image, line, (50, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
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
    smooth = attr.ib(default=True)
    ym_per_pix = attr.ib(default=30 / 720)  # meters per pixel in y dimension
    xm_per_pix = attr.ib(default=3.7 / 614)  # meters per pixel in x dimension
    return_binary_warped = attr.ib(default=False)

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

        new_left_fit = np.polyfit(lefty, leftx, 2)
        new_right_fit = np.polyfit(righty, rightx, 2)

        self.left_fit = (self.left_fit + new_left_fit) / 2 if self.smooth else new_left_fit
        self.right_fit = (self.right_fit + new_right_fit) / 2 if self.smooth else new_right_fit
        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        left_fitx = self.left_fit[0] * ploty ** 2 + self.left_fit[1] * ploty + self.left_fit[2]
        right_fitx = self.right_fit[0] * ploty ** 2 + self.right_fit[1] * ploty + self.right_fit[2]

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

        self.leftx = leftx
        self.rightx = rightx
        # Fit a second order polynomial to each
        self.left_fit = np.polyfit(lefty, leftx, 2)
        self.right_fit = np.polyfit(righty, rightx, 2)

        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        return self.__plot_lane_line(out_img)
        # return self.__one_eyed_search(binary_warped)

    def __measure_curvature(self, ploty, center_fitx):
        y_eval = np.max(ploty)
        center_fit_cr = np.polyfit(ploty * self.ym_per_pix, center_fitx * self.xm_per_pix, 2)

        # print(center_fit_cr)
        # Calculate the new radii of curvature
        center_curverad = ((1 + (
            2 * center_fit_cr[0] * y_eval * self.ym_per_pix + center_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * center_fit_cr[0])

        # print(center_curverad)
        # Now our radius of curvature is in meters
        return "Radius of curvature: %9.2fm" % (center_curverad)

        # Example values: 632.1 m    626.2 m

    def __plot_lane_line(self, binary_warped):
        # Generate x and y values for plotting
        out_img = np.zeros_like(binary_warped)
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        left_fitx = self.left_fit[0] * ploty ** 2 + self.left_fit[1] * ploty + self.left_fit[2]
        right_fitx = self.right_fit[0] * ploty ** 2 + self.right_fit[1] * ploty + self.right_fit[2]

        center_fitx = left_fitx + (left_fitx + right_fitx) / 2

        camera_center = (left_fitx[-1] + right_fitx[-1]) / 2
        center_diff = float((camera_center - binary_warped.shape[1] / 2) * self.xm_per_pix)
        side_pos = 'left'
        if center_diff <= 0:
            side_pos = 'right'
        side_pos = "%.2fm to the %s\n%s" % \
                   (center_diff, side_pos,
                    self.__measure_curvature(ploty, center_fitx))
        margin = 20

        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        center_line_window1 = np.array(
            [np.transpose(np.vstack([left_fitx + (right_fitx - left_fitx) / 2 - margin / 2, ploty]))])
        center_line_window2 = np.array(
            [np.flipud(np.transpose(np.vstack([left_fitx + (right_fitx - left_fitx) / 2 + margin / 2, ploty])))])
        center_line_pts = np.hstack((center_line_window1, center_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))
        complete_line_pts = np.hstack((left_line_window1, right_line_window2))

        for i in [out_img]:
            cv2.fillPoly(i, np.int_([complete_line_pts]), (0xbc, 0xe3, 0xff))
            cv2.fillPoly(i, np.int_([center_line_pts]), (0x40, 0x62, 0xbb))
            cv2.fillPoly(i, np.int_([left_line_pts]), (00, 0x48, 0x7c))
            cv2.fillPoly(i, np.int_([right_line_pts]), (0x3e, 0x66, 0x80))

        out_img = out_img if not self.return_binary_warped else binary_warped
        return side_pos, out_img, binary_warped
