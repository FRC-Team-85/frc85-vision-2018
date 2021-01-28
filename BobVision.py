import libjevois as jevois
import cv2
import numpy as np
import math
from enum import Enum

## Simple example of image processing using OpenCV in Python on JeVois
#
# This module is here for you to experiment with Python OpenCV on JeVois.
#
# By default, we get the next video frame from the camera as an OpenCV BGR (color) image named 'inimg'.
# We then apply some image processing to it to create an output BGR image named 'outimg'.
# We finally add some text drawings to outimg and send it to host over USB.
#
# See http://jevois.org/tutorials for tutorials on getting started with programming JeVois in Python without having
# to install any development software on your host computer.
#
# @author Laurent Itti
# 
# @videomapping YUYV 352 288 30.0 YUYV 352 288 30.0 JeVois PythonSandbox
# @email itti\@usc.edu
# @address University of Southern California, HNB-07A, 3641 Watt Way, Los Angeles, CA 90089-2520, USA
# @copyright Copyright (C) 2017 by Laurent Itti, iLab and the University of Southern California
# @mainurl http://jevois.org
# @supporturl http://jevois.org/doc
# @otherurl http://iLab.usc.edu
# @license GPL v3
# @distribution Unrestricted
# @restrictions None
# @ingroup modules

class BobVision:
    # ###################################################################################################
    ## Constructor
    def __init__(self):
        # Instantiate a JeVois Timer to measure our processing framerate:
        self.timer = jevois.Timer("sandbox", 100, jevois.LOG_INFO)
        
        self.__hsv_threshold_hue = [46.94244604316547, 180.0]
        self.__hsv_threshold_saturation = [105.48561151079136, 255.0]
        self.__hsv_threshold_value = [52.74280575539568, 229.24242424242425]

        self.hsv_threshold_output = None

        self.__blur_input = self.hsv_threshold_output
        self.__blur_type = BlurType.Median_Filter
        self.__blur_radius = 15.315315315315313

        self.blur_output = None


        self.__mask_mask = self.blur_output

        self.mask_output = None

        self.__desaturate_input = self.mask_output

        self.desaturate_output = None

        self.__find_contours_input = self.desaturate_output
        self.__find_contours_external_only = False

        self.find_contours_output = None

        self.__filter_contours_contours = self.find_contours_output
        self.__filter_contours_min_area = 2000.0
        self.__filter_contours_min_perimeter = 200.0
        self.__filter_contours_min_width = 50.0
        self.__filter_contours_max_width = 1000.0
        self.__filter_contours_min_height = 50.0
        self.__filter_contours_max_height = 1000.0
        self.__filter_contours_solidity = [73.74100719424462, 100.0]
        self.__filter_contours_max_vertices = 1000000.0
        self.__filter_contours_min_vertices = 0.0
        self.__filter_contours_min_ratio = 0.75
        self.__filter_contours_max_ratio = 3.25

        self.filter_contours_output = None

        
    # ###################################################################################################
    ## Process function with USB output
    def process(self, inframe, outframe):
        self.timer.start()
      
        #rawinimg = inframe.get()
        #inimg = jevois.convertToCvBGR(rawinimg)
        
        #rawinimg.done()

        inimg = inframe.getCvBGR()

        h, w, chans = inimg.shape
        #outimg = outframe.get()

        #outimg.require("output", w, h, jevois.V4L2_PIX_FMT_YUYV)
        #jevois.paste(rawinimg, outimg, 0, 0)

        self.__hsv_threshold_input = inimg
        (self.hsv_threshold_output) = self.__hsv_threshold(self.__hsv_threshold_input, self.__hsv_threshold_hue, self.__hsv_threshold_saturation, self.__hsv_threshold_value)

        # Step Blur0:
        self.__blur_input = self.hsv_threshold_output
        (self.blur_output) = self.__blur(self.__blur_input, self.__blur_type, self.__blur_radius)

        # Step Mask0:
        self.__mask_input = inimg
        self.__mask_mask = self.blur_output
        (self.mask_output) = self.__mask(self.__mask_input, self.__mask_mask)

        # Step Desaturate0:
        self.__desaturate_input = self.mask_output
        (self.desaturate_output) = self.__desaturate(self.__desaturate_input)

        # Step Find_Contours0:
        self.__find_contours_input = self.desaturate_output
        (self.find_contours_output) = self.__find_contours(self.__find_contours_input, self.__find_contours_external_only)

        # Step Filter_Contours0:
        self.__filter_contours_contours = self.find_contours_output
        (self.filter_contours_output) = self.__filter_contours(self.__filter_contours_contours, self.__filter_contours_min_area, self.__filter_contours_min_perimeter, self.__filter_contours_min_width, self.__filter_contours_max_width, self.__filter_contours_min_height, self.__filter_contours_max_height, self.__filter_contours_solidity, self.__filter_contours_max_vertices, self.__filter_contours_min_vertices, self.__filter_contours_min_ratio, self.__filter_contours_max_ratio)

        length = len(self.filter_contours_output)

        for i in range(0, length):
            x, y, w, h = cv2.boundingRect(self.filter_contours_output[i])
            jevois.sendSerial("{}_{}: {}".format(i, "centerX", (x + w / 2)))
            if (i == 0):
              #jevois.drawRect(rawinimg, x, y, w, h, jevois.YUYV.MedGreen)
              cv2.rectangle(inimg, (x, y), (x + w, y + h), (0, 255, 0))
            else:
              #jevois.drawRect(rawinimg, x, y, w, h, jevois.YUYV.MedPink)
              cv2.rectangle(inimg, (x, y), (x + w, y + h), (0, 0, 255))


        #jevois.writeText(rawinimg, "BobVision", 3, 20, jevois.YUYV.White, jevois.Font.Font6x10)
        cv2.putText(inimg, "BobVision", (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),
                    1, cv2.LINE_AA)

        #jevois.writeText(rawinimg, "Input: Width: {} Height: {} FPS: {}".format(rawinimg.width, rawinimg.height, rawinimg.fps), 3, 40, jevois.YUYV.White, jevois.Font.Font6x10)

        #newout = jevois.convertToCvBGR(rawinimg)

        #jevois.writeText(newout, "Output: Width: {} Height: {} FPS: {}".format(outimg.width, outimg.height, outimg.fps), 3, 50, jevois.YUYV.White, jevois.Font.Font6x10)
 
        outframe.sendCvBGR(inimg, 50)

    
    @staticmethod
    def __hsv_threshold(input, hue, sat, val):
        """Segment an image based on hue, saturation, and value ranges.
        Args:
            input: A BGR numpy.ndarray.
            hue: A list of two numbers the are the min and max hue.
            sat: A list of two numbers the are the min and max saturation.
            lum: A list of two numbers the are the min and max value.
        Returns:
            A black and white numpy.ndarray.
        """
        out = cv2.cvtColor(input, cv2.COLOR_BGR2HSV)
        return cv2.inRange(out, (hue[0], sat[0], val[0]),  (hue[1], sat[1], val[1]))

    @staticmethod
    def __blur(src, type, radius):
        """Softens an image using one of several filters.
        Args:
            src: The source mat (numpy.ndarray).
            type: The blurType to perform represented as an int.
            radius: The radius for the blur as a float.
        Returns:
            A numpy.ndarray that has been blurred.
        """
        if(type is BlurType.Box_Blur):
            ksize = int(2 * round(radius) + 1)
            return cv2.blur(src, (ksize, ksize))
        elif(type is BlurType.Gaussian_Blur):
            ksize = int(6 * round(radius) + 1)
            return cv2.GaussianBlur(src, (ksize, ksize), round(radius))
        elif(type is BlurType.Median_Filter):
            ksize = int(2 * round(radius) + 1)
            return cv2.medianBlur(src, ksize)
        else:
            return cv2.bilateralFilter(src, -1, round(radius), round(radius))

    @staticmethod
    def __mask(input, mask):
        """Filter out an area of an image using a binary mask.
        Args:
            input: A three channel numpy.ndarray.
            mask: A black and white numpy.ndarray.
        Returns:
            A three channel numpy.ndarray.
        """
        return cv2.bitwise_and(input, input, mask=mask)

    @staticmethod
    def __desaturate(src):
        """Converts a color image into shades of gray.
        Args:
            src: A color numpy.ndarray.
        Returns:
            A gray scale numpy.ndarray.
        """
        (a, b, channels) = src.shape
        if(channels == 1):
            return numpy.copy(src)
        elif(channels == 3):
            return cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        elif(channels == 4):
            return cv2.cvtColor(src, cv2.COLOR_BGRA2GRAY)
        else:
            raise Exception("Input to desaturate must have 1, 3 or 4 channels") 

    @staticmethod
    def __find_contours(input, external_only):
        """Sets the values of pixels in a binary image to their distance to the nearest black pixel.
        Args:
            input: A numpy.ndarray.
            external_only: A boolean. If true only external contours are found.
        Return:
            A list of numpy.ndarray where each one represents a contour.
        """
        if(external_only):
            mode = cv2.RETR_EXTERNAL
        else:
            mode = cv2.RETR_LIST
        method = cv2.CHAIN_APPROX_SIMPLE
        im2, contours, hierarchy =cv2.findContours(input, mode=mode, method=method)
        return contours

    @staticmethod
    def __filter_contours(input_contours, min_area, min_perimeter, min_width, max_width,
                        min_height, max_height, solidity, max_vertex_count, min_vertex_count,
                        min_ratio, max_ratio):
        """Filters out contours that do not meet certain criteria.
        Args:
            input_contours: Contours as a list of numpy.ndarray.
            min_area: The minimum area of a contour that will be kept.
            min_perimeter: The minimum perimeter of a contour that will be kept.
            min_width: Minimum width of a contour.
            max_width: MaxWidth maximum width.
            min_height: Minimum height.
            max_height: Maximimum height.
            solidity: The minimum and maximum solidity of a contour.
            min_vertex_count: Minimum vertex Count of the contours.
            max_vertex_count: Maximum vertex Count.
            min_ratio: Minimum ratio of width to height.
            max_ratio: Maximum ratio of width to height.
        Returns:
            Contours as a list of numpy.ndarray.
        """
        output = []
        for contour in input_contours:
            x,y,w,h = cv2.boundingRect(contour)
            if (w < min_width or w > max_width):
                continue
            if (h < min_height or h > max_height):
                continue
            area = cv2.contourArea(contour)
            if (area < min_area):
                continue
            if (cv2.arcLength(contour, True) < min_perimeter):
                continue
            hull = cv2.convexHull(contour)
            solid = 100 * area / cv2.contourArea(hull)
            if (solid < solidity[0] or solid > solidity[1]):
                continue
            if (len(contour) < min_vertex_count or len(contour) > max_vertex_count):
                continue
            ratio = (float)(w) / h
            if (ratio < min_ratio or ratio > max_ratio):
                continue
            output.append(contour)
        return output


BlurType = Enum('BlurType', 'Box_Blur Gaussian_Blur Median_Filter Bilateral_Filter')
