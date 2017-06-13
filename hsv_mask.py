
import os
import argparse
import numpy as np
import cv2

def create_hue_mask(image, hsv_image, lower_color, upper_color):
    lower = np.array(lower_color, np.uint8)
    upper = np.array(upper_color, np.uint8)
 
    # Create a mask from the colors
    mask = cv2.inRange(hsv_image, lower, upper)
    output_image = cv2.bitwise_and(image, image, mask = mask)
    return output_image, mask

def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("-i", "--image_path", dest = 'image_path', required = True)

    args = parser.parse_args()

    # Load image
    image = cv2.imread(args.image_path)

    # Blur image to make it easier to detect objects
    blur_image = cv2.medianBlur(image, 3)
    # Convert to HSV in order to
    hsv_image = cv2.cvtColor(blur_image, cv2.COLOR_BGR2HSV)

    cv2.namedWindow('hsv_mask', 0)
    global middle_hue, low_saturation, low_value, delta_hue
    middle_hue = 119
    low_saturation = 28
    low_value = 105
    delta_hue = 13

    def show():
      global middle_hue, low_saturation, low_value, delta_hue
      out_image, lower_blue_hue_mask = create_hue_mask(blur_image, hsv_image,
        [middle_hue - delta_hue, low_saturation, low_value],
        [middle_hue + delta_hue, 255, 255])
      #cv2.imshow('out_image', out_image)
      #cv2.imshow('lower_blue_hue_mask', lower_blue_hue_mask)
      # Blur the final image to reduce noise from image
      #gaussian_blur = cv2.GaussianBlur(out_image, (9, 9), 2, 2)
      #cv2.imshow('gaussian_blur', gaussian_blur)

      gray = cv2.cvtColor(blur_image, cv2.COLOR_BGR2GRAY)
      gray = cv2.bilateralFilter(gray, 11, 17, 17)
      cv2.imshow('gray', gray)
      edged = cv2.Canny(gray, 30, 200)
      cv2.imshow('blur_image_edged', edged)

      gray = cv2.cvtColor(out_image, cv2.COLOR_BGR2GRAY)
      gray = cv2.bilateralFilter(gray, 11, 17, 17)
      cv2.imshow('hsv_mask_gray', gray)
      edged = cv2.Canny(gray, 30, 200)
      cv2.imshow('hsv_masked_edged', edged)

      print(len(edged))
      (_, cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
      cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
      screenCnt = None
      for c in cnts:
      	# approximate the contour
      	peri = cv2.arcLength(c, True)
      	approx = cv2.approxPolyDP(c, 0.02 * peri, True)

      	# if our approximated contour has four points, then
      	# we can assume that we have found our screen
      	if len(approx) == 4:
      		screenCnt = approx
      		break
      cv2.drawContours(image, cnts, -1, (0, 255, 0), 3)
      cv2.imshow("contoured_image", image)

    def update_middle_hue(val):
      global middle_hue, low_saturation, low_value, delta_hue
      middle_hue = val
      print(middle_hue, low_saturation, low_value, delta_hue)
      show()

    def update_delta_hue(val):
      global middle_hue, low_saturation, low_value, delta_hue
      delta_hue = val
      print(middle_hue, low_saturation, low_value, delta_hue)
      show()

    def update_lower_saturation(val):
      global middle_hue, low_saturation, low_value, delta_hue
      low_saturation = val
      print(middle_hue, low_saturation, low_value, delta_hue)
      show()

    def update_lower_value(val):
      global middle_hue, low_saturation, low_value, delta_hue
      low_value = val
      print(middle_hue, low_saturation, low_value, delta_hue)
      show()

    cv2.createTrackbar('Middle Hue', 'hsv_mask', middle_hue, 180, update_middle_hue)
    cv2.createTrackbar('Low Saturation', 'hsv_mask', low_saturation, 255, update_lower_saturation)
    cv2.createTrackbar('Low Value', 'hsv_mask', low_value, 255, update_lower_value)
    cv2.createTrackbar('Delta Hue', 'hsv_mask', delta_hue, 255, update_delta_hue)
    show()

    while True:
      ch = cv2.waitKey(1)
      if ch == 27:
        break

    # Find circles in the image
    circles = cv2.HoughCircles(image_gray, cv2.HOUGH_GRADIENT, 1.2, 100)

#        # Draw the circles on the original image
#        circles = np.round(circles[0, :]).astype("int")
#        for (center_x, center_y, radius) in circles:
#            cv2.circle(image, (center_x, center_y), radius, (0, 255, 0), 4)

if __name__ == '__main__':
    main()
