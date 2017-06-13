import argparse
import numpy as np
import cv2
from collections import namedtuple

def merge(img1, img2):
  rows,cols,channels = img2.shape
  roi = img1[0:rows, 0:cols ]
  # Now create a mask of logo and create its inverse mask also
  img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
  ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
  mask_inv = cv2.bitwise_not(mask)
  # Now black-out the area of logo in ROI
  img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
  # Take only region of logo from logo image.
  img2_fg = cv2.bitwise_and(img2,img2,mask = mask)
  # Put logo in ROI and modify the main image
  dst = cv2.add(img1_bg,img2_fg)
  img1[0:rows, 0:cols ] = dst
  return img1


def auto_canny(image, sigma=0.33):
  # compute the median of the single channel pixel intensities
  v = np.median(image)

  # apply automatic Canny edge detection using the computed median
  lower = int(max(0, (1.0 - sigma) * v))
  upper = int(min(255, (1.0 + sigma) * v))
  edged = cv2.Canny(image, lower, upper)

  # return the edged image
  return edged

SizeX = 640
SizeY = 480
affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR
def deskew(img):
  m = cv2.moments(img)
  if abs(m['mu02']) < 1e-2:
      return img.copy()
  skew = m['mu11']/m['mu02']
  M = np.float32([[1, skew, -0.5*SizeX*skew], [0, 1, 0]])
  img = cv2.warpAffine(img,M,(SizeX, SizeY),flags=affine_flags)
  return img


def make_edges(frame):
  # Blur image to make it easier to detect objects
  #frame = cv2.medianBlur(frame, 3)
  # to gray
  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  frame = cv2.bilateralFilter(frame, 11, 17, 17)
  #frame = deskew(frame)
  #cv2.imshow('gray', frame)
  #edged = cv2.Canny(frame, 30, 200)
  #frame = auto_canny(frame)
  #cv2.imshow('blur_image_edged_auto', edged)
  return frame


def make_warp(frame):
  height, width, _ = frame.shape
  pts_src = np.array([[width / 2 - 35, height / 2], [width / 2 + 35 , height / 2], [0, height], [width, height]])
  pts_dst = np.array([[0, 0], [width, 0], [0, height], [width, height]])
  # Calculate Homography
  h, status = cv2.findHomography(pts_src, pts_dst)
  # Warp source image to destination based on homography
  return cv2.warpPerspective(frame, h, (width, height))


def find_blobs(frame):
  # Set up the detector with default parameters.
  detector = cv2.SimpleBlobDetector_create()
  # Detect blobs.
  keypoints = detector.detect(frame)
  # Draw detected blobs as red circles.
  # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
  return cv2.drawKeypoints(frame, keypoints, np.array([]), (0, 0, 255),
                                        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

def draw_sift(frame):
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  sift = cv2.xfeatures2d.SIFT_create()
  kp = sift.detect(gray, None)
  return cv2.drawKeypoints(gray, kp, frame)


def region_of_interest(img, vertices):
    """ Applies an image mask. Everything outside of the region defined by vertices will be set to black.
    Vertices should be in the form Sequence[np.array[Tuple[int, int]]] but we can't document this with mypy. """

    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    # Vertices is a list containing an np.array of vertices. This is not obvious.
    # It needs to look like: [np.array(vertex1, vertex2, vertex3)]
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)

    if lines is None:
      return []
    # Use our tuple object to make line calculations friendlier
    # A line is given to use in the format [[x1,y1,x2,y2]]
    Line = namedtuple("Line", "x1 y1 x2 y2")
    return [Line(*line[0]) for line in lines]

def find_lane(frame):
  grayscale_img = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
  # Choose a Region Of Interest
  height, width, _ = frame.shape
  # bottom_left_corner = (0, height - 100)
  # bottom_right_corner = (width, height - 100)
  # center = (int(width / 2 - 50), int(height/2 + 50))
  bottom_left_corner = (33, 357)
  bottom_right_corner = (610, 361)
  center_left = (290, 225)
  center_right = (337, 225)
  region = [np.array([bottom_left_corner,center_left, center_right, bottom_right_corner])]
  region_selected_image = region_of_interest(grayscale_img, region)
  #cv2.imshow('region_selected_image', region_selected_image)
  kernel_size = 3
  blurred =  cv2.GaussianBlur(region_selected_image, (kernel_size, kernel_size), 0)
  cv2.imshow('blurred_region_selected_image', blurred)
  low_threshold = 100
  high_threshold = 300
  canny_transformed = cv2.Canny(blurred, low_threshold, high_threshold)
  cv2.imshow('canny_transformed', canny_transformed)
  RHO = 2
  THETA = np.pi / 180
  HOUGH_LINE_THRESHOLD = 50
  MIN_LINE_LEN = 20
  MAX_LINE_GAP = 150
  lines =  hough_lines(canny_transformed, RHO, THETA, HOUGH_LINE_THRESHOLD, MIN_LINE_LEN, MAX_LINE_GAP)
  color=(255, 0, 0)
  thickness=10
  lanes_img = np.zeros((height, width, 3), dtype=np.uint8)
  for lane in lines:
    cv2.line(lanes_img, (lane.x1, lane.y1), (lane.x2, lane.y2), color, thickness)
  lane_annotated = cv2.addWeighted(frame, .8, lanes_img, 1, 0)
  return lane_annotated


def main():
  parser = argparse.ArgumentParser(description='Process some integers.')
  parser.add_argument("-i", "--input_video_path", dest = 'input_video_path', required = True)
  parser.add_argument("-t", "--input_txt_path", dest = 'input_txt_path', required = False)
  parser.add_argument("-b", "--batch_read", dest = 'batch_read', required = False)
  args = parser.parse_args()

  mphs = []
  if args.input_txt_path:
    with open(args.input_txt_path) as f:
      mphs = [line for line in f]

  def show_mph(img, mphs, i):
    cv2.line(img, (0,240), (640,240), (255,255,255), 2)
    cv2.putText(img = img,
              text = mphs[c],
              org = (0, 400),
              fontFace = cv2.FONT_HERSHEY_DUPLEX,
              fontScale = 3,
              color = (255,0,0),
              thickness = 2)

  def on_mouse(event, x, y, flag, param):
    if (event == cv2.EVENT_LBUTTONUP):
      print x, y

  cap = cv2.VideoCapture(args.input_video_path)
  fps = cap.get(cv2.CAP_PROP_FPS)
  print "Frames per second: {0}".format(fps)
  c = 0
  while(cap.isOpened()):
    if args.batch_read:
      # Read fps frames
      frames = [x[1] for x in [cap.read() for i in range(int(fps))] if x[0]]
      if len(frames) > 0:
        edged_frames = [make_edges(frame) for i, frame in enumerate(frames)]
        c += len(frames)
        img = sum(edged_frames)
        show_mph(img, mphs, c)
        cv2.imshow('acc_edged', img)
    else:
      _, frame = cap.read()
      #img = make_edges(frame)
      #img = draw_sift(frame)
      #img = find_blobs(frame)
      #img = make_warp(frame)
      img = find_lane(frame)
      show_mph(img, mphs, c)
      c += 1
      cv2.imshow('edged', img)
      cv2.imshow('orginal', frame)
      #cv2.setMouseCallback("orginal", on_mouse, param=())
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  cap.release()
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
