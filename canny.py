import argparse
import numpy as np
import cv2
from collections import namedtuple
import os
import datetime as dt
from skimage.measure import compare_ssim as ssim

def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype('float') - imageB.astype('float')) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err


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


def deskew(img):
    height, width, _ = img.shape
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
      return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*width*skew], [0, 1, 0]])
    img = cv2.warpAffine(img,M,(width, height),flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR)
    return img


def make_edges(frame):
    # Blur image to make it easier to detect objects
    # frame = cv2.medianBlur(frame, 3)
    # to gray
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.bilateralFilter(frame, 11, 17, 17)
    # frame = deskew(frame)
    #cv2.imshow('gray', frame)
    #edged = cv2.Canny(frame, 30, 200)
    frame = auto_canny(frame)
    #cv2.imshow('blur_image_edged_auto', edged)
    return frame


def make_warp(frame, x1, y1, x2, y2, x3, y3, x4, y4):
    height, width, _ = frame.shape
    pts_src = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    pts_dst = np.array([[0, 0], [width, 0], [0, height], [width, height]])
    # Calculate Homography
    h, status = cv2.findHomography(pts_src, pts_dst)
    # Warp source image to destination based on homography
    return cv2.warpPerspective(frame, h, (width, height))

def perspective_transform(frame, x1, y1, x2, y2, x3, y3, x4, y4):
    height, width, _ = frame.shape
    pts1 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(frame, M, (width, height))

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
    if (len(frame.shape) > 2):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(frame, None)
    return cv2.drawKeypoints(frame, kp, frame)


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


def find_roi(frame, bottom_left_corner, bottom_right_corner, top_left, top_right):
    # if (len(frame.shape) > 2):
    #     frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # Sharpen using kernel
    #kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    #frame = cv2.filter2D(frame, -1, kernel)
    #frame = cv2.equalizeHist(frame)
    #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    #frame = clahe.apply(frame)
    #frame = auto_canny(frame)
    # Choose a Region Of Interest
    region = [np.array([bottom_left_corner,top_left, top_right, bottom_right_corner])]
    return region_of_interest(frame, region)

def find_lane(frame, bottom_left_corner, bottom_right_corner, top_left, top_right):
    region_selected_image = find_roi(frame, bottom_left_corner, bottom_right_corner, top_left, top_right)
    #cv2.imshow('region_selected_image', region_selected_image)
    kernel_size = 3
    blurred =  cv2.GaussianBlur(region_selected_image, (kernel_size, kernel_size), 0)
    #cv2.imshow('blurred_region_selected_image', blurred)
    low_threshold = 100
    high_threshold = 300
    canny_transformed = cv2.Canny(blurred, low_threshold, high_threshold)
    #cv2.imshow('canny_transformed', canny_transformed)
    RHO = 2
    THETA = np.pi / 180
    HOUGH_LINE_THRESHOLD = 50
    MIN_LINE_LEN = 20
    MAX_LINE_GAP = 150
    lines =  hough_lines(canny_transformed, RHO, THETA, HOUGH_LINE_THRESHOLD, MIN_LINE_LEN, MAX_LINE_GAP)
    color=(255, 0, 0)
    thickness=10
    height, width, _ = frame.shape
    lanes_img = np.zeros((height, width, 3), dtype=np.uint8)
    for lane in lines:
        cv2.line(lanes_img, (lane.x1, lane.y1), (lane.x2, lane.y2), color, thickness)
    lane_annotated = cv2.addWeighted(frame, .8, lanes_img, 1, 0)
    return lane_annotated


def verify_keypoint_move_direction(width, pt1, pt2):
    (x1,y1) = pt1
    (x2,y2) = pt2
    if y1 > y2:
        return False
    if abs(x1 - x2) > 100:
        return False
    return True
    # if x1 < width / 2:
    #   return x2 < x1
    # else:
    #   return x2 > x1

def bf_match(img1, img2, kp1, des1, kp2, des2, drawToImage = False):
    if des1 is None:
        print("des1 is None")
        cv2.imwrite("tmp/des1.png", img1)
        quit(-1)
        #return False, None, 0, None, None
    if des2 is None:
        print("des2 is None")
        cv2.imwrite("tmp/des2.png", img1)
        quit(-1)
        #return False, None, 0, None, None
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1, des2)
    # Sort them in the order of their distance.
    #matches = sorted(matches, key = lambda x:x.distance)
    height, width = img1.shape[:2]
    good = [match for match in matches if verify_keypoint_move_direction(width, kp1[match.queryIdx].pt, kp2[match.trainIdx].pt)]
    if len(good) is 0:
        print("good is None")
        img1 = cv2.drawKeypoints(img1, kp1, 0, color=(0, 255, 0), flags=0)
        cv2.imwrite("tmp/good_none1.png", img1)
        img2 = cv2.drawKeypoints(img2, kp2, 0, color=(0, 255, 0), flags=0)
        cv2.imwrite("tmp/good_none2.png", img2)
        match_img = cv2.drawMatches(img1,kp1,img2,kp2, matches, 0, flags=2)
        cv2.imwrite("tmp/good_none_match.png", match_img)
        quit(-1)
        #return False, None, 0, None, None
    movement_y_avg = sum([(kp2[match.trainIdx].pt[1] - kp1[match.queryIdx].pt[1]) for match in good]) / len(good)
    #print(movement_y)
    out = None
    if drawToImage:
        out = cv2.drawMatches(img1,kp1,img2,kp2, good, 0, flags=2)
        cv2.putText(img=out, text=str(len(good)), org=(width * 2 - 100, 50), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=2,
                    color=(0, 0, 255), thickness=2)
        cv2.imshow('out', out)
    return True, out, movement_y_avg, kp2, des2, len(good)

def sift_knn_match(img1, img2, kp1, des1, kp2, des2, drawToImage = False):
    if des1 is None:
        print("des1 is None")
        cv2.imwrite("tmp/des1.png", img1)
        quit(-1)
        #return False, None, 0, None, None
    if des2 is None:
        print("des2 is None")
        cv2.imwrite("tmp/des2.png", img1)
        quit(-1)
        #return False, None, 0, None, None
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    # Apply ratio test
    good = []
    for m in matches:
        if len(m) == 2 and m[0].distance < 0.75 * m[1].distance:
            good.append(m[0])
    height, width = img1.shape
    good = [match for match in good if verify_keypoint_move_direction(width, kp1[match.queryIdx].pt, kp2[match.trainIdx].pt)]
    if len(good) is 0:
        print("good is None")
        img1 = cv2.drawKeypoints(img1, kp1, 0, color=(0, 255, 0), flags=0)
        cv2.imwrite("tmp/good_none1.png", img1)
        img2 = cv2.drawKeypoints(img2, kp2, 0, color=(0, 255, 0), flags=0)
        cv2.imwrite("tmp/good_none2.png", img2)
        match_img = cv2.drawMatches(img1,kp1,img2,kp2, matches, 0, flags=2)
        cv2.imwrite("tmp/good_none_match.png", match_img)
        quit(-1)
        #return False, None, 0, None, None
    movement_y_avg = sum([(kp2[match.trainIdx].pt[1] - kp1[match.queryIdx].pt[1]) for match in good]) / len(matches)
    #print(movement_y)
    out = None
    if drawToImage:
        out = cv2.drawMatches(img1,kp1,img2,kp2, good, 0, flags=2)
        cv2.putText(img=out, text=str(len(good)), org=(width * 2 - 100, 50), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=2,
                    color=(0, 0, 255), thickness=2)
        cv2.imshow('out', out)
    return True, out, movement_y_avg, kp2, des2, len(good)

def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("-i", "--input_video_path", dest = 'input_video_path', required = True)
    parser.add_argument("-t", "--input_txt_path", dest = 'input_txt_path', required = False)
    parser.add_argument("-b", "--batch_read", dest = 'batch_read', required = False)
    parser.add_argument("-f", "--feature_algorithm", dest = 'feature_algorithm', default='orb', required = False)
    args = parser.parse_args()

    mphs = []
    if args.input_txt_path:
        with open(args.input_txt_path) as f:
            mphs = [line for line in f]

    def on_mouse(event, x, y, flag, param):
        if (event == cv2.EVENT_LBUTTONUP):
            print x, y

    cap = cv2.VideoCapture(args.input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print "Frames per second: {0}".format(fps)
    c = 0
    y1 = 225
    y2 = 361
    x1 = 33
    x2 = 610
    bottom_left_corner = (x1, y2)
    bottom_right_corner = (x2, y2)
    top_left = (290, y1)
    top_right = (337, y1)
    if args.batch_read:
        # Read fps frames
        initial_frames = [x[1] for x in [cap.read() for i in range(int(fps * 5))] if x[0]]
        imgs = [find_roi(frame, bottom_left_corner, bottom_right_corner, top_left, top_right) for frame in initial_frames]
        # Crop to roi
        imgs = [img[y1:y2, x1:x2] for img in imgs]
        crop_height = y2 - y1
        crop_width = x2 - x1
        # sub_imgs width: 577 , height: 50 * len(imgs)
        #sub_imgs = np.concatenate([img[(crop_height - 50):crop_height, 0:crop_width] for img in imgs], axis=0)
        sub_imgs = np.concatenate([img for img in imgs], axis=0)
        cv2.imwrite("tmp/" + os.path.basename(args.input_video_path) + '.subs.png', sub_imgs)
        if args.feature_algorithm == 'orb':
          print 'Use ORB'
          feature_detector = cv2.ORB_create()
          feature_matcher = bf_match
        elif args.feature_algorithm == 'sift':
          print 'Use SIFT'
          feature_detector = cv2.xfeatures2d.SIFT_create()
          feature_matcher = sift_knn_match
        else:
          print 'Invalid feature detect algorithm'
          return
        match_results = []
        kps2 = []
        for i, img in enumerate(imgs[:-1]):
            kp1, des1 = feature_detector.detectAndCompute(img, None)
            kp2, des2 = feature_detector.detectAndCompute(imgs[i + 1], None)
            mr = feature_matcher(img, imgs[i + 1], kp1, des1, kp2, des2, True)
            if mr[0]:
                cv2.putText(img=mr[1], text=str(i + 1), org=(30, 50), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=2, color=(255, 0, 0), thickness=2)
                match_results.append(mr)
            else:
                print("feature match failed for initial frames")
            kps2.append((kp2, des2))
        imgs = np.concatenate([img for Succeeded, img, movement_y, kp2, des2, match_count in match_results], axis=0)
        # cv2.imshow('acc_edged', img)
        cv2.imwrite("tmp/" + os.path.basename(args.input_video_path) + '.' + args.feature_algorithm + ".png", imgs)
        movement_ys = [movement_y for Succeeded, _, movement_y, kp2, des2, match_count in match_results]
        prev_img = imgs[-1]
        prev_kp2 = kps2[-1][0]
        prev_des2 = kps2[-1][1]
        c += len(initial_frames)
        min_ratio = 100.0
        max_ratio = 0.0
        avg_ratio = 0.0
        avg_match_count = sum([match_count for Succeeded, _, movement_y, kp2, des2, match_count in match_results]) / len(match_results)
        text_output = np.zeros((480, 640, 3), dtype=np.uint8)
        speeds = np.array([float(mph) for mph in mphs[:len(initial_frames)]])
        x = 3.0
        estimated_speeds = np.array([(movement_y * x) for Succeeded, _, movement_y, kp2, des2, match_count in match_results])
        estimated_speeds = np.insert(estimated_speeds, 0, mphs[0])
        print(speeds)
        print(estimated_speeds)
        mean_square_error = ((speeds - estimated_speeds) ** 2).mean()
        while (cap.isOpened()):
            Succeeded, frame = cap.read()
            if not Succeeded:
                break
            #edged_frames = [make_edges(frame) for i, frame in enumerate(frames)]
            #img = sum(edged_frames)
            #n1 = dt.datetime.now()
            img = find_roi(frame, bottom_left_corner, bottom_right_corner, top_left, top_right)
            # Crop to roi
            img = img[y1:y2, x1:x2]
            kp1, des1 = feature_detector.detectAndCompute(img, None)
            Succeeded, _, movement_y, kp2, des2, match_count = feature_matcher(prev_img, img, prev_kp2, prev_des2, kp1, des1, True)
            #n2 = dt.datetime.now()
            #print((n2.microsecond-n1.microsecond)/1e6, 'secs')
            if Succeeded:
                movement_ys = movement_ys[1:]
                movement_ys.append(movement_y)
                movement_y = sum(movement_ys) / len(movement_ys)
                prev_img = img
                prev_kp2 = kp2
                prev_des2 = des2
                text_output = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(img=text_output, text=('Real speed:' + str(float(mphs[c]))), org=(0, 50), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(255, 0, 0), thickness=2)
                cv2.putText(img=text_output, text=('Est speed:' + str(movement_y)), org=(0, 100), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(255, 0, 0), thickness=2)
                ratio = float(mphs[c]) / movement_y
                if min_ratio > ratio:
                    min_ratio = ratio
                if max_ratio < ratio:
                    max_ratio = ratio
                if avg_ratio == 0.0:
                    avg_ratio = ratio
                else:
                    avg_ratio = (ratio + avg_ratio) / 2
                avg_match_count = (avg_match_count + match_count) / 2
                cv2.putText(img=text_output, text=str(ratio), org=(0, 150), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(255, 0, 0), thickness=2)
                cv2.putText(img=text_output, text=('Min Ratio:' + str(min_ratio)), org=(0, 200), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(255, 0, 0), thickness=2)
                cv2.putText(img=text_output, text=('Max Ratio:' + str(max_ratio)), org=(0, 250), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(255, 0, 0), thickness=2)
                cv2.putText(img=text_output, text=('Avg Ratio:' + str(avg_ratio)), org=(0, 300), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(255, 0, 0), thickness=2)
                cv2.putText(img=text_output, text=('Avg Matches:' + str(avg_match_count)), org=(0, 350), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(255, 0, 0), thickness=2)
                #cv2.imshow('match_result', match_result)
                #cv2.imwrite("tmp/matches.png", img)
            else:
                print('feature match failed')
                break
            speeds = np.append(speeds, [float(mphs[c])])
            estimated_speeds = np.append(estimated_speeds, [movement_y * x])
            mean_square_error = ((speeds - estimated_speeds) ** 2).mean()
            cv2.putText(img=text_output, text=('MSE:' + str(mean_square_error)), org=(0, 400), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(255, 0, 0), thickness=2)
            cv2.imshow('frame', frame)
            cv2.imshow('text_output', text_output)
            c += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.imwrite("tmp/" + os.path.basename(args.input_video_path) + '.' + args.feature_algorithm + ".stats.png", text_output)
    else:
        cv2.namedWindow('params', 0)
        global top_left_x, top_left_y, top_right_x, top_right_y
        top_left_x = top_left[0]
        top_left_y = top_left[1]
        top_right_x = top_right[0]
        top_right_y = top_right[1]

        def update_top_left_x(val):
            global top_left_x, top_left_y, top_right_x, top_right_y
            top_left_x = val
            print(top_left_x, top_left_y, top_right_x, top_right_y)


        def update_top_right_x(val):
            global top_left_x, top_left_y, top_right_x, top_right_y
            top_right_x = val
            print(top_left_x, top_left_y, top_right_x, top_right_y)


        def update_top_left_y(val):
            global top_left_x, top_left_y, top_right_x, top_right_y
            top_left_y = val
            print(top_left_x, top_left_y, top_right_x, top_right_y)


        def update_top_right_y(val):
            global top_left_x, top_left_y, top_right_x, top_right_y
            top_right_y = val
            print(top_left_x, top_left_y, top_right_x, top_right_y)

        cv2.createTrackbar('top_left_x', 'params', top_left_x, 640, update_top_left_x)
        cv2.createTrackbar('top_left_y', 'params', top_left_y, 480, update_top_left_y)
        cv2.createTrackbar('top_right_x', 'params', top_right_x, 640, update_top_right_x)
        cv2.createTrackbar('top_right_y', 'params', top_right_y, 480, update_top_right_y)

        global mouse_btn_down1, rect1, x11, y11, x12, y12, y21, y22, frame1, frame2
        _, frame1 = cap.read()
        _, frame2 = cap.read()
        height1, width1, _ = frame1.shape
        x11 = 220
        x12 = 442
        y21 = y11 = 256
        y22 = y12 = 337
        mouse_btn_down1 = False
        def on_mouse_event1(event, x, y, flags, frame):
            global mouse_btn_down1, rect1, x11, y11, x12, y12, y21, y22, frame1, frame2

            # Draw Rectangle
            if event == cv2.EVENT_LBUTTONDOWN:
                mouse_btn_down1 = True
                x11, y11 = x, y
                y21 = y
                if y12 > y11 and x12 > x11:
                    redraw()
            elif event == cv2.EVENT_MOUSEMOVE:
                if mouse_btn_down1 is True:
                    x12, y12 = x, y
                    y22 = y
                    if y12 > y11 and x12 > x11:
                        redraw()
            elif event == cv2.EVENT_LBUTTONUP:
                mouse_btn_down1 = False

        def redraw():
            global mouse_btn_down1, rect1, x11, y11, x12, y12, y21, y22, frame1, frame2
            frame1_1 = frame1.copy()
            frame2_1 = frame2.copy()
            f = auto_canny(frame1_1)
            f = cv2.cvtColor(f,cv2.COLOR_GRAY2BGR)
            #f = [128 if x == 255 else 0 for x in f]
            for x in f: print(x)
            cv2.imshow('f', f)
            enlarged_frame1 = cv2.resize(frame1, None, fx=1.1, fy=1.1, interpolation=cv2.INTER_CUBIC)
            enlarged_height1, enlarged_width1, _ = enlarged_frame1.shape
            crop_y1 = (enlarged_height1 - height1) / 2
            crop_y2 = (enlarged_height1 - height1) / 2 + height1
            crop_x1 = (enlarged_width1 - width1) / 2
            crop_x2 = (enlarged_width1 - width1) / 2 + width1
            cropped_enlarged_frame1 = enlarged_frame1[crop_y1: crop_y2, crop_x1:crop_x2]
            cv2.putText(img=frame1_1, text='{} {}'.format((x11, y11), (x12, y12)), org=(0, 50), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1,
                        color=(255, 0, 0), thickness=2)
            sub_frame1 = cropped_enlarged_frame1[y11:y12, x11:x12]
            sub_frame2 = frame2_1[y21:y22, x11:x12]
            cv2.putText(img=frame1_1, text='mse: {:.4f}, ssim: {:.4f}'.format(mse(sub_frame1, sub_frame2), ssim(sub_frame1, sub_frame2, multichannel=True)), org=(0, 100), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1,
                        color=(255, 0, 0), thickness=2)
            cv2.putText(img=frame2_1, text='{} {}'.format((x11, y21), (x12, y22)), org=(0, 50), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1,
                        color=(255, 0, 0), thickness=2)
            cv2.rectangle(cropped_enlarged_frame1, (x11, y11), (x12, y12), (0, 255, 0), 1)
            cv2.rectangle(frame2_1, (x11, y21), (x12, y22), (0, 255, 0), 1)
            cv2.imshow('frame1', frame1_1)
            cv2.imshow('frame2', frame2_1)
            cv2.imshow('cropped_enlarged_frame1', cropped_enlarged_frame1)

        redraw()
        cv2.setMouseCallback("frame1", on_mouse_event1, param=())
        cv2.setMouseCallback("frame2", on_mouse_event1, param=())
        while (cap.isOpened()):
            #img = make_edges(frame)
            #img = find_roi(frame, bottom_left_corner, bottom_right_corner, top_left, top_right)
            #img = perspective_transform(img, top_left_x, top_left_y, top_right_x, top_right_y, bottom_left_corner[0], bottom_left_corner[1], bottom_right_corner[0], bottom_right_corner[1])
            #img = img[y1:y2, x1:x2]
            #img = draw_sift(img)
            #img = find_blobs(frame)
            #img = find_lane(frame, bottom_left_corner, bottom_right_corner, top_left, top_right)
            #img2 = make_warp(frame, top_left_x, top_left_y, top_right_x, top_right_y, bottom_left_corner[0], bottom_left_corner[1], bottom_right_corner[0], bottom_right_corner[1])
            #cv2.putText(img=img, text=mphs[c], org=(0, 50), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(255, 0, 0), thickness=2)
            #c += 1
            #cv2.imshow('img', img)
            #cv2.imshow('img2', img2)
            #height, width, _ = frame.shape
            #print(frame[(height - 100):height, 0:width].mean())
            #_, frame = cap.read()
            #print(frame[(height - 100):height, 0:width].mean())
            # break
            k = cv2.waitKey(100) & 0xFF
            if k == ord('q'):
                break
            elif k == ord('w'):
                if y12 < height1:
                    y11 -= 1
                    y12 -= 1
                    redraw()
            elif k == ord('s'):
                if y12 < height1:
                    y11 += 1
                    y12 += 1
                    redraw()
            elif k == ord('e'):
                if y22 < height1:
                    y21 -= 1
                    y22 -= 1
                    redraw()
            elif k == ord('d'):
                if y22 < height1:
                    y21 += 1
                    y22 += 1
                    redraw()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
