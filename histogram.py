import cv2
import numpy as np
from matplotlib import pyplot as plt
import argparse

def main():
  parser = argparse.ArgumentParser(description='Process some integers.')
  parser.add_argument("-i", "--image_path", dest = 'image_path', required = True)
  args = parser.parse_args()

  img = cv2.imread(args.image_path)
  hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
  hist = cv2.calcHist( [hsv], [0, 1], None, [180, 256], [0, 180, 0, 256] )
  plt.imshow(hist,interpolation = 'nearest')
  plt.show()

if __name__ == '__main__':
  main()
