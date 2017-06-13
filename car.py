import argparse
import cv2

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("-i", "--input_video_path", dest = 'input_video_path', required = True)
args = parser.parse_args()
# capture frames from a video
cap = cv2.VideoCapture(args.input_video_path)

# Trained XML classifiers describes some features of some object we want to detect
car_cascade = cv2.CascadeClassifier('cars.xml')

# loop runs if capturing has been initialized.
while True:
  # reads frames from a video
  ret, frames = cap.read()

  # convert to gray scale of each frames
  gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)


  # Detects cars of different sizes in the input image
  cars = car_cascade.detectMultiScale(gray, 1.1, 1)

  # To draw a rectangle in each cars
  for (x,y,w,h) in cars:
      cv2.rectangle(frames,(x,y),(x+w,y+h),(0,0,255),2)

  # Display frames in a window
  cv2.imshow('video2', frames)

  # Wait for Esc key to stop
  if cv2.waitKey(33) == 27:
    break

# De-allocate any associated memory usage
cv2.destroyAllWindows()