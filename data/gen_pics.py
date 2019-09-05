import argparse
import os
import cv2
from time import time

parser = argparse.ArgumentParser(description='Get pics of frames from videos')
parser.add_argument('output_path', help='directory to store the pictures')
parser.add_argument('name', help='Name of the action')
parser.add_argument('-i', '--interval', help='intervals in which to take pictures', type=int, default=1)
parser.add_argument('-n', '--number', help='number of pics to take', type=int)

args = parser.parse_args()

if not os.path.exists(args.output_path):
    os.mkdir(args.output_path)

vidcap = cv2.VideoCapture(0)
ret = True
start = time()
counter = 1

while ret and counter <= args.number:
    ret, frame= vidcap.read()

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

    if time() - start >= args.interval:
        cv2.imwrite(os.path.join(args.output_path, args.name+'_{}.jpg'.format(counter)), frame)
        start = time()
        counter+=1

vidcap.release()
cv2.destroyAllWindows()
