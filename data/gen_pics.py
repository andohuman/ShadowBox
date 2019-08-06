import argparse
import os
import cv2

parser = argparse.ArgumentParser(description='Get pics of frames from videos')
parser.add_argument('input_path', help='full path to input video file')
parser.add_argument('output_path', help='directory to store the pictures')
parser.add_argument('-i', '--interval', help='intervals in which to take pictures', type=int, default=1)

args = parser.parse_args()

name = os.path.join(args.output_path,os.path.basename(os.path.splitext(args.input_path)[0])+"_sec{}.jpg") #Output file name for image (gotta fill the placeholder)

if not os.path.isdir(args.output_path):
	os.mkdir(args.output_path)

vidcap = cv2.VideoCapture(args.input_path)
vidcap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
secs = int(vidcap.get(cv2.CAP_PROP_POS_MSEC)/1000)

for i in range(0,secs,args.interval):
	vidcap.set(cv2.CAP_PROP_POS_MSEC, i*1000)
	rval, frame = vidcap.read()
	if rval:
		cv2.imwrite(name.format(i), frame)
	else:
		break