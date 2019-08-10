import torch
import argparse
from utils import Pairloader, SiameseNet
import cv2
import numpy as np

parser = argparse.ArgumentParser(description='Validate SiameseNet')
parser.add_argument('--model_location', '-l', type=str, default='model/{}-epoch-{}.pth')
parser.add_argument('--epoch', '-e', type=int, default=None)
parser.add_argument('--device','-d', type=str, default=None)
args = parser.parse_args()

if not args.device:
	args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = SiameseNet().to(device=args.device)
model.load_state_dict(torch.load(args.model_location.format('model',args.epoch), map_location=args.device))
model.eval()


hand = torch.as_tensor(cv2.imread('data/valid/hand_4.jpg', 0)[None, None, ...]/255, dtype=torch.float).to(device=args.device)
nyet = torch.as_tensor(cv2.imread('data/valid/nyet_5.jpg', 0)[None, None, ...]/255, dtype=torch.float).to(device=args.device)
cap = cv2.VideoCapture(0)
rval = True

while rval:
	rval, frame = cap.read()
	frame_tensor = torch.as_tensor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[None, None, ...]/255, dtype=torch.float).to(device=args.device)
	
	hand_compare = model([hand, frame_tensor])
	nyet_compare = model([nyet, frame_tensor])

	if hand_compare.item() > nyet_compare.item():
		print('Hand', hand_compare.item(), nyet_compare.item())
	elif nyet_compare.item() > hand_compare.item():
		print('Nyet', hand_compare.item(), nyet_compare.item())

	cv2.imshow('frame', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()