import torch
import argparse
from utils import SiameseNet
import cv2
import numpy as np
import os

parser = argparse.ArgumentParser(description='Live test')
parser.add_argument('--model_location', '-l', type=str, default='model/model-epoch-{}.pth')
parser.add_argument('--epoch', '-e', type=int, default=None)
parser.add_argument('--device','-d', type=str, default=None)
parser.add_argument('--ref','-r', type=str, default='references/')
parser.add_argument('--verbose','-v', action='store_true')
args = parser.parse_args()
if not args.device:
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

def preprocess(img=None):
    return torch.as_tensor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[None, None]/255, dtype=torch.float).to(device=args.device)

refs = {
        'jump':preprocess(cv2.imread(os.path.join(args.ref, 'jump.jpg'))),
        'none':preprocess(cv2.imread(os.path.join(args.ref, 'none.jpg')))
        }

print('Loading model')
model = SiameseNet(mode='inference', weights_path=args.model_location.format(args.epoch), refs_dict=refs, device=args.device)

cap = cv2.VideoCapture(0)
rval = True

print('Recording')

while rval:
    rval, frame = cap.read()
    frame_tensor = preprocess(frame)

    scores = model(frame_tensor)

    if np.argmax(scores) == 0:
        print('jump')

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
