#!/usr/bin/python
import vot
import sys
import time

import _init_paths

import os
import cv2
import random
import argparse
import numpy as np

import models.models as models

from os.path import exists, join
from torch.autograd import Variable
from tracker.siamfc import SiamFC
from tracker.siamrpn import SiamRPN
from easydict import EasyDict as edict
from utils.utils import load_pretrain, cxy_wh_2_rect, get_axis_aligned_bbox, load_dataset, poly_iou

args = '--arch SiamRPNRes22 --resume snapshot/CIResNet22_RPN.pth --video /home/atila/Pictures/misato'
arch = 'SiamRPNRes22'
resume = '/home/atila/aliberk_ws/vot_ws/trackers/paper_imp/SiamDW/snapshot/CIResNet22_RPN.pth'

handle = vot.VOT("rectangle")
selection = handle.region()

# Get paths of color images
colorimage = handle.frame()
if not colorimage:
    sys.exit(0)

# TODO: Read images and initialize tracker
frame = cv2.cvtColor(cv2.imread(colorimage),cv2.COLOR_BGR2RGB)
info = edict()
info.arch = arch
info.epoch_test = True
info.cls_type = 'thinner'
info.dataset = None
net = models.__dict__[info.arch](anchors_nums=5, cls_type='thinner')
tracker = SiamRPN(info)

net = load_pretrain(net, resume)
net.eval()
net = net.cuda()

init_box = [selection.x, selection.y, selection.width, selection.height]
target_pos = np.array([init_box[0] + int(init_box[2]/2), init_box[1] + int(init_box[3]/2)])
target_sz = np.array([init_box[2], init_box[3]])
state = tracker.init(frame, target_pos, target_sz, net)


while True:
    imagefile = handle.frame()
    if not imagefile:
        break

    # Get paths of color and depth images
    # colorimage = handle.frame()
    print("colorimage sonrasi")
    print(imagefile)
    # TODO: Read both images and track the object
    frame_upd = cv2.cvtColor(cv2.imread(imagefile),cv2.COLOR_BGR2RGB)
    state = tracker.track(state,frame_upd)
    location = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
    x1, y1, x2, y2 = int(location[0]), int(location[1]), int(location[0] + location[2]), int(location[1] + location[3])
    # selection.x = x1
    # selection.y = y1
    # selection.w = x2-x1
    # selection.h = y2-y1
    
    handle.report(vot.Rectangle(x1,y1,(x2-x1),(y2-y1)))