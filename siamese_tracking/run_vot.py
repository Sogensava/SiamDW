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

import vot

misato_path = "/home/atila/Pictures/misato/"

display_name = 'Sequence'
sequence_length = 300

class siamrpn_vot(object):
    def __init__(self,tracker_name='SiamRPN',para_name='baseline'):
        tracker_info = SiamRPN(info)

    def initialize(self, img_rgb,box):
        self.H, self.W, _ = img_rgb.shape
        init_info('init_bbox':box)
        _ = self.tracker.initialize()


def parse_args():
    """
    args for fc testing.
    """
    parser = argparse.ArgumentParser(description='PyTorch SiamFC Tracking Test')
    parser.add_argument('--arch', default='SiamRPNRes22', type=str, help='backbone architecture')
    parser.add_argument('--resume', default='/data/zpzhang/project4/siamese/Siamese/snapshot/CIResNet22RPN.model', type=str, help='pretrained model')
    parser.add_argument('--video', default='/data/zpzhang/project4/siamese/Siamese/videos/bag.mp4', type=str, help='video file path')
    parser.add_argument('--init_bbox', default=None, help='bbox in the first frame None or [lx, ly, w, h]')
    args = parser.parse_args()

    return args

def track_sequence(tracker,model,sequence_path,init_box=None):
    frame_number = "00000000"
    cv2.namedWindow(display_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(display_name, 960, 720)
    input_string = sequence_path+frame_number+'.bmp'
    print(input_string)
    frame = cv2.imread(input_string)

    if init_box is not None:
            lx, ly, w, h = init_box
            target_pos = np.array([lx + w/2, ly + h/2])
            target_sz = np.array([w, h])
            state = tracker.init(frame, target_pos, target_sz, model)  # init tracker
        
    else:
        while True:

            frame_disp = frame.copy()

            cv2.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                       1, (0, 0, 255), 1)

            lx, ly, w, h = cv2.selectROI(display_name, frame_disp, fromCenter=False)
            target_pos = np.array([lx + w / 2, ly + h / 2])
            target_sz = np.array([w, h])
            state = tracker.init(frame_disp, target_pos, target_sz, model)  # init tracker

            break
    
    while True:
        input_string = sequence_path+frame_number+'.bmp'
        print(input_string)
        frame = cv2.imread(input_string)

        if int(frame_number) == sequence_length:
            break
        
        frame_disp = frame.copy()
        state = tracker.track(state,frame_disp)
        location = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
        x1, y1, x2, y2 = int(location[0]), int(location[1]), int(location[0] + location[2]), int(location[1] + location[3])

        cv2.rectangle(frame_disp, (x1, y1), (x2, y2), (0, 255, 0), 5)

        font_color = (0, 0, 0)
        cv2.putText(frame_disp, 'Tracking!', (20, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                   font_color, 1)
        cv2.putText(frame_disp, 'Press r to reset', (20, 55), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                   font_color, 1)
        cv2.putText(frame_disp, 'Press q to quit', (20, 80), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                   font_color, 1)


        frame_number += f"{int(frame_number) + 1:08d}"
        frame_number = frame_number[-8:]
        
        cv2.imshow(display_name,frame_disp)

        if cv2.waitKey(33) & 0xFF == ord('q'):
            break

def main():
    args = parse_args()

    info = edict()
    info.arch = args.arch
    info.dataset = args.video
    info.epoch_test = True
    info.cls_type = 'thinner'

    if 'FC' in args.arch:
        net = models.__dict__[args.arch]()
        tracker = SiamFC(info)
    else:
        net = models.__dict__[args.arch](anchors_nums=5, cls_type='thinner')
        tracker = SiamRPN(info)

    print('[*] ======= Track video with {} ======='.format(args.arch))

    net = load_pretrain(net, args.resume)
    net.eval()
    net = net.cuda()

    handle = vot.VOT("rectangle")
    selection = handle.region()
    imagefile = handle.frame()

    # check init box is list or not
    if not isinstance(args.init_bbox, list) and args.init_bbox is not None:
        args.init_bbox = list(eval(args.init_bbox))
    else:
        pass

    track_sequence(tracker,net,misato_path,init_box=args.init_bbox)
    

if __name__ == "__main__":
    main()