
import cv2

import shutil
import random
import evaluation_tools
import statistics
import copy
import json
import os
import torch
import pandas as pd
import train
import numpy as np
import yaml

from torch import nn
from madeye_utils import parse_orientation_string, extract_pan, extract_tilt, extract_zoom, find_tilt_dist, find_pan_dist


from torch.utils.data import DataLoader
from torchvision import transforms
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, boolean_string
from tqdm.autonotebook import tqdm
from zoom_explore_helper import add_zoom_factors, reset_zoom_factors
from backbone import EfficientDetBackbone
from efficientdet.dataset import CocoDataset, Resizer, Normalizer, Augmenter, collater
from efficientdet.loss import FocalLoss
from utils.sync_batchnorm import patch_replication_callback
from utils.utils import replace_w_sync_bn, CustomDataParallel, get_last_weights, init_weights, boolean_string


from jetson_inference import detectNet
from jetson_utils import videosource, videooutput, log, loadimage, cudaresize, cudaallocmapped, cudafromnumpy

import requests
import time
import random

MODELS = ['yolov4', 'ssd-voc', 'tiny-yolov4', 'faster-rcnn']
SKIP = 6 # Only consider frames where frame % SKIP == 0
PERSON_CONFIDENCE_THRESH = 50.0
CAR_CONFIDENCE_THRESH = 70.0


nms_threshold = 0.5
model_params = train.Params(f'projects/coco.yml')
obj_list = model_params.obj_list
compound_coef = 0
input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]


use_float16 = False






def run_efficientdet(orientation_to_file,  model, gpu, car_thresh, person_thresh):

    threshold = 0.05
    use_cuda = gpu >= 0
    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()
    orientation_to_car_count = {}
    orientation_to_person_count = {}

    orientation_to_cars_detected = {}
    orientation_to_people_detected = {}
    # In format frame,orientation,file
    with torch.no_grad():
        for o in orientation_to_file:
            image_path  = orientation_to_file[o]
#            print('Processing ', image_path)
            ori_imgs, framed_imgs, framed_metas = preprocess(image_path, max_size=input_sizes[compound_coef], mean=model_params.mean, std=model_params.std)
            x = torch.from_numpy(framed_imgs[0])

            if use_cuda:
                x = x.cuda(gpu)
                if use_float16:
                    x = x.half()
                else:
                    x = x.float()
            else:
                x = x.float()

            people_count = 0
            car_count = 0
            x = x.unsqueeze(0).permute(0, 3, 1, 2)
            features, regression, classification, anchors = model(x)

            preds = postprocess(x,
                                anchors, regression, classification,
                                regressBoxes, clipBoxes,
                                threshold, nms_threshold)
            if not preds:
                orientation_to_car_count[o] = 0
                orientation_to_person_count[o] = 0
                continue

            preds = invert_affine(framed_metas, preds)[0]

            scores = preds['scores']
            class_ids = preds['class_ids']
            rois = preds['rois']
            idx = 0

            orientation_to_cars_detected[o] = []
            orientation_to_people_detected[o] = []
            if len(scores) == 0:
                orientation_to_car_count[o] = 0
                orientation_to_person_count[o] = 0
                continue
            for i in range(0,len(scores)):
                # Class id 0 is car, 1 is person
                if scores[i] >= car_thresh and class_ids[i] == 0 :
                    car_count += 1
                    orientation_to_cars_detected[o].append(  rois[i])
                elif scores[i] >= person_thresh  and class_ids[i] == 1:
#                    print(type(rois[i]))
#                    print( rois[i][0]  )
                    orientation_to_people_detected[o].append(rois[i])
                    people_count += 1
            orientation_to_car_count[o] = car_count
            orientation_to_person_count[o] = people_count
    return orientation_to_car_count, orientation_to_person_count, orientation_to_cars_detected, orientation_to_people_detected



# Detctnet main
def main():

    url = "http://128.112.34.129/cgi-bin/ptzctrl.cgi"

    # Left, middle/up, right, middle/down, middle
    commands = ("?ptzcmd&abs&24&20&FF80&0000", "?ptzcmd&abs&24&20&FE80&0070", "?ptzcmd&abs&24&20&FD80&0000", "?ptzcmd&abs&24&20&FE80&FF70", "?ptzcmd&abs&24&20&FE80&0000" )
    trace = [

    ]
    for i in range(0,100):
        trace.append(commands)

    # load the object detection network
    net = detectnet(args.network, sys.argv, args.threshold)


    # to properly know which camera to use, make sure to do `ls /dev | grep video`.
    # the input to videocapture is the index corresponding to the connected camera
    cap = cv2.videocapture(0)

    for movements in trace:
        for m in movements:
            command = url + m
            time.sleep(0.3)
            ret, frame = cap.read()
            img = Image.fromarray(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            cuda_mem = cudaFromNumpy(frame)
            detections = net.Detect(cuda_mem, overlay=args.overlay)
            print(len(detections), ' detected')
            for idx,d in enumerate(detections):
                print(d)
                print(dir(d))
                cv2.rectangle(frame,(int(d.Left),int(d.Top)),(int(d.Right),int(d.Bottom)),(0,255,0),2)
                cv2.putText(frame,coco_classes[d.ClassID],(int(d.Right)+10,int(d.Bottom)),0,0.3,(0,255,0))

    #            roi=frame[int(d.Top):int(d.Bottom),int(d.Left):int(d.Right)]
    #            cv2.imwrite(str(idx) + '.jpg', roi)

            cv2.imshow('Video stream ', frame)





# EfficientDet main
#def main():
#
#
#
#    # EfficientDet initialization
#    with open('controller_params.yml') as f_params:
#        params = yaml.safe_load(f_params)
#    model_to_weights_paths = {}
#    model_to_weights_paths['faster-rcnn'] = params['faster_rcnn_weights']
#    model_to_weights_paths['yolov4'] = params['yolov4_weights']
#    model_to_weights_paths['tiny-yolov4'] = params['tiny_yolov4_weights']
#    model_to_weights_paths['ssd-voc'] = params['ssd_voc_weights']
#    gpu = params['gpu']
#    weights_path = None
#    model_to_efficientdet = {}
#    workload = [('yolov4', 'count', 'person')]
#    for q in workload:
#        if params['use_efficientdet']:
#            efficientdet = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
#                                     ratios=eval(model_params.anchors_ratios), scales=eval(model_params.anchors_scales))
#            if q[0] in model_to_weights_paths:
#                weights_path = model_to_weights_paths[q[0]]
#                print('Loading ', weights_path)
#                efficientdet.load_state_dict(torch.load(weights_path))
#                if gpu >= 0:
#                    efficientdet.to(f'cuda:{gpu}')
#            efficientdet.requires_grad_(False)
#            model_to_efficientdet[q[0]] = efficientdet
#        else:
#            model_to_efficientdet[q[0]] = None
#
#    model_to_car_thresh = {}
#    model_to_person_thresh = {}
#    model_to_car_thresh['faster-rcnn'] = 0.3
#    model_to_car_thresh['yolov4'] = 0.3
#    model_to_car_thresh['tiny-yolov4'] = 0.3
#    model_to_car_thresh['ssd-voc'] = 0.3
#
#    model_to_person_thresh['faster-rcnn'] = 0.2
#    model_to_person_thresh['yolov4'] = 0.2
#    model_to_person_thresh['tiny-yolov4'] = 0.2
#    model_to_person_thresh['ssd-voc'] = 0.2
#
#
#
#    exit()
##    orientation_to_file = {}
##    orientation_to_file['0-0-1'] = '/home/mikedw/Desktop/cars.jpg'
##    print(model_to_efficientdet['yolov4'])
##    run_efficientdet(orientation_to_file,  model_to_efficientdet['yolov4'], gpu, model_to_car_thresh['yolov4'], model_to_person_thresh['yolov4'])
##
##    # To properly know which camera to use, make sure to do `ls /dev | grep video`.
##    # THe input to VideoCapture is the index corresponding to the connected camera
##    cap = cv2.VideoCapture(0)
##
##
##    while True:
##        ret, frame = cap.read()
##        print(type(frame))
###        cv2.imshow('Video stream ', frame)
##        if cv2.waitKey(1) & 0xFF == ord('q'):
##            break
##
##    cap.release()
##    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()



