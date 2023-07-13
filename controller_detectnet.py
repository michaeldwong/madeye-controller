
import cv2

import argparse
import shutil
import random
import evaluation_tools
import statistics
import copy
import json
import os
import torch
import pandas as pd
import numpy as np
import yaml
import sys

from torch import nn
from madeye_utils import parse_orientation_string, extract_pan, extract_tilt, extract_zoom, find_tilt_dist, find_pan_dist


from zoom_explore_helper import add_zoom_factors, reset_zoom_factors


from jetson_inference import detectNet
from jetson_utils import videoSource, videoOutput,  loadImage, cudaResize, cudaAllocMapped, cudaFromNumpy, Log

from PIL import Image

import requests
import time
import random

MODELS = ['yolov4', 'ssd-voc', 'tiny-yolov4', 'faster-rcnn']
SKIP = 6 # Only consider frames where frame % SKIP == 0
PERSON_CONFIDENCE_THRESH = 50.0
CAR_CONFIDENCE_THRESH = 70.0


coco_classes = [
"unlabeled",
"person",
"bicycle",
"car",
"motorcycle",
"airplane",
"bus",
"train",
"truck",
"boat",
"traffic light",
"fire hydrant",
"street sign",
"stop sign",
"parking meter",
"bench",
"bird",
"cat",
"dog",
"horse",
"sheep",
"cow",
"elephant",
"bear",
"zebra",
"giraffe",
"hat",
"backpack",
"umbrella",
"shoe",
"eye glasses",
"handbag",
"tie",
"suitcase",
"frisbee",
"skis",
"snowboard",
"sports ball",
"kite",
"baseball bat",
"baseball glove",
"skateboard",
"surfboard",
"tennis racket",
"bottle",
"plate",
"wine glass",
"cup",
"fork",
"knife",
"spoon",
"bowl",
"banana",
"apple",
"sandwich",
"orange",
"broccoli",
"carrot",
"hot dog",
"pizza",
"donut",
"cake",
"chair",
"couch",
"potted plant",
"bed",
"mirror",
"dining table",
"window",
"desk",
"toilet",
"door",
"tv",
"laptop",
"mouse",
"remote",
"keyboard",
"cell phone",
"microwave",
"oven",
"toaster",
"sink",
"refrigerator",
"blender",
"book",
"clock",
"vase",
"scissors",
"teddy bear",
"hair drier",
"toothbrush"
]






def generate_plus_formation(current_orientation):
    items = current_orientation.split('-')
    pan = int(items[0])
    zoom = int(items[-1])
    if pan == 0:
        left_horz = 330
    else:
        left_horz = int(items[0]) - 30
    if pan == 330:
        right_horz = 0
    else:
        right_horz = int(items[0]) + 30

    if len(items) == 4:
        tilt = int(items[2]) * -1
    else:
        tilt = int(items[1])
    top_tilt = tilt + 15
    bottom_tilt = tilt - 15

    if tilt == 30:
        return [ f'{left_horz}-{tilt}-{zoom}',
                 f'{right_horz}-{tilt}-{zoom}',
                 current_orientation,
                 f'{pan}-{bottom_tilt}-{zoom}' ]
    elif tilt == -30:
        return [ f'{left_horz}-{tilt}-{zoom}',
                 f'{right_horz}-{tilt}-{zoom}',
                 current_orientation,
                 f'{pan}-{top_tilt}-{zoom}']
    return [ f'{left_horz}-{tilt}-{zoom}',
             f'{right_horz}-{tilt}-{zoom}',
             current_orientation,
             f'{pan}-{top_tilt}-{zoom}',
             f'{pan}-{bottom_tilt}-{zoom}' ]

def rank_orientations(orientation_to_count):
    sorted_dict = {k: v for k, v in sorted(orientation_to_count.items(), key=lambda item: item[1] * -1)}
    orientation_to_rank = {}
    last_count = 0
    rank = 0
    for o in sorted_dict:
        count = sorted_dict[o]
        if count != last_count:
            last_count = count
            rank += 1
        if rank == 0:
            rank += 1
        orientation_to_rank[o] = rank
    return orientation_to_rank

class ModelWithLoss(nn.Module):
    def __init__(self, model, debug=False):
        super().__init__()
        self.criterion = FocalLoss()
        self.model = model
        self.debug = debug

    def forward(self, imgs, annotations, obj_list=None):
        _, regression, classification, anchors = self.model(imgs)
        if self.debug:
            cls_loss, reg_loss = self.criterion(classification, regression, anchors, annotations,
                                                imgs=imgs, obj_list=obj_list)
        else:
            cls_loss, reg_loss = self.criterion(classification, regression, anchors, annotations)
        return cls_loss, reg_loss


def create_annotations(f, image_file, orientation_df, orientation, object_type, json_dict, image_id, annotation_id):
    if object_type != 'car' and object_type != 'person' and object_type != 'both':
        raise Exception('Incorrect object type')
    count = 0
    for idx, row in orientation_df.iterrows():
        if object_type == 'car' or object_type == 'both':
            if row['class'] == 'car' and row['confidence'] >= CAR_CONFIDENCE_THRESH:
                xmin = row['left']
                xmax = row['right']
                ymin = row['top']
                ymax = row['bottom']
                json_dict['annotations'].append({"id": annotation_id,"image_id": image_id, "category_id": 1, "iscrowd": 0, "image_id": image_id, "bbox": [xmin, ymin, xmax - xmin, ymax - ymin ], "area": (ymax - ymin) * (xmax - xmin), "segmentation": [[xmin, ymin, xmax , ymin, xmax, ymax, xmin, ymax ]] })
        if object_type == 'person' or object_type == 'both':
            if row['class'] == 'person' and row['confidence'] >= PERSON_CONFIDENCE_THRESH:
                xmin = row['left']
                xmax = row['right']
                ymin = row['top']
                ymax = row['bottom']
                json_dict['annotations'].append({"id": annotation_id, "image_id": image_id, "category_id": 2, "iscrowd": 0, "image_id": image_id, "bbox": [xmin, ymin, xmax - xmin, ymax - ymin ], "area": (ymax - ymin) * (xmax - xmin), "segmentation": [[xmin, ymin, xmax , ymin, xmax, ymax, xmin, ymax ]] })
        annotation_id += 1
    return annotation_id




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



def find_directions_from_orientations(worst_orientation, best_orientation, orientations):
    worst_pan = extract_pan(worst_orientation)
    best_pan = extract_pan(best_orientation)

    worst_tilt = extract_tilt(worst_orientation)
    best_tilt = extract_tilt(best_orientation)
   
    best_is_left = False
    best_is_right = False
    best_is_top = False
    best_is_bottom = False 
    if worst_pan != best_pan:
        while worst_pan != best_pan:
            # Keep going left. If we reach best pan, we know best pan is to the left.
            # Otherwise, we run into the end of the region and know best pan is to the right
            new_worst_pan = rotate_left(worst_pan)
            if new_worst_pan == worst_pan:
                best_is_right = True
                break
            worst_pan = new_worst_pan
        if not best_is_right:
            best_is_left = True
    if worst_tilt != best_tilt:
        if best_tilt > worst_tilt:
            best_is_top = True
        else:
            best_is_bottom = True
    return best_is_left, best_is_top, best_is_right, best_is_bottom




def neighboring_orientations_madeye(anchor_orientation, current_formation, orientations, orientation_to_historical_scores):
    total_num_historical_scores = 0
    for o in orientation_to_historical_scores:
        total_num_historical_scores += orientation_to_historical_scores[o]

    if len(current_formation) == 0 or total_num_historical_scores < 7:
        is_left = False
        is_right = False
        is_top = False
        is_bottom = False
        o1 = rotate_left(anchor_orientation, orientations)
        if o1 == anchor_orientation:
            is_left = True 
        o2 = rotate_right(anchor_orientation, orientations)
        if o2 == anchor_orientation:
            is_right = True
        o3 = rotate_up(anchor_orientation, orientations)
        if o3 == anchor_orientation:
            is_top = True
        o4 = rotate_down(anchor_orientation, orientations)
        if o4 == anchor_orientation:
            is_bottom = True
        # Generate alternate orientations if we're on the edge of the region
        if is_left:
            # If anchor is far left, o1 is bad
            if is_top:
                # anchor is in top left, o3 is bad too
                o1 = rotate_down(o2, orientations)
                o3 =  rotate_down(o1, orientations)
            elif is_bottom:
                # anchor is in bottom left, o4 is bad too
                o1 = rotate_up(o2, orientations)
                o4 = rotate_up(o1, orientations)
            else:
                # Top right by default
                o1 = rotate_up(o2, orientations)

        if is_right:
            # If current orientation is far right, o2 is bad
            if is_top:
                # anchor is top right, o3 is bad too
                o2 = rotate_down(o1, orientations)
                o3 = rotate_down(o2, orientations)
            elif is_bottom:
                # anchor is bottom right, o4 is bad too
                o2 = rotate_up(o1, orientations)
                o4 = rotate_up(o2, orientations)
            else:
                # Top left by default
                o2 = rotate_up(o1, orientations)

        if is_top:
            # If current orientation is top, o3 is bad
            if is_left:
                # anchor is far left, o1 is bad
                o1 = rotate_down(o2, orientations)
                o3 = rotate_down(o1, orientations)
            elif is_right:
                # anchor is far right, o2 is bad
                o2 = rotate_down(o1, orientations)
                o3 = rotate_down(o2, orientations)
            else:
                # Bottom right by default
                o3 = rotate_down(o2, orientations)
        if is_bottom:
            # If current orientation is bottom, o4 is bad
            if is_left:
                # anchor is far left, o1 is bad
                o1 = rotate_up(o2, orientations)
                o4 = rotate_up(o1, orientations)
            elif is_right:
                # anchor is far right, o2 is bad
                o2 = rotate_up(o1, orientations)
                o4 = rotate_up(o2, orientations)
            else:
                # Top left by default
                o4 = rotate_up(o1, orientations)
        return [o1, o2, o3, o4, anchor_orientation]
    # Swap out worst orientation with a new orientation in the direction of best
    orientation_to_avg_count = {}
    worst_orientation = current_formation[0]
    worst_score = 1000.0
    best_orientation = current_formation[0]
    best_score = -1000.0
    for o in orientation_to_historical_scores:
        orientation_to_avg_count[o] = sum(orientation_to_historical_scores[o]) / len(orientation_to_historical_scores[o])
        if orientation_to_avg_count[o] < worst_score:
            worst_score = orientation_to_avg_count[o]
            worst_orientation = o
        if orientation_to_avg_count[o] > best_score:
            best_score = orientation_to_avg_count[o]
            best_orientation = o
    if worst_score == anchor_orientation:
        orientation_to_historical_scores.clear()
        return current_formation
    if best_score / worst_score >= 2.0:
        # Try to get new orientation in direction of best direction
        best_is_left, best_is_top, best_is_right, best_is_bottom = find_directions_from_orientations(worst_orientation, best_orientation, orientations)
        best_orientation_copy = best_orientation
        new_orientation = best_orientation
        if best_is_left:
            new_orientation = rotate_left(best_orientation_copy)
            if best_is_top or (new_orientation == best_orientation_copy or new_orientation in current_formation):
                new_orientation = rotate_up(best_orientation_copy)
            elif best_is_bottom or  (new_orientation == best_orientation_copy or new_orientation in current_formation):
                new_orientation = rotate_down(best_orientation_copy)
        elif best_is_right:
            new_orientation = rotate_right(new_orientation)
            if best_is_top or (new_orientation == best_orientation_copy  or new_orientation in current_formation):
                new_orientation = rotate_up(best_orientation_copy)
            elif best_is_bottom or  (new_orientation == best_orientation_copy or new_orientation in current_formation):
                new_orientation = rotate_down(best_orientation_copy)
        if best_is_top and (new_orientation == best_orientation_copy or new_orientation in current_formation):
            new_orientation = rotate_top(best_orientation_copy)
        elif best_is_bottom and (new_orientation == best_orientation_copy or new_orientation in current_formation):
            new_orientation = rotate_bottom(best_orientation_copy)
        if new_orientation == best_orientation_copy or new_orientation in current_formation:
            # best orientation is already at edge of region or new orientation is already in the current formation,
            # return old formation
            return current_formation
        new_formation = []
        for o in current_formation:
            if o == worst_orientation:
                continue 
            new_formation.append(o)
        new_formation.append(new_orientation)
        return new_formation
    else:
        return current_formation



def zoom_out(current_orientation, orientations):
    pan = extract_pan(current_orientation)
    tilt = extract_tilt(current_orientation)
    zoom = extract_zoom(current_orientation)
    if zoom > 1:
        zoom -= 1
    new_orientation = "{}-{}-{}".format(pan, tilt, zoom)
    return new_orientation

def zoom_in(current_orientation, orientations):
    pan = extract_pan(current_orientation)
    tilt = extract_tilt(current_orientation)
    zoom = extract_zoom(current_orientation)
    if zoom < 3:
        zoom += 1
    new_orientation = "{}-{}-{}".format(pan, tilt, zoom)
    return new_orientation

def rotate_up(current_orientation, orientations):
    pan = extract_pan(current_orientation)
    tilt = extract_tilt(current_orientation)
    zoom = extract_zoom(current_orientation)
    if tilt < 30:
        tilt += 15
    new_orientation = "{}-{}-{}".format(pan, tilt, zoom)
    if new_orientation in orientations:
        return new_orientation
    else:
        return current_orientation


def rotate_down(current_orientation, orientations):
    pan = extract_pan(current_orientation)
    tilt = extract_tilt(current_orientation)
    zoom = extract_zoom(current_orientation)
    if tilt > -30:
        tilt -= 15
    new_orientation = "{}-{}-{}".format(pan, tilt, zoom)
    if new_orientation in orientations:
        return new_orientation
    else:
        return current_orientation

def rotate_left(current_orientation, orientations):
    pan = extract_pan(current_orientation)
    tilt = extract_tilt(current_orientation)
    zoom = extract_zoom(current_orientation)
    pan -= 30
    if pan < 0:
        pan = 330
    new_orientation = "{}-{}-{}".format(pan, tilt, zoom)
    if new_orientation in orientations:
        return new_orientation
    else:
        return current_orientation

def rotate_right(current_orientation, orientations):
    pan = extract_pan(current_orientation)
    tilt = extract_tilt(current_orientation)
    zoom = extract_zoom(current_orientation)
    pan += 30
    if pan > 330:
        pan = 0
    new_orientation = "{}-{}-{}".format(pan, tilt, zoom)
    if new_orientation in orientations:
        return new_orientation
    else:
        return current_orientation
 




def compute_shortest_distance(orientation, shape_orientations, all_orientations):
    orientation = orientation[:-1]+'1'
    shape_orientations = [s[:-1]+'1' for s in shape_orientations]
    all_orientations = [s[:-1]+'1' for s in all_orientations]
    all_orientations = set(all_orientations)
    shape_orientations = set(shape_orientations)
    bfs_queue = [(orientation,0)]
    visited = set()
    while len(bfs_queue) > 0:
        (current_orientation, gap) = bfs_queue.pop(0)
        if current_orientation in shape_orientations:
            return gap
        neighbors = get_all_neighbors(current_orientation)
        for n in neighbors:
            if n not in visited:
                visited.add(n)
                if n in all_orientations:
                    bfs_queue.append((n, gap + 1))
            
    return 100000

def print_stats_about_our_perf(trace_of_our_orientations, trace_of_static_cross_formation, trace_of_best_dynamic_orientations, trace_of_distance_between_us_and_best_dynamic, trace_of_regions_we_chose, static_cross_formation):
    print(f"trace of our111 orientations: {trace_of_our_orientations}")
    print(f"trace of static orientations: {trace_of_static_cross_formation}")
    print(f"trace of dynami orientations: {trace_of_best_dynamic_orientations}")
    print(f"trace of dynamic distances  : {trace_of_distance_between_us_and_best_dynamic}")
    backup_trace_of_best_dynamic_orientations = trace_of_best_dynamic_orientations
    backup_trace_of_regions_we_chose = trace_of_regions_we_chose
    backup_static_cross_formation = static_cross_formation
    # input(f'one set of frames complete')
    number_of_indices_that_overlap_ignoring_zoom = 0
    number_of_indices_that_overlap_considering_zoom = 0
    for i in range(len(trace_of_our_orientations)):
        if trace_of_best_dynamic_orientations[i] in set(trace_of_regions_we_chose[i]["our_region"]):
            number_of_indices_that_overlap_considering_zoom += 1
        trace_of_best_dynamic_orientations[i] = trace_of_best_dynamic_orientations[i][:-1]+'1' 
        trace_of_regions_we_chose[i]["our_region"] = [x[:-1] + '1' for x in trace_of_regions_we_chose[i]["our_region"]]
        if trace_of_best_dynamic_orientations[i] in set(trace_of_regions_we_chose[i]['our_region']):
            number_of_indices_that_overlap_ignoring_zoom += 1
    print(f"mur: ours: num choices we made: {len(trace_of_our_orientations)}. out of that, fraction of chosen orientations that overlap with zoom: {number_of_indices_that_overlap_considering_zoom/len(trace_of_our_orientations)}, ignoring zoom: {number_of_indices_that_overlap_ignoring_zoom/len(trace_of_our_orientations)}")
    with open("our_results.json", "a") as f:
        f.write(f"mur: ours: num choices we made: {len(trace_of_our_orientations)}. out of that, fraction of chosen orientations that overlap with zoom: {number_of_indices_that_overlap_considering_zoom/len(trace_of_our_orientations)}, ignoring zoom: {number_of_indices_that_overlap_ignoring_zoom/len(trace_of_our_orientations)}\n")
    trace_of_best_dynamic_orientations = backup_trace_of_best_dynamic_orientations
    number_of_indices_that_overlap_ignoring_zoom = 0
    number_of_indices_that_overlap_considering_zoom = 0
    for i in range(len(trace_of_static_cross_formation)):
        # print(f"trace_of_best_dynamic_orientations[i]:{trace_of_best_dynamic_orientations[i]}")
        # print(f"trace_of_static_cross_formation[i]:{trace_of_static_cross_formation[i]}")
        if trace_of_best_dynamic_orientations[i] in set(static_cross_formation):
            number_of_indices_that_overlap_considering_zoom += 1
            # print(f"number_of_indices_that_overlap_considering_zoom+=1")
        trace_of_best_dynamic_orientations[i] = trace_of_best_dynamic_orientations[i][:-1]+'1' 
        static_cross_formation = [x[:-1] + '1' for x in static_cross_formation]
        if trace_of_best_dynamic_orientations[i] in set(static_cross_formation):
            number_of_indices_that_overlap_ignoring_zoom += 1
    #         print(f"number_of_indices_that_overlap_ignoring_zoom+=1")
    # print(f"number_of_indices_that_overlap_ignoring_zoom is {number_of_indices_that_overlap_ignoring_zoom}")
    # print(f"number_of_indices_that_overlap_considering_zoom is {number_of_indices_that_overlap_considering_zoom}")
    print(f"mur: static: num choices we made: {len(trace_of_our_orientations)}. out of that, fraction of chosen orientations that overlap with zoom: {number_of_indices_that_overlap_considering_zoom/len(trace_of_our_orientations)}, ignoring zoom: {number_of_indices_that_overlap_ignoring_zoom/len(trace_of_our_orientations)}")
    with open("our_results.json", "a") as f:
        # f.write(f"mur: static choices: {trace_of_static_cross_formation}")
        # f.write(f"mur: dynamic choices: {trace_of_best_dynamic_orientations}")
        f.write(f"mur: static: num choices we made: {len(trace_of_our_orientations)}. out of that, fraction of chosen orientations that overlap with zoom: {number_of_indices_that_overlap_considering_zoom/len(trace_of_our_orientations)}, ignoring zoom: {number_of_indices_that_overlap_ignoring_zoom/len(trace_of_our_orientations)}\n")
    with open("our_results.json", "a") as f:
        f.write(f"mur: choices: ours: {trace_of_our_orientations}, static: {trace_of_static_cross_formation}, dynamic: {trace_of_best_dynamic_orientations}")
    trace_of_regions_we_chose = backup_trace_of_regions_we_chose
    static_cross_formation = backup_static_cross_formation


def generate_all_orientations():
    orientations = []
    # r1 controls horizontal rotation. r1 = 0 means center point of 0.5
    # r2 contorls vertical rotation. r2 = 0 is focused on the ground, r2 = 90 is straight
    for r1 in range(0,360,30):
        for r2 in  [ -30, -15, 0, 15, 30]:
            for z in [1,2,3]:
                orientations.append(f'{r1}-{r2}-{z}')
    return orientations



def save_checkpoint_continual_learning(model, name, saved_path):
    if isinstance(model, CustomDataParallel):
        torch.save(model.module.model.state_dict(), os.path.join(saved_path, name))
    else:
        torch.save(model.model.state_dict(), os.path.join(saved_path, name))

def get_all_neighbors(orientation):
    neighbors_with_rotation = get_all_neighbors_with_rotation(orientation)
    neighbors = [n[0] for n in neighbors_with_rotation]
    return neighbors

##### shape helper stuff below
def get_all_neighbors_with_rotation(orientation):
    neighbors = []
    pan = extract_pan(orientation)
    tilt = extract_tilt(orientation)
    zoom = extract_zoom(orientation)
    if pan == 330:
        left_pan = 300
        right_pan = 0
    elif pan == 0:
        left_pan = 330
        right_pan = 30
    else:
        left_pan = pan - 30
        right_pan = pan + 30
    up_tilt = tilt + 15
    down_tilt = tilt - 15
    if left_pan != -1:
        left_orientation = "{}-{}-{}".format(left_pan, tilt, zoom)
        neighbors.append((left_orientation, 30))
    if right_pan != -1:
        right_orientation = "{}-{}-{}".format(right_pan, tilt, zoom)
        neighbors.append((right_orientation, 30))            
    if up_tilt >= -30 and up_tilt <= 30:
        up_orientation = "{}-{}-{}".format(pan, up_tilt, zoom)
        neighbors.append((up_orientation, 15))
    if down_tilt >= -30 and down_tilt <= 30:
        down_orientation = "{}-{}-{}".format(pan, down_tilt, zoom)
        neighbors.append((down_orientation, 15))

    # diagonals
    if left_pan != -1:
        # upper left
        if up_tilt >= -30 and up_tilt <= 30:
            left_up_orientation = "{}-{}-{}".format(left_pan, up_tilt, zoom)
            neighbors.append((left_up_orientation, 33.5))
        # lower left
        if down_tilt >= -30 and down_tilt <= 30:
            left_down_orientation = "{}-{}-{}".format(left_pan, down_tilt, zoom)
            neighbors.append((left_down_orientation, 33.5))
    
    if right_pan != -1:
        # upper right
        if up_tilt >= -30 and up_tilt <= 30:
            right_up_orientation = "{}-{}-{}".format(right_pan, up_tilt, zoom)
            neighbors.append((right_up_orientation, 33.5))
        # lower right
        if down_tilt >= -30 and down_tilt <= 30:
            right_down_orientation = "{}-{}-{}".format(right_pan, down_tilt, zoom)
            neighbors.append((right_down_orientation, 33.5))
    return neighbors

def get_all_neighbors(orientation):
    neighbors_with_rotation = get_all_neighbors_with_rotation(orientation)
    neighbors = [n[0] for n in neighbors_with_rotation]
    return neighbors



def neighboring_orientations_delta_method_madeye(anchor_orientation, 
                                                 current_formation, 
                                                 orientations, 
                                                 orientation_to_historical_scores, 
                                                 orientation_to_current_scores, 
                                                 orientation_to_current_counts, 
                                                 orientation_to_historical_counts, 
                                                 orientation_to_current_mike_factor, 
                                                 step_number, 
                                                 orientation_to_visited_step_numbers, 
                                                 peek_orientations, 
                                                 orientation_to_current_car_boxes, 
                                                 orientation_to_current_person_boxes, 
                                                 zoom_explorations_in_progress, 
                                                 num_frames_to_keep):
    # print(f"muralis function is happenning")
    # print(f"current_formation: {current_formation}")
    # print(f"orientations: {orientations}")
    # print(f"orientation_to_historical_scores: {orientation_to_historical_scores}")
    # print(f"orientation_to_current_scores: {orientation_to_current_scores}")
    # print(f"orientation_to_historical_counts: {orientation_to_historical_counts}")
    # print(f"orientation_to_current_counts: {orientation_to_current_counts}")
    # input("enter to continue")

    def remove_orientations_from_formation(orientations_to_be_removed, formation):
        output_formation = []
        orientations_to_be_removed = set(orientations_to_be_removed)
        for o in formation:
            if o not in orientations_to_be_removed:
                output_formation.append(o)
        return output_formation
    
    def add_orientations_to_formation(orientations_to_be_added, formation):
        for o in orientations_to_be_added:
            formation.append(o)
        return list(set(formation))

    def distance_from_point_to_line(x1, y1, x2, y2, x3, y3):
        if x1 == x2 and y1 == y2: # distance between two points
            return ((y3 - y1)**2 + (x3 - x1)**2)**0.5
        # print(f"x1 is {x1}, y1 is {y1}")
        # print(f"x2 is {x2}, y2 is {y2}")
        # print(f"x3 is {x3}, y3 is {y3}")
        
        # method 1
        # A is x1, y1
        # B is x2, y2
        # C is x3, y3
        # calculates the shortest distance from a point C to a line defined by two points A and B. The input points A, B, and C are represented by their x and y coordinates.
        # The code first calculates the difference between the x and y coordinates of points A and B to obtain the direction vector of the line. Then it calculates the projection of the vector from point A to point C onto the direction vector of the line, and stores it in the variable u.
        # Next, the code checks if u is outside the range of 0 to 1. If u is greater than 1, it means that the closest point on the line is beyond point B, so the code sets u to 1. If u is less than 0, it means that the closest point on the line is before point A, so the code sets u to 0.
        # Finally, the code calculates the closest point on the line to point C by using u to find the linear combination of the direction vector and point A, and stores the result in variables x and y. 
        # The code then calculates the difference between the x and y coordinates of the closest point on the line and point C, and finds the Euclidean distance between the two points. 
        # The result of this calculation is returned as the shortest distance from point C to the line.
        px = x2 - x1
        py = y2 - y1
        something = px * px + py * py
        # print(f"something is {something}")
        u = ((x3 - x1) * px + (y3 - y1) * py) / float(something)
        if u > 1:
            u = 1
        elif u < 0:
            u = 0
        x = x1 + u * px
        y = y1 + u * py
        dx = x - x3
        dy = y - y3
        # print(f"(dx * dx + dy * dy) is {(dx * dx + dy * dy)}")
        dist = (dx * dx + dy * dy)**0.5
        # input(f"returning {dist}")
        return dist

        # method 2 (simpler)
        # # https://stackoverflow.com/questions/39840030/distance-between-point-and-a-line-from-two-points
        # p1 = np.asarray((x1, y1))
        # p2 = np.asarray((x2, y2))
        # p3 = np.asarray((x3, y3))
        # return np.abs(np.cross(p2-p1, p1-p3)) / np.linalg.norm(p2-p1)

    # for the given orientation, get all 8 neighbors
    # and return the list of neighbors that are currently not in the shape
    # but are still within the bounds of the region
    def get_neighbors_outside_shape(orientation, current_formation, all_orientations):
        all_neighbors = get_all_neighbors(orientation)
        current_orientations_set = set(current_formation)
        all_orientations_set = set(all_orientations)
        potential_candidate_neighbors = []

        for neighbor in all_neighbors:
            if neighbor not in current_orientations_set:
                if neighbor in all_orientations_set:
                    potential_candidate_neighbors.append(neighbor)

        return potential_candidate_neighbors
    
    def can_expand_right(num_prior_expansions_to_this_right, lower_score, higher_score):
        if lower_score == 0.0:
            return True
        else:
            ratio = higher_score/lower_score
            if num_prior_expansions_to_this_right == 0:
                return True if ratio >= 1.1 else False
            elif num_prior_expansions_to_this_right == 1: 
                # don't allow expansions on same node repeatedly by setting unrealistic target
                return True if ratio >= 2 else False
            else:
                return True if ratio >= 2.5 else False

    def can_swap_right(num_prior_expansions_to_this_right, lower_score, higher_score):
        if lower_score == 0.0:
            return True
        else:
            ratio = higher_score/lower_score
            if num_prior_expansions_to_this_right == 0:
                return True if ratio >= 1.5 else False
            elif num_prior_expansions_to_this_right == 1:
                return True if ratio >= 1.75 else False
            else:
                return True if ratio >= 2 else False

    def only_added_orientation_last_frame(orientation, orientation_to_visited_step_numbers, current_step_num):
        if orientation not in orientation_to_visited_step_numbers:
            # haven't seen before
            return True
        else:
            if len(orientation_to_visited_step_numbers[o]) == 1:
                # have seen once before so it must be newly added and this is it's second frame in shape
                return True
            else:
                last_visit = orientation_to_visited_step_numbers[o][-1]
                penultimate_visit = orientation_to_visited_step_numbers[o][-2]
                if last_visit - penultimate_visit > 1: # visited last time, but didn't visit before that => newly added last time
                    return True
                else:
                    return False

    def extrapolate_orientation(right_orientation, neighbor_used_for_this_extension, orientations):
        pan_1 = extract_pan(right_orientation)
        tilt_1 = extract_tilt(right_orientation)
        zoom_1 = extract_zoom(right_orientation)

        pan_2 = extract_pan(neighbor_used_for_this_extension)
        tilt_2 = extract_tilt(neighbor_used_for_this_extension)
        zoom_2 = extract_zoom(neighbor_used_for_this_extension)

        pan_3 = pan_2
        tilt_3 = tilt_2
        zoom_3 = zoom_2
        if pan_1 < pan_2:
            pan_3 = pan_2 + 30
        elif pan_1 > pan_2:
            pan_3 = pan_2 - 30
        if tilt_1 < tilt_2:
            tilt_3 = tilt_2 + 15
        elif tilt_1 > tilt_2:
            tilt_3 = tilt_2 - 15

        if pan_3 > 330:
            pan_3 = -330
        if pan_3 < -330:
            pan_3 = 330

        if tilt_3 < -30:
            tilt_3 = 30
        if tilt_3 > 30:
            tilt_3 = -30

        new_orientation = "{}-{}-{}".format(pan_3, tilt_3, zoom_3)
        
        return new_orientation if new_orientation in set(orientations) else neighbor_used_for_this_extension

    def get_coordinates_of_orientation_from_boxes(all_boxes):
        min_x, min_y, max_x, max_y = 0,0,0,0
        for box in all_boxes:
            (x1, y1, x2, y2) = box
            if x1 < min_x:
                min_x = x1
            if y1 < min_y:
                min_y = y1
            if x2 > max_x:
                max_x = x2
            if y2 > max_y:
                max_y = y2
        return min_x, min_y, max_x, max_y

    def get_center_of_list_of_boxes(all_boxes):
        min_x, min_y, max_x, max_y = get_coordinates_of_orientation_from_boxes(all_boxes)
        return (max_x - min_x) / 2, (max_y - min_y) / 2 


    def get_centroid_of_list_of_boxes(all_boxes):
        centroids = []
        for box in all_boxes:
            (x1, y1, x2, y2) = box
            centroid_x = (x1 + x2) / 2
            centroid_y = (y1 + y2) / 2
            centroids.append((centroid_x, centroid_y))
        centroid_x = sum(x for x, y in centroids) / len(centroids)
        centroid_y = sum(y for x, y in centroids) / len(centroids)
        return centroid_x, centroid_y

    def get_coordinates_of_line_corresponding_to_orientation_border(o, n, coordinates_of_o):
        min_x, min_y, max_x, max_y = coordinates_of_o
        # o is an orientation
        # n is another orientation that is adjacent
        # coordinates of o gives us the min_x,min_y max_x, max_y of o
        
        # find out which side of o n is on
        if extract_pan(o) == extract_pan(n):
            # n is on top or bottom of o
            current_tilt = extract_tilt(o)
            target_tilt = extract_tilt(n)
            if target_tilt > current_tilt:
                # n is above
                return min_x, min_y, max_x, min_y
            else:
                # n is below
                return min_x, max_y, max_x, max_y
        elif extract_tilt(o) == extract_tilt(n):
            # n is to the left or right of o
            current_pan = extract_pan(o)
            target_pan = extract_pan(n)
            if current_pan > target_pan:
                if current_pan - target_pan <= 180:
                    # Rotating left
                    return min_x, min_y, min_x, max_y
                # Rotating right
                return max_x, min_y, max_x, max_y
            else:
                if target_pan - current_pan <= 180:
                    # Rotating right
                    return max_x, min_y, max_x, max_y
                # Rotating left
                return min_x, min_y, min_x, max_y
        else:
            # n is diagonal
            current_tilt = extract_tilt(o)
            target_tilt = extract_tilt(n)
            current_pan = extract_pan(o)
            target_pan = extract_pan(n)
            if current_tilt < target_tilt:
                if current_pan < target_pan:
                    # n is at left bottom. tilt increases from bottom to top
                    return min_x, max_y, min_x, max_y
                else:
                    # n is at right bottom
                    return max_x, max_y, max_x, max_y
            else:
                if current_pan < target_pan:
                    # n is at top left
                    return min_x, min_y, min_x, min_y
                else:
                    # n is at top right
                    return max_x, min_y, max_x, min_y

        

    def fraction_of_box_areas_towards_neighbor(o, n, orientation_to_current_car_boxes, orientation_to_current_person_boxes):
        # we have an orientation and a neighbor
        # we know all the boxes within the orientation
        # if the boxes in o are in aggregate gathered nearer to n, n should get a boost
        # this function returns this metric. it returns 0 if the boxes are exactly 
        # in the center of o. a negative number of the boxes are away from n (more negative the farther away)
        # and a positive number if the boxes are closer to n (more positive if closer to n).

        # get centroid C of all boxes
        current_all_boxes = []
        if o in orientation_to_current_person_boxes:  
            current_all_boxes.extend(orientation_to_current_person_boxes[o])
        elif o in orientation_to_current_car_boxes:
            current_all_boxes.extend(orientation_to_current_car_boxes[o])
        current_all_boxes = [tuple(list(x)) for x in current_all_boxes]
        if len(current_all_boxes) == 0:
            return 0
        centroid_x, centroid_y = get_centroid_of_list_of_boxes(current_all_boxes)
        
        # get cumulative area of all boxes
        cumulative_area_of_all_boxes = evaluation_tools.get_nonoverlapping_sum_of_areas_of_list_of_bounding_boxes(current_all_boxes)

        # get center of o called c
        # center_x, center_y = get_center_of_list_of_boxes(current_all_boxes)
        center_x = 720.0
        center_y = 360.0
        
        neighbor_x1, neighbor_y1, neighbor_x2, neighbor_y2 = get_coordinates_of_line_corresponding_to_orientation_border(o, n, get_coordinates_of_orientation_from_boxes(current_all_boxes))
        # get distance of C to n
        distance_of_centroid_to_neighbor = distance_from_point_to_line(neighbor_x1, neighbor_y1, neighbor_x2, neighbor_y2, centroid_x, centroid_y)
        
        # get distance of c to n
        distance_of_center_to_neighbor = distance_from_point_to_line(neighbor_x1, neighbor_y1, neighbor_x2, neighbor_y2, center_x, center_y)

        # in addition to distance, we also account for avg area of bounding box. we take average here otherwise one large object can dominate (i.e. we would not be fair towards count queries. considering area is for detect, and considering num bounding boxes is for count)
        # return ((distance_of_center_to_neighbor-distance_of_centroid_to_neighbor)/distance_of_center_to_neighbor) * (cumulative_area_of_all_boxes / len(current_all_boxes))
        
        # option 2: didn't work if i added cumulative area
        if distance_of_center_to_neighbor == 0:
            return 0
        return ((distance_of_center_to_neighbor-distance_of_centroid_to_neighbor)/distance_of_center_to_neighbor)


    def is_formation_of_orientations_feasible(potential_set_of_shapes, all_orientations):
        # TODO: later we'll also want to determine ideal location to start and end

        # TODO: we now assume 500 degrees per second we can visit all the shapes for sure
        # contiguous if each orientation has at least one other neighbor within the shape
        # set_of_nodes_to_explore = copy.deepcopy(potential_set_of_shapes)
        if len(potential_set_of_shapes) == 0:
            return True 
        contiguous=False
        for starting_node in potential_set_of_shapes:
            set_of_nodes_to_visit_to_reach_everyone = copy.deepcopy(potential_set_of_shapes)
            set_of_nodes_to_visit = []
            set_of_nodes_to_visit.append(starting_node)
            while len(set_of_nodes_to_visit) > 0:
                if len(set_of_nodes_to_visit_to_reach_everyone) == 0:
                    # print(f"returning True feasibility for {potential_set_of_shapes}")
                    contiguous = True
                    break
                current_position = set_of_nodes_to_visit[0]
                set_of_nodes_to_visit.pop(0)
                if current_position in set_of_nodes_to_visit_to_reach_everyone:
                    set_of_nodes_to_visit_to_reach_everyone.remove(current_position)
                    neighbors = get_neighbors_outside_shape(current_position, [], all_orientations)    
                    for n in neighbors:
                        set_of_nodes_to_visit.append(n)
            if len(set_of_nodes_to_visit_to_reach_everyone) == 0:
                # print(f"returning True feasibility for {potential_set_of_shapes}")
                contiguous = True
                break

        MST_less_than_500 = len(potential_set_of_shapes) <= num_frames_to_keep

#        MST_gt_lb = len(potential_set_of_shapes) >= LOWER_ORIENTATION_BOUND
 #       if contiguous and MST_less_than_500:# and MST_gt_lb:
  #          from dfs_helper import find_least_cost_path
   #         path, cost = find_least_cost_path(potential_set_of_shapes)
    #        MST_less_than_500 = cost <= 200
            

            
        # print(f"returning False feasibility for {potential_set_of_shapes}, orientations: {all_orientations}")
        return contiguous and MST_less_than_500

    
    original_formation = current_formation

    # print(f"*****INPUT shape is {current_formation} and its feasibility is {is_formation_of_orientations_feasible(current_formation, orientations)}")
    scores_and_deltas_used = {}
    
    # for each peek orientation, decide whether to keep it or remove it
    avg_mike_factor = statistics.mean(list(orientation_to_current_mike_factor.values()))
    # print(f"{step_number}: clearing peek")
    for o in peek_orientations:
        if (o in orientation_to_current_mike_factor and orientation_to_current_mike_factor[o] < avg_mike_factor/4) or (o not in orientation_to_current_mike_factor): 
            if len(current_formation) >= num_frames_to_keep + 1:
                current_formation_as_set = set(current_formation)
                current_formation_as_set.discard(o)
                current_formation = list(current_formation_as_set)
    peek_orientations.clear()

    # use the current counts to indicate how much we'd benefit by staying in 
    # the same formation
    reward_from_current_orientations = copy.deepcopy(orientation_to_current_mike_factor)
    current_formation = sorted(current_formation, key=lambda x: reward_from_current_orientations[x], reverse=True)


    # Remove extra items from formation if current_formation is not feasible
    while not is_formation_of_orientations_feasible(current_formation, orientations):
        if len(current_formation) <= num_frames_to_keep:
            break
        if len(current_formation) <= num_frames_to_keep+ 1:
            break
        current_formation.pop(-1)

    while len(current_formation) > num_frames_to_keep and len(current_formation) >= num_frames_to_keep + 1:
        current_formation.pop(-1)

    current_best_scoring_o = current_formation[0]
    orientation_to_count_factor = {}
    for o in reward_from_current_orientations:
        if o not in orientation_to_historical_counts or len(orientation_to_historical_counts[o]) < 4:
            orientation_to_count_factor[o] = 0.3 * orientation_to_current_mike_factor[o]
        else:
            # did we visit it 4 times within the last 30 frames?
            if step_number - orientation_to_visited_step_numbers[o][-4] < 30:
                c1 = orientation_to_historical_counts[o][-1]
                c2 = orientation_to_historical_counts[o][-2]
                c3 = orientation_to_historical_counts[o][-3]
                c4 = orientation_to_historical_counts[o][-4]
                orientation_to_count_factor[o] = (c1-c2) * 0.7 + (c2-c3) * 0.2 + (c3-c4) * 0.1
            else:
                # dont use the historical count as it is stale
                orientation_to_count_factor[o] = 0.3 * orientation_to_current_mike_factor[o]
    orientation_to_potential = {}
    for o, v in reward_from_current_orientations.items():
        orientation_to_potential[o] = (0.3 * orientation_to_current_mike_factor[o]) + (0.7 * orientation_to_count_factor[o]) 
        # print(f"step:{step_number}. o:{o}, (0.3 * orientation_to_current_mike_factor[o]):{(0.3 * orientation_to_current_mike_factor[o])}, (0.7 * orientation_to_count_factor[o]):{(0.7 * orientation_to_count_factor[o]) }")
    
    # print(f"sorting current formation {current_formation} by mike factor alone")
    # current_formation = sorted(current_formation, key=lambda x: orientation_to_current_mike_factor[x])

    
    current_formation = sorted(current_formation, key=lambda x: orientation_to_potential[x])
    # print(f"step:{step_number} sorting current formation {current_formation} by potential instead of mike factor alone")
    

    orientation_to_candidate_neighbors = {}
    for o in orientation_to_potential:
        orientation_to_candidate_neighbors[o] = get_neighbors_outside_shape(o, current_formation, orientations)    

    neighbor_to_viability_score = {}
    neighbor_to_number_of_touching_boundary_objects = {}
    for o in orientation_to_potential:
        neighbors = orientation_to_candidate_neighbors[o]
        for n in neighbors:
            if n not in neighbor_to_viability_score:
                neighbor_to_viability_score[n] = 0
            if n not in neighbor_to_number_of_touching_boundary_objects:
                neighbor_to_number_of_touching_boundary_objects[n] = 0
            neighbor_to_number_of_touching_boundary_objects[n] += 1
            neighbor_to_viability_score[n] += fraction_of_box_areas_towards_neighbor(o, n, orientation_to_current_car_boxes, orientation_to_current_person_boxes)
    # input(f"neighbor to viability score is {neighbor_to_viability_score}")
    for o in orientation_to_potential:
        orientation_to_candidate_neighbors[o] = sorted(orientation_to_candidate_neighbors[o], key=lambda x: neighbor_to_viability_score[x] * neighbor_to_number_of_touching_boundary_objects[x], reverse=True)
    


    scores_and_deltas_used = {
        "current shape scores": reward_from_current_orientations,
        "potential for border": orientation_to_potential,
        "orientation_to_current_mike_factor": orientation_to_current_mike_factor 
    }

    left_index = 0
    right_index = len(current_formation) - 1
    continue_modifying_shape = True
    num_prior_expansions_to_this_right = 0
    number_of_neighbors_used_in_this_right_index = 0
    number_of_changes = 0
    set_of_orientations_in_new_shape = set(current_formation)
    orientations_to_number_of_expansions = {}
    
    while left_index < right_index and continue_modifying_shape:
        if current_formation[left_index] == current_best_scoring_o:
            left_index += 1
        if only_added_orientation_last_frame(current_formation[left_index], orientation_to_visited_step_numbers, step_number):
            left_index += 1
        if left_index >= right_index:
            break
        # print(f"left index is {left_index}, right_index is {right_index}")
        # print(f"set_of_orientations_in_new_shape is {set_of_orientations_in_new_shape}")
        lower_score = reward_from_current_orientations[current_formation[left_index]]
        higher_score = reward_from_current_orientations[current_formation[right_index]]
        # print(f"lower_score={lower_score}, higher_score={higher_score}")
        if not can_swap_right(num_prior_expansions_to_this_right, lower_score, higher_score):
            # print(f"can_swap_right with {num_prior_expansions_to_this_right} is false")
            # print("1")
            if False and can_expand_right(num_prior_expansions_to_this_right, lower_score, higher_score):
                # not a big enough threshold to remove old one confidently and add 
                # but is there some gap between higher and lowest to warrant exploration 
                # near the higher?
                print('3')
                if len(orientation_to_candidate_neighbors[current_formation[right_index]]) > num_prior_expansions_to_this_right:
                    potential_new_shape = copy.deepcopy(set_of_orientations_in_new_shape)
                    is_new_shape_feasible = True
                    number_of_neighbors_used_in_this_right_index = 0
                    while orientation_to_candidate_neighbors[current_formation[right_index]][number_of_neighbors_used_in_this_right_index] in potential_new_shape:
                        number_of_neighbors_used_in_this_right_index += 1
                        if number_of_neighbors_used_in_this_right_index >= len(orientation_to_candidate_neighbors[current_formation[right_index]]):
                            break
                    if number_of_neighbors_used_in_this_right_index >= len(orientation_to_candidate_neighbors[current_formation[right_index]]):
                        is_new_shape_feasible = False
                        break
                    if number_of_neighbors_used_in_this_right_index < len(orientation_to_candidate_neighbors[current_formation[right_index]]):
                        potential_new_shape.add(orientation_to_candidate_neighbors[current_formation[right_index]][number_of_neighbors_used_in_this_right_index])
                    if (is_new_shape_feasible) and is_formation_of_orientations_feasible(potential_new_shape, orientations):
                        num_prior_expansions_to_this_right += 1
                        right_orientation = current_formation[right_index]
                        if right_orientation not in orientations_to_number_of_expansions:
                            orientations_to_number_of_expansions[right_orientation] = 0
                        orientations_to_number_of_expansions[right_orientation] += 1
                        number_of_changes += 1
                        set_of_orientations_in_new_shape = copy.deepcopy(potential_new_shape)
                    else:
                        # couldn't swap right
                        # couldn't expand right
                        right_index -= 1
                        number_of_neighbors_used_in_this_right_index = 0
                        num_prior_expansions_to_this_right = 0
                
            else:
                # couldn't swap right
                # couldn't expand right
                right_index -= 1
                number_of_neighbors_used_in_this_right_index = 0
                num_prior_expansions_to_this_right = 0
        else:
            # print(f"can_swap_right with {num_prior_expansions_to_this_right} is true")
            if len(orientation_to_candidate_neighbors[current_formation[right_index]]) > num_prior_expansions_to_this_right:
                # print(f"when trying to expand: {set_of_orientations_in_new_shape}")
                potential_new_shape = copy.deepcopy(set_of_orientations_in_new_shape)
                is_new_shape_feasible = True
                # print(f"removing {current_formation[left_index]}")
                potential_new_shape.remove(current_formation[left_index])
                number_of_neighbors_used_in_this_right_index = 0
                while orientation_to_candidate_neighbors[current_formation[right_index]][number_of_neighbors_used_in_this_right_index] in potential_new_shape:
                    # print(f"adding {orientation_to_candidate_neighbors[current_formation[right_index]][number_of_neighbors_used_in_this_right_index]}")
                
                    # print(f"but {orientation_to_candidate_neighbors[current_formation[right_index]][number_of_neighbors_used_in_this_right_index]} is in {potential_new_shape}")
                    number_of_neighbors_used_in_this_right_index += 1
                    if number_of_neighbors_used_in_this_right_index >= len(orientation_to_candidate_neighbors[current_formation[right_index]]):
                        is_new_shape_feasible = False
                        break
                    # print(f"going to try if {orientation_to_candidate_neighbors[current_formation[right_index]][number_of_neighbors_used_in_this_right_index]} is in {potential_new_shape}")
                neighbor_used_for_this_extension = None
                if number_of_neighbors_used_in_this_right_index < len(orientation_to_candidate_neighbors[current_formation[right_index]]):
                    neighbor_used_for_this_extension = orientation_to_candidate_neighbors[current_formation[right_index]][number_of_neighbors_used_in_this_right_index]
                
                    potential_new_shape.add(neighbor_used_for_this_extension)
                # print(f"when evaluating {potential_new_shape}, is_new_shape_feasible is {is_new_shape_feasible}")
                if (is_new_shape_feasible) and is_formation_of_orientations_feasible(potential_new_shape, orientations):
                    # print(f"step:{step_number} removing {current_formation[left_index]} adding neighbor of {current_formation[right_index]}")
                    # print(f"{lower_score}, {higher_score}")
                    num_prior_expansions_to_this_right += 1
                    left_index += 1
                    right_orientation = current_formation[right_index]
                    if right_orientation not in orientations_to_number_of_expansions:
                        orientations_to_number_of_expansions[right_orientation] = 0
                    orientations_to_number_of_expansions[right_orientation] += 1
                    number_of_changes += 1
                    set_of_orientations_in_new_shape = copy.deepcopy(potential_new_shape)
                    if neighbor_used_for_this_extension is not None and neighbor_to_number_of_touching_boundary_objects[neighbor_used_for_this_extension] >= 2:
                        # do a second hop
                        # print(f"********")
                        # print(f"********")
                        # print(f"neighbor_to_viability_score[neighbor_used_for_this_extension]:{neighbor_to_viability_score[neighbor_used_for_this_extension]}")
                        second_hop_neighbor = extrapolate_orientation(right_orientation, neighbor_used_for_this_extension, orientations)
                        # print(f"r:{right_orientation}, n:{neighbor_used_for_this_extension}, s:{second_hop_neighbor}")
                        potential_new_shape.add(second_hop_neighbor)
                        if is_formation_of_orientations_feasible(potential_new_shape, orientations):
                            set_of_orientations_in_new_shape = copy.deepcopy(potential_new_shape)
                            # print(f"{step_number}: added peek {second_hop_neighbor}")
                            peek_orientations.add(second_hop_neighbor)
                    # print(f"it is feasible so updated shape to be {set_of_orientations_in_new_shape}")
                else:
                    # print(f"it is infeasible so set_of_orientations_in_new_shape remains {set_of_orientations_in_new_shape}")
                    right_index -= 1
                    num_prior_expansions_to_this_right = 0
                    number_of_neighbors_used_in_this_right_index = 0
                    # print(f"shape is infeasible")
            else: 
                # print(f"right has no more neighbors")
                right_index -= 1
                num_prior_expansions_to_this_right = 0
                number_of_neighbors_used_in_this_right_index = 0
        if (not is_formation_of_orientations_feasible(set_of_orientations_in_new_shape, orientations)):
            # print(f"stop modifying shape due to too many changes or infeasible shape")
            continue_modifying_shape = False
        # add a dampener on too many hasty changes
        if step_number <= 3 and len(set(set_of_orientations_in_new_shape) - set(current_formation)) >= 1:
            continue_modifying_shape = False


    current_formation = list(set_of_orientations_in_new_shape)
    # print(f"current formation is {current_formation}")
    # current_formation = reset_zoom_factors(current_formation, anchor_orientation)
    current_formation, zoom_explorations_in_progress = add_zoom_factors(current_formation, orientation_to_current_car_boxes, orientation_to_current_person_boxes, zoom_explorations_in_progress)
    # input(f"after adding zoom factors {current_formation}")
    for o in current_formation:
        if o not in orientation_to_visited_step_numbers:
            orientation_to_visited_step_numbers[o] = []
        orientation_to_visited_step_numbers[o].append(step_number)
    

#    plus_formation = []
#    plus_formation.append(anchor_orientation)
#    plus_formation.append(rotate_down(anchor_orientation, orientations))
#    plus_formation.append(rotate_left(anchor_orientation, orientations))
#    plus_formation.append(rotate_right(anchor_orientation, orientations))
#    plus_formation.append(rotate_up(anchor_orientation, orientations))
#    for p in plus_formation:
#        if len(current_formation) < num_frames_to_keep and  p not in current_formation:
#            current_formation.append(p)

    formation_to_use = current_formation                
    return original_formation, current_formation, formation_to_use, scores_and_deltas_used, step_number + 1, zoom_explorations_in_progress



# this returns the weighted sum of car and person count. weight based on queries in workload. 
# if workload contains no people, this returns count of cars
# if workload contains no cars, this returns count of people 
# if workload contains half people and half cars, this return weight sum of both with weight = 0.5
def get_count_of_orientation(workload, 
                     current_frame,
                     current_orientation,
                     orientations,
                     frame_to_model_to_orientation_to_car_count,
                     frame_to_model_to_orientation_to_person_count,
                     ):
    num_car_queries = 0
    num_person_queries = 0
    car_count = 0
    person_count = 0
    for q in workload:
        if q[2] == 'car':
            num_car_queries += 1
        elif q[2] == 'person':
            num_person_queries += 1
    
    car_count = frame_to_model_to_orientation_to_car_count[current_frame][q[0]][current_orientation]
    person_count = frame_to_model_to_orientation_to_person_count[current_frame][q[0]][current_orientation]

    car_weight = (num_car_queries / (num_person_queries + num_car_queries))
    person_weight = (num_person_queries / (num_person_queries + num_car_queries))
    return ( (car_weight * car_count) + (person_weight * person_count) )


def main():

    num_frames_to_send = 1
    num_frames_to_keep = 4
    orientations = generate_all_orientations()
    anchor_orientation = '180-0-1'
    workload =  [('yolov4', 'count', 'car'),]

    url = "http://128.112.34.129/cgi-bin/ptzctrl.cgi"

    # Left, middle/up, right, middle/down, middle
    movements = ("?ptzcmd&abs&24&20&FF80&0000", "?ptzcmd&abs&24&20&FE80&0070", "?ptzcmd&abs&24&20&FD80&0000", "?ptzcmd&abs&24&20&FE80&FF70", "?ptzcmd&abs&24&20&FE80&0000" )
    trace = [
    ]
    for i in range(0, 100):
        trace.append(movements)

    parser = argparse.ArgumentParser(description="Locate objects in a live camera stream using an object detection DNN.", 
                                     formatter_class=argparse.RawTextHelpFormatter, 
                                     epilog=detectNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage())

    parser.add_argument("input", type=str, default="", nargs='?', help="URI of the input stream")
    parser.add_argument("--network", type=str, default="ssd-mobilenet-v2", help="pre-trained model to load (see below for options)")
    parser.add_argument("--overlay", type=str, default="box,labels,conf", help="detection overlay flags (e.g. --overlay=box,labels,conf)\nvalid combinations are:  'box', 'labels', 'conf', 'none'")
    parser.add_argument("--threshold", type=float, default=0.5, help="minimum detection threshold to use") 

    try:
        args = parser.parse_known_args()[0]
    except:
        print("")
        parser.print_help()
        sys.exit(0)




    orientation_to_frames_since_last_visit = {}
    num_aggregate_queries = evaluation_tools.num_aggregate_queries_in_workload(workload)


    running_non_aggregate_accuracy = 0.0


    model_to_object_ids_found = {}
    orientation_to_historical_scores = {}
    orientation_to_historical_counts = {}

    car_query_weight = evaluation_tools.num_car_queries_in_workload(workload) / len(workload)
    person_query_weight = 1.0 - car_query_weight

    current_formation = []
    static_cross_formation = [anchor_orientation, 
                                rotate_down(anchor_orientation, orientations),
                                rotate_left(anchor_orientation, orientations),
                                rotate_right(anchor_orientation, orientations),
                                rotate_up(anchor_orientation, orientations)]
    print('cross ', static_cross_formation)
    current_formation = static_cross_formation
    trace_of_our_orientations = []
    trace_of_best_dynamic_orientations = []
    trace_of_static_cross_formation = []
    trace_of_regions_we_chose = []
    trace_of_distance_between_us_and_best_dynamic = []
    orientation_to_visited_step_numbers = {}
    peek_orientations = set()
    step_num = 1
    zoom_explorations_in_progress = {}

    ranks = []



    frame_num = 0

    # load the object detection network
    net = detectNet(args.network, sys.argv, args.threshold)


    # to properly know which camera to use, make sure to do `ls /dev | grep video`.
    # the input to videocapture is the index corresponding to the connected camera
    cap = cv2.VideoCapture(0)

    orientation_to_visits = {}

    
    frame_to_model_to_orientation_to_person_count = {}
    frame_to_model_to_orientation_to_car_count = {}
    frame_to_model_to_orientation_to_cars_detected = {}
    frame_to_model_to_orientation_to_people_detected = {}
    for movements in trace:

        model_to_orientation_to_efficientdet_car_count = {}
        model_to_orientation_to_efficientdet_person_count = {}
        model_to_orientation_to_efficientdet_cars_detected = {}
        model_to_orientation_to_efficientdet_people_detected = {}


        model_to_orientation_to_efficientdet_person_count['yolov4'] = {}
        model_to_orientation_to_efficientdet_car_count['yolov4'] = {}
        model_to_orientation_to_efficientdet_cars_detected['yolov4'] = {}
        model_to_orientation_to_efficientdet_people_detected['yolov4'] = {}

        for m_idx,m in enumerate(movements):

            if m_idx >= len(current_formation):
                break 
            ret, frame = cap.read()
            img = Image.fromarray(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            cuda_mem = cudaFromNumpy(frame)
            detections = net.Detect(cuda_mem, overlay=args.overlay)
            print(len(detections), ' detected')
           
            if current_formation[m_idx] not in model_to_orientation_to_efficientdet_person_count['yolov4']:
                model_to_orientation_to_efficientdet_person_count['yolov4'][current_formation[m_idx]] = 0
            if current_formation[m_idx] not in model_to_orientation_to_efficientdet_people_detected['yolov4']:
                model_to_orientation_to_efficientdet_people_detected['yolov4'][current_formation[m_idx]] = []

            if current_formation[m_idx] not in model_to_orientation_to_efficientdet_car_count['yolov4']:
                model_to_orientation_to_efficientdet_car_count['yolov4'][current_formation[m_idx]] = 0
            if current_formation[m_idx] not in model_to_orientation_to_efficientdet_cars_detected['yolov4']:
                model_to_orientation_to_efficientdet_cars_detected['yolov4'][current_formation[m_idx]] = []
            for idx,d in enumerate(detections):
                cv2.rectangle(frame,(int(d.Left),int(d.Top)),(int(d.Right),int(d.Bottom)),(0,255,0),2)
                cv2.putText(frame,coco_classes[d.ClassID],(int(d.Right)+10,int(d.Bottom)),0,0.3,(0,255,0))
                if d.ClassID == 1:

                    model_to_orientation_to_efficientdet_person_count['yolov4'][current_formation[m_idx]] += 1
                    model_to_orientation_to_efficientdet_people_detected['yolov4'][current_formation[m_idx]].append([d.Left, d.Top, d.Right, d.Bottom]) 
                elif d.ClassID == 3:

                    model_to_orientation_to_efficientdet_car_count['yolov4'][current_formation[m_idx]] += 1
                    model_to_orientation_to_efficientdet_cars_detected['yolov4'][current_formation[m_idx]].append([d.Left, d.Top, d.Right, d.Bottom]) 

    #            roi=frame[int(d.Top):int(d.Bottom),int(d.Left):int(d.Right)]
    #            cv2.imwrite(str(idx) + '.jpg', roi)

            command = url + m
            time.sleep(0.3)
            cv2.imshow('Video stream ', frame)
        frame_to_model_to_orientation_to_person_count[frame_num] = model_to_orientation_to_efficientdet_person_count
        frame_to_model_to_orientation_to_car_count[frame_num] = model_to_orientation_to_efficientdet_car_count
        frame_to_model_to_orientation_to_cars_detected[frame_num] = model_to_orientation_to_efficientdet_cars_detected
        frame_to_model_to_orientation_to_people_detected[frame_num] = model_to_orientation_to_efficientdet_people_detected



        when_was_an_orientation_seen_last = evaluation_tools.compute_when_an_orientation_was_last_visited(trace_of_our_orientations, current_formation)

        
        orientation_to_current_scores = {}
        orientation_to_current_counts = {}
        orientation_to_current_mike_factor = {}
        # ** EfficientDet inference **



        orientation_to_score = {}
        print('Current formation ', current_formation)
        for o in current_formation:

            score = evaluation_tools.get_mikes_mike_factor(workload, 
                                 frame_num,
                                 o,
                                 current_formation,
                                 model_to_orientation_to_efficientdet_car_count,
                                 model_to_orientation_to_efficientdet_person_count,
                                 model_to_orientation_to_efficientdet_cars_detected,
                                 model_to_orientation_to_efficientdet_people_detected,
                                 orientation_to_frames_since_last_visit,
                                orientation_to_visits)
            orientation_to_score[o] = score
            orientation_to_current_counts[o] = get_count_of_orientation(workload, frame_num, o, orientations,frame_to_model_to_orientation_to_car_count,
                frame_to_model_to_orientation_to_person_count)

        current_orientation_to_ranking = rank_orientations(orientation_to_score)
#        print('Current orientation to rank ', current_orientation_to_ranking)
        # ** End EffientDet stuff **

        ### GET ground truth values for eval

        orientation_to_actual_est_accuracies = {}

        sorted_orientations = []
        sorted_dict = {k: v for k, v in sorted(current_orientation_to_ranking.items(), key=lambda item: item[1] )}
        best_orientations = []
        for o in sorted_dict:
            best_orientations.append(o)
        min_visits = 100000
        current_orientation = best_orientations[0]



        # populate the scores for the current frame for each of the orientations


        # Update stats to study freqency of orientation vistis
        if current_orientation not in orientation_to_visits:
            orientation_to_visits[current_orientation] = 0
        orientation_to_visits[current_orientation] += 1

        # *********************


        for o in current_formation:
            orientation_to_current_mike_factor[o] = evaluation_tools.get_muralis_mike_factor(workload, 
                                    frame_num,
                                    o,
                                    when_was_an_orientation_seen_last[o],
                                    model_to_orientation_to_efficientdet_car_count,
                                    model_to_orientation_to_efficientdet_person_count,
                                    model_to_orientation_to_efficientdet_cars_detected,
                                    model_to_orientation_to_efficientdet_people_detected)

        model_to_use = 'faster-rcnn'
        if model_to_use not in model_to_orientation_to_efficientdet_people_detected:
            model_to_use = random.choice(list(model_to_orientation_to_efficientdet_people_detected))
        previous_formation, current_formation, formation_to_use, scores_and_deltas_used, step_num, zoom_explorations_in_progress = neighboring_orientations_delta_method_madeye(anchor_orientation, current_formation, orientations, orientation_to_historical_scores, orientation_to_score, orientation_to_current_counts, orientation_to_historical_counts, orientation_to_current_mike_factor, step_num, orientation_to_visited_step_numbers, peek_orientations, model_to_orientation_to_efficientdet_cars_detected[model_to_use], model_to_orientation_to_efficientdet_people_detected[model_to_use], zoom_explorations_in_progress, num_frames_to_keep )
        # print(f"in madeye zoom explorations is {json.dumps(zoom_explorations_in_progress, indent=2)}")
        # print(f"current formation is {current_formation}")
        # input(f"formation to use is {formation_to_use}")
        for o in orientation_to_score:
            if o not in orientation_to_historical_scores:
                orientation_to_historical_scores[o] = []
            orientation_to_historical_scores[o].append(orientation_to_score[o])
        for o in orientation_to_current_counts:
            if o not in orientation_to_historical_counts:
                orientation_to_historical_counts[o] = []
            orientation_to_historical_counts[o].append(orientation_to_score[o])
        orientation_to_current_counts.clear()



        orientation_idx = 0
        non_aggregate_accuracies = []
        while orientation_idx < min(len(best_orientations) , num_frames_to_send):
            # Send images to the server
            current_orientation = best_orientations[orientation_idx]
            print('\tSelected orientation to send ', current_orientation)
            orientation_idx += 1
        frame_num += 1



if __name__ == '__main__':
    main()

