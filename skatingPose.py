#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import PIL.Image as Image
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import cv2
import six
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import functools
import stomp
# from lib.config import config as cfg
# from lib.utils.nms_wrapper import nms
# from lib.utils.test import im_detect
# from lib.nets.resnet_v1 import resnetv1
# from lib.nets.vgg16 import vgg16
# from lib.utils.timer import Timer
import multiprocessing as mp
from multiprocessing import Queue
from queue import Queue as gQueue
from datetime import datetime
import time
import logging
import sys
import threading
# import stomp
from ctypes import c_wchar_p
import json
from collections import deque , Counter

# from object_detection.utils import ops as utils_ops

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
tf.logging.set_verbosity(tf.logging.INFO)

CLASSES = ('__background__',  # always index 0
           '旋转', '跳跃', '滑行')
COLOR = ('red', 'blue', 'yellow')
PATH_TO_CKPT = r'D:\work\ob_api\huahua_pose_model_faster_rcnn_resnet101_coco_2018_01_28\graph\frozen_inference_graph.pb'

def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color='red',
                               thickness=4,
                               display_str_list=(),
                               use_normalized_coordinates=True):
  """Adds a bounding box to an image.

  Bounding box coordinates can be specified in either absolute (pixel) or
  normalized coordinates by setting the use_normalized_coordinates argument.

  Each string in display_str_list is displayed on a separate line above the
  bounding box in black text on a rectangle filled with the input 'color'.
  If the top of the bounding box extends to the edge of the image, the strings
  are displayed below the bounding box.

  Args:
    image: a PIL.Image object.
    ymin: ymin of bounding box.
    xmin: xmin of bounding box.
    ymax: ymax of bounding box.
    xmax: xmax of bounding box.
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
    display_str_list: list of strings to display in box
                      (each to be shown on its own line).
    use_normalized_coordinates: If True (default), treat coordinates
      ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
      coordinates as absolute.
  """
  draw = ImageDraw.Draw(image)
  im_width, im_height = image.size
  if use_normalized_coordinates:
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                  ymin * im_height, ymax * im_height)
  else:
    (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
  draw.line([(left, top), (left, bottom), (right, bottom),
             (right, top), (left, top)], width=thickness, fill=color)
  try:
    # font = ImageFont.truetype('arial.ttf', 12)
    font = ImageFont.truetype(r'E:\work\faster-rcnn-tf-resnet101-test\Fonts\msyh.ttc', 22)
  except IOError:
    font = ImageFont.load_default()

  # If the total height of the display strings added to the top of the bounding
  # box exceeds the top of the image, stack the strings below the bounding box
  # instead of above.
  display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
  # Each display_str has a top and bottom margin of 0.05x.
  total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

  if top > total_display_str_height:
    text_bottom = top
  else:
    text_bottom = bottom + total_display_str_height
  # Reverse list and print from bottom to top.
  for display_str in display_str_list[::-1]:
    text_width, text_height = font.getsize(display_str)
    margin = np.ceil(0.05 * text_height)
    draw.rectangle(
        [(left, text_bottom - text_height - 2 * margin), (left + text_width,
                                                          text_bottom)],
        fill=color)
    draw.text(
        (left + margin, text_bottom - text_height - margin),
        display_str,
        fill='black',
        font=font)
    text_bottom -= text_height - 2 * margin
def draw_bboxes_to_image(image , detections):

    image_pil = Image.fromarray(image[:,:,[2,1,0]])
    for i in range(len(detections)):
        xmin ,ymin , xmax  , ymax , score ,lable = tuple(list(detections[i]))
        draw_bounding_box_on_image(image_pil,
                                   ymin,
                                   xmin,
                                   ymax,
                                   xmax,
                                   color=COLOR[int(lable)-1],
                                #    color='yellow',
                                   thickness=4,
                                   display_str_list=['{:s} {:.2f}%'.format(CLASSES[int(lable)], score*100), ],
                                #    display_str_list=['{:s} {:.2f}%'.format('运动员识别', score*100), ],
                                   use_normalized_coordinates=False)
    np.copyto(image, np.array(image_pil)[:,:,[2,1,0]])

class OBModel(object):
    def __init__(self, name, path_pb, gpu_index = 0, gpu_memory_fraction = 0):
        self.model_name = name

        self.gpu_memory_fraction = gpu_memory_fraction
        self.gpu_index = gpu_index
        self.tfmodel = path_pb
        self.min_score_thresh = 0.5 
        # # data filter params
        # self.is_draw_box = False
        # self.per_class_score_thresh = 0.9
        # self.per_class_NMS_thresh = 0.1
        # self.all_class_NMS_thresh = 0.5

        self.sess = self.load_model()
        print('the %s OBModel is loaded , model pb is %s '%(name, path_pb))

    
    def load_model(self):

        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.tfmodel, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        if self.gpu_memory_fraction == 0:
            gpu_options = tf.GPUOptions(allow_growth = True, visible_device_list = str(self.gpu_index) ) 
        else:
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = self.gpu_memory_fraction, visible_device_list = str(self.gpu_index) ) 
        tfconfig = tf.ConfigProto(device_count={'GPU': 1}, allow_soft_placement=True, gpu_options = gpu_options)
        sess = tf.Session(config= tfconfig, graph= detection_graph)

        with sess.as_default():
            with sess.graph.as_default():
                # Get handles to input and output tensors
                ops = tf.get_default_graph().get_operations()
                all_tensor_names = {
                    output.name for op in ops for output in op.outputs}
                tensor_dict = {}
                for key in ['num_detections', 'detection_boxes', 'detection_scores',
                            'detection_classes', 'detection_masks'
                            ]:
                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                            tensor_name)
                self.tensor_dict = tensor_dict
                
                self.image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
        return sess

    def run_recognition(self, frame ):
        # start_time = time.time()
        # frame (h*w*3 BGR) is read by cv2, image is (h*w*3 RGB)
        image = frame[:,:,[2,1,0]]
        # inference
        output_dict = self.sess.run(self.tensor_dict,
                            feed_dict={self.image_tensor: np.expand_dims(image, 0)})
        
        '''
            all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(
                output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.uint8)
            # ymin, xmin, ymax, xmax = box
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
        '''
        index = np.where(output_dict['detection_scores'][0] > self.min_score_thresh)[0]

        data_summary = {}
        # data_summary['bbox'] is xmin ,ymin , xmax  , ymax , score ,lable
        data_summary['bbox'] = np.concatenate((output_dict['detection_boxes'][0][:,[1,0,3,2]][index,:], 
                                               output_dict['detection_scores'][0][index, np.newaxis],
                                               output_dict['detection_classes'][0].astype(np.uint8)[index, np.newaxis] ), axis = 1)
        data_summary['bbox'][:,[0,2]] = data_summary['bbox'][:,[0,2]] * image.shape[1]
        data_summary['bbox'][:,[1,3]] = data_summary['bbox'][:,[1,3]] * image.shape[0]
        
        if 'detection_masks' in output_dict:
            ''' # to_do : output_dict['detection_masks'] neeed post process
                if 'detection_masks' in tensor_dict:
                    # The following processing is only for single image
                    detection_boxes = tf.squeeze(
                        tensor_dict['detection_boxes'], [0])
                    detection_masks = tf.squeeze(
                        tensor_dict['detection_masks'], [0])
                    # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                    real_num_detection = tf.cast(
                        tensor_dict['num_detections'][0], tf.int32)
                    detection_boxes = tf.slice(detection_boxes, [0, 0], [
                                            real_num_detection, -1])
                    detection_masks = tf.slice(detection_masks, [0, 0, 0], [
                                            real_num_detection, -1, -1])
                    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                        detection_masks, detection_boxes, image.shape[0], image.shape[1])
                    detection_masks_reframed = tf.cast(
                        tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                    # Follow the convention by adding back the batch dimension
                    tensor_dict['detection_masks'] = tf.expand_dims(
                        detection_masks_reframed, 0)
            '''
            data_summary['detection_masks'] = output_dict['detection_masks'][0]

        data_summary['people_num'] = len(index)
        data_summary['processed_image'] = frame
        # end_time = time.time()
        # print('detect speed is %s frames per second'%(1/(end_time - start_time)))
        return data_summary

class  VideoExtract():
    def __init__(self, name  ,gpu_arrange, video_source, pb_path, OB_Model, video_save_path = None, athele_num_dir = None):
        self.video_extracter_name = name
        self.video_source = video_source
        self.pb_path = pb_path
        self.detector_nums = len(gpu_arrange)
        self.gpu_arrange = list(gpu_arrange)
        self.OB_Model = OB_Model
        
        # video read process params
        self.detect_img_read_model = 0 # 0: realtime read ; num(>0): read one frame per nums frames
        self.video_param_dict = {}
        self.video_read_signal = mp.Value('i', 1) # 1:play 2:kill 3:suspend
        self.video_read_process_alive_status = mp.Value('i', 1)
        # videoclips save process params
        self.video_save_dir = mp.Value('i', 0)
        self.video_save_path = video_save_path
        self.athele_num_dir = athele_num_dir 
        self.video_save_process_alive_status = mp.Value('i', 1)

        self.video_save_queue_start = mp.Value('i', 1)
        self.video_save_queue_end = mp.Value('i', 1)
        self.video_save_queue = Queue()
        self.video_save_out_queue = Queue()
        # detector process params
        self.detector_process_dead_nums = mp.Value('i', 0)
        self.detection_process_status_array = mp.Array('i', self.detector_nums)
        for i in range(len(self.detection_process_status_array)):
            self.detection_process_status_array[i]= 0

        self.input_image_queue = Queue()
        self.queue_list = [ Queue(maxsize=5) for i in range( self.detector_nums)]
        # arrange process params
        self.video_arrange_process_alive_status = mp.Value('i', 1)
        self.record_array = mp.Array('i', 30)
        for i in range(len(self.record_array)):
            self.record_array[i]= -1

        self.processed_img_queue = Queue()
    
    def verify_video_source(self):
        video_capture = cv2.VideoCapture(self.video_source)
        if video_capture.isOpened():
            self.video_param_dict['FRAME_WIDTH'] = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.video_param_dict['FRAME_HEIGHT'] = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.video_param_dict['FRAME_FPS'] = int(video_capture.get(cv2.CAP_PROP_FPS))
            self.video_param_dict['FRAME_FOURCC'] = int(video_capture.get(cv2.CAP_PROP_FOURCC))
            # self.video_param_dict['FRAME_COUUNT'] = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            # print('video frame counts is %s , fps is %s '%(self.video_param_dict['FRAME_COUUNT'], self.video_param_dict['FRAME_FPS'] ), self.video_param_dict)
            video_capture.release()
            return True
        else:
            return False
    # delete save queue
    def video_reader(self):
        self.video_read_process_alive_status.value = 1
        cur_pid = os.getpid()
        print('%s : video reader start, pid is %s, video source is "%s" .'%(self.video_extracter_name , cur_pid , self.video_source ))

        detect_img_read_model = self.detect_img_read_model
        video_read_signal = self.video_read_signal
        video_capture = cv2.VideoCapture(self.video_source)
        if video_capture.isOpened():
            frame_count = 1
            ret, frame = video_capture.read()
            # self.video_save_queue.put((frame_count, frame))
            # self.video_save_queue_end.value = frame_count
            self.input_image_queue.put((frame_count, frame))
            while ret:
                while video_read_signal.value == 3:
                    time.sleep(1)
                    print('video reader is suspend' , end = '\r')
                # simulate the real time fps
                # cv2.waitKey(int(1000/7))
                ret, frame = video_capture.read()
                if frame is not None:
                    frame_count += 1
                    # input the video save queue
                    # self.video_save_queue.put((frame_count, frame))
                    # self.video_save_queue_end.value = frame_count

                    # input the image detect queue , process the detection image input mode 
                    if detect_img_read_model == 0 and self.input_image_queue.qsize() < 1:
                        self.input_image_queue.put((frame_count, frame))
                    elif detect_img_read_model == 1 :
                        self.input_image_queue.put((frame_count, frame))
                    elif detect_img_read_model > 1 and frame_count%detect_img_read_model == 1:
                        self.input_image_queue.put((frame_count, frame))
                    else:
                        pass
                if video_read_signal.value == 2:
                    break
            self.input_image_queue.put(0)
            print('video source "%s" is end or interrupted, the reader process (pid is %s ) killed'%(self.video_source, cur_pid))
        else:
            print('video source "%s" can not be open, the reader process (pid is %s ) killed'%(self.video_source, cur_pid))
        video_capture.release()
        self.video_read_process_alive_status.value = 0
        return None
        # os.kill(cur_pid , -9)    
    def detector_process(self, process_index, gpu_index):
        cur_pid = os.getpid()
        print("%s : detector process of %s start, the pid is %s"%(self.video_extracter_name ,process_index, cur_pid))
        self.detection_process_status_array[process_index] = 0
        try:
            obModel = self.OB_Model(name = 'OB_Model_'+str(process_index) , path_pb = self.pb_path , gpu_index = gpu_index)
        except:
            print('No.%s detector process is killed ,pid is %s, due to its OBModel can not be loaded.'%(process_index, cur_pid))
            self.detector_process_dead_nums.value += 1
            return None
        print('obModel of No.%s detector process is loaded, the pid is %s'%(process_index, cur_pid))
        self.detection_process_status_array[process_index] = 1
        detector_nums = self.detector_nums

        while 1:
            ele = self.input_image_queue.get()
            if isinstance(ele, int):
                if ele != detector_nums - 1:
                    print('No.%s detector end.'%(ele))
                    self.input_image_queue.put(ele + 1)
                    break
                elif ele == detector_nums - 1:
                    print('No.%s (all) detector end.'%(ele))
                    break 
            frame_count, frame = ele
            data_result = obModel.run_recognition(frame)
            # print('the detector %s process put out frame is %s'%(process_index, frame_count))
            self.queue_list[process_index].put((frame_count, data_result))
        self.detector_process_dead_nums.value += 1
        self.detection_process_status_array[process_index] = 0
        print('No.%s detector process is killed ,pid is %s, due to the video reader process is end'%(process_index, cur_pid))
        return None
        # os.kill(cur_pid , -9)
    def data_arrange(self):
        cur_pid = os.getpid()
        print('%s : detected img arrange process starts, pid is %s:'%(self.video_extracter_name, cur_pid))
        self.video_arrange_process_alive_status.value = 1
        
        now_time = datetime.now()
        logfile_name = now_time.strftime('%Y_%m_%d_%H_%M') + '_highlightStartEnd' +'.log'
        logfile_dir = './log/'
        if not os.path.isdir(logfile_dir):
            os.makedirs(logfile_dir)
        logging.basicConfig(filename=logfile_dir+logfile_name, filemode="w", level=logging.DEBUG)
        mylogger_1 = logging.getLogger('logger_highlightStartEnd')

        detected_count_num = 0
        start_time = 0
        end_time = 0
        time_interval = 50

        process_num = self.detector_nums
        data_list = []
        frame_num_list = np.zeros((process_num,), dtype = np.int32)

        cur_alive_queue = [i for i in range(process_num)]
        cur_alive_queue_temp = []
        cur_dead_queue = []
        cur_dead_queue_temp = []

        for i in range(process_num):
            data_list.append(self.queue_list[i].get())
            frame_num_list[i] = data_list[i][0]
        frame_min = frame_num_list.min()
        frame_min_index = np.argmin(frame_num_list)
        
        while 1:
            self.processed_img_queue.put((int(0), data_list[frame_min_index]))
            detected_count_num += 1
            if (detected_count_num-1)%time_interval == 0:
                if start_time == 0:
                    start_time = time.time()
                else:
                    end_time = time.time()
                    cost_time = end_time - start_time
                    start_time = end_time
                    print('Continue, the detected frames nums is %s , avg per sencond detected frames is %.2f'%(detected_count_num-1, (time_interval)/cost_time))

            if self.detector_process_dead_nums.value == process_num and self.queue_list[frame_min_index].qsize() == 0:
                break

            data_list[frame_min_index] = self.queue_list[frame_min_index].get()
            frame_num_list[frame_min_index] = data_list[frame_min_index][0]
            frame_min = frame_num_list[cur_alive_queue].min()
            frame_min_index = cur_alive_queue[np.argmin(frame_num_list[cur_alive_queue])]

            # data_list[frame_min_index] = self.queue_list[frame_min_index].get()
            # frame_num_list[frame_min_index] = data_list[frame_min_index][0]
            # frame_min = frame_num_list.min()
            # frame_min_index = np.argmin(frame_num_list)
        self.video_arrange_process_alive_status.value = 0
        print('the data arrange process is killed ,pid is %s, due to all detector processes is end. '%( cur_pid))
        return None
        # os.kill(cur_pid , -9)
    def send_processed_img(self):
        cur_pid = os.getpid()
        print('img send process starts, pid is %s.'%( cur_pid))

        processed_queue=self.processed_img_queue
        queue_maxsize = 0
        video_arrange_process_alive_status= self.video_arrange_process_alive_status
        """ def send_msg():
            # conn = stomp.Connection([('120.79.16.31', 61613)])  
            conn = stomp.Connection([('localhost', 61613)])  
            # conn.set_listener('', MyListener())
            conn.start()
            conn.connect('admin', 'admin', True) 
            return conn
        conn = send_msg() """

        count = 0
        time_interval = 50
        start_time = 0 
        while 1:
            if processed_queue.qsize() > queue_maxsize:
                (status,(frame_count,data_show)) = processed_queue.get()
                show_frame = draw_bboxes_to_image(data_show['processed_image'], data_show['bbox'])
                _ , img_encode = cv2.imencode('.jpg',show_frame)
                # conn.send(body=img_encode.tobytes(), destination='video',content_type = 'image/jpeg',headers={'id': frame_count,'status': status}) 
                count += 1
                if (count-1)%time_interval == 0:
                    if start_time == 0:
                        start_time = time.time()
                    else:
                        end_time = time.time()
                        cost_time = end_time - start_time
                        start_time = end_time
                        print('Continue, the send frame nums is %s , avg per sencond send frames is %.2f'%(count, (time_interval)/cost_time))
            else:
                if video_arrange_process_alive_status.value != 1:
                    break
                else: 
                    time.sleep(0.1)
        print('image send process is killed, pid is %s, due to the data arrange process is end. the last send frame num is %s, total send frames is %s. '%(cur_pid, frame_count, count ))
        return None
    def save_video(self):
        cur_pid = os.getpid()
        print('%s : video save process starts, pid is %s:'%(self.video_extracter_name , cur_pid))
        self.video_save_process_alive_status.value = 1

        now_time = datetime.now()
        logfile_name = now_time.strftime('%Y_%m_%d_%H_%M') + '_videoSave' +'.log'
        logfile_dir = './log/'
        if not os.path.isdir(logfile_dir):
            os.makedirs(logfile_dir)
        logging.basicConfig(filename=logfile_dir+logfile_name, filemode="w", level=logging.DEBUG)
        mylogger_2 = logging.getLogger('logger_videoSave')

        """ def send_msg():
            # conn = stomp.Connection([('120.79.16.31', 61613)])  
            # conn = stomp.Connection([('192.168.3.84', 61613)])  
            conn = stomp.Connection([('localhost', 61613)])  
            # conn.set_listener('', MyListener())
            conn.start()
            conn.connect('admin', 'admin', True) 
            return conn
        conn = send_msg() """

        FRAME_WIDTH =   self.video_param_dict['FRAME_WIDTH'] 
        FRAME_HEIGHT =  self.video_param_dict['FRAME_HEIGHT'] 
        FRAME_FPS =     self.video_param_dict['FRAME_FPS']
        # FRAME_FOURCC =  self.video_param_dict['FRAME_FOURCC'] 
        FRAME_FOURCC =  cv2.VideoWriter_fourcc(*'avc1')
        video_name_suf = '.mp4'

        clip_num = 0 
        max_queue_size = 3*FRAME_FPS
        saving_maxsize = 5*FRAME_FPS

        video_save_queue_start = self.video_save_queue_start
        record_array = self.record_array
        video_save_queue = self.video_save_queue
        video_save_out_queue = self.video_save_out_queue
        video_save_dir = self.video_save_dir
        video_save_path = self.video_save_path
        athelet_num_dir = self.athele_num_dir

        accumalate_frames_thresh = 0*FRAME_FPS
        is_accumulate_save = False
        cur_accumulate_frames = 0

        pre_athelet_num = ''

        while 1:
            frame_start = -1
            while 1:
                while video_save_queue.qsize() <= max_queue_size and self.video_read_process_alive_status.value == 1 and record_array[clip_num%20] == -1 and frame_start < 0:
                    mylogger_2.info('the saving queue is short, start frame has not occur , 1 second waiting , clip num is {}, queue start frame num is {} and the total nums is {}'.format(clip_num, video_save_queue_start.value, video_save_queue.qsize()))
                    time.sleep(1)
                    pass
                if frame_start < 0 and record_array[clip_num%20] != -1:
                    frame_start = record_array[clip_num%20]
                    mylogger_2.info('the start frame has occured , save start frame num is {} , clip num is {}'.format(frame_start, clip_num))
                    if frame_start < video_save_queue_start.value:
                        mylogger_2.info('WARNING the start frame is {} is not in save queue which start at {} ,total nums is {} and clip num is {} '.format(frame_start, video_save_queue_start.value, video_save_queue.qsize(), clip_num))
                        frame_start = video_save_queue_start.value

                if frame_start != -1 and video_save_queue_start.value >= frame_start:
                    clip_num += 1
                    break

                if self.video_read_process_alive_status.value == 0 and video_save_queue.qsize() == 0:
                    print('the video save process is killed, pid is %s , due to video reader process is end.'%( cur_pid))
                    self.video_save_process_alive_status.value = 0
                    # conn.disconnect()
                    return None
                data = video_save_queue.get()
                # video_save_out_queue.put((0,data))
                if data[0]%200 == 0:
                    print('Continue, the cur frame num is %s '%(data[0]))
                video_save_queue_start.value += 1
                # modify
                mylogger_2.info('SAVING wait, the start frame is {} , queue start frame num is {} and the total nums is {}'.format(frame_start, video_save_queue_start.value, video_save_queue.qsize()))
            
            if not is_accumulate_save:
                mylogger_2.info('SAVING start ,save start frame num is {} , clip num is {},  queue start frame num is {} and the total nums is {}'.format(frame_start, clip_num - 1, video_save_queue_start.value, video_save_queue.qsize()))
                
                """ video_num = clip_num//2
                video_class = record_array[((clip_num-1)%20)//2+20]
                video_name_class = 'video'
                if video_class == 1:
                    video_name_class = 'rotate'
                elif video_class == 2:
                    video_name_class = 'jump'
                    
                video_name = self.video_extracter_name + '_' + str(video_num)+'_'+ video_name_class +  video_name_suf
                mylogger_2.info('the status has been changed, video class is %s '%(video_name_class))
                if video_save_dir.value == 0:
                    video_dir = 'video_out/'
                else :
                    video_dir =  str(video_save_dir.value) +'/'
                video_dir = os.path.join(video_save_path , video_dir)
                if not os.path.isdir(video_dir):
                    os.makedirs(video_dir)
                print('save start ,video name "%s" save to dir "%s" '%(video_name , video_dir))
                video_writer = cv2.VideoWriter(video_dir+video_name,FRAME_FOURCC,FRAME_FPS,(FRAME_WIDTH,FRAME_HEIGHT)) """

                athelet_num_list = os.listdir(athelet_num_dir)
                if athelet_num_list:
                    athelet_num = athelet_num_list[0]
                else:
                    athelet_num = 'video_out'
                if pre_athelet_num != athelet_num:
                    pre_athelet_num = athelet_num
                    if not os.path.isdir(os.path.join(video_save_path, athelet_num)):
                        video_count = 0
                        os.mkdir(os.path.join(video_save_path, athelet_num))
                    else:
                        video_list = os.listdir(os.path.join(video_save_path, athelet_num))
                        if video_list:
                            video_num_list = [int(video_name.split(video_name_suf)[0]) for video_name in video_list]
                            video_count = max(video_num_list) + 1
                        else:
                            video_count = 0
                else:
                    video_count += 1
                video_name = str(video_count) + video_name_suf
                print('save start ,video name "%s" save to dir "%s" '%(video_name , athelet_num ))
                video_writer = cv2.VideoWriter(os.path.join( video_save_path, athelet_num, video_name),FRAME_FOURCC,FRAME_FPS,(FRAME_WIDTH,FRAME_HEIGHT))
            else:
                mylogger_2.info('SAVING accumulate ,save start frame num is {} , clip num is {},  queue start frame num is {} and the total nums is {}'.format(frame_start, clip_num - 1, video_save_queue_start.value, video_save_queue.qsize()))
            
            frame_end = -1
            while 1:
                while video_save_queue.qsize() <= saving_maxsize and self.video_read_process_alive_status.value == 1 and record_array[clip_num%20] == -1 and frame_end < 0:
                    mylogger_2.info('the saving queue is short, end frame has not occur , 1 second waiting , clip num is {}, queue start frame num is {} and the total nums is {}'.format(clip_num, video_save_queue_start.value, video_save_queue.qsize()))
                    time.sleep(1)
                    pass
                if frame_end < 0 and record_array[clip_num%20] != -1:
                    frame_end = record_array[clip_num%20]
                    if clip_num%20 ==19:
                        record_array[clip_num%20] = -1
                        
                    mylogger_2.info('end frame has occur, the end frame is {}, queue start frame num is {} and the total nums is {}'.format(frame_end ,video_save_queue_start.value, video_save_queue.qsize()))
                    
                if frame_end != -1 and frame_end < video_save_queue_start.value:
                    clip_num += 1
                    break
                if self.video_read_process_alive_status.value == 0 and video_save_queue.qsize() == 0:
                    video_writer.release()
                    print('the video save process is killed, pid is %s , due to video reader process is end.'%( cur_pid))
                    self.video_save_process_alive_status.value = 0
                    # conn.disconnect()
                    return None
                data = video_save_queue.get()
                video_save_queue_start.value += 1
                # video_save_out_queue.put((video_class,data))
                if data[0]%200 == 0:
                    print('Continue, the cur frame num is %s '%(data[0]))
                video_writer.write(data[1])
                # mosify
                mylogger_2.info('SAVING is on, save frame num is {} , the end frame num is {}, queue start frame is {}, total video queue is  {} '.format(data[0], frame_end, video_save_queue_start.value,  video_save_queue.qsize()))
            
            cur_accumulate_frames += (frame_end - frame_start + 1)
            if cur_accumulate_frames > accumalate_frames_thresh:
                video_writer.release()
                mylogger_2.info('SAVING end , file "{}" save done , save frames is {} '.format(video_name , max(cur_accumulate_frames, 0)))
                # conn.send(body='{"video_path":"'+'video_out/'+video_name+'","video_source":"vieo.mp4","video_class":"'+video_name_class +'"}', destination='videoFragment',content_type = 'text/plain',) 
                is_accumulate_save = False
                cur_accumulate_frames = 0 
            else:
                is_accumulate_save = True
                
                pass

            if clip_num%20 == 0:
                time.sleep(2)

    def get_out_queue(self):
        return self.video_save_out_queue,self.processed_img_queue
    def load_detector_model(self):
        self.detector_processers = []
        for i in range(self.detector_nums):
            process = mp.Process(target= self.detector_process, args= (i,self.gpu_arrange[i])) 
            process.start()
            self.detector_processers.append(process)
        status = 0
        while 1 :
            for i in range(self.detector_nums):
                status += self.detection_process_status_array[i]
            if status == self.detector_nums:
                return True
    def start_process(self):
        self.data_arrange_process = mp.Process(target= self.data_arrange) 
        self.data_arrange_process.start()
        # self.video_save_process = mp.Process(target= self.save_video) 
        # self.video_save_process.start()
        # self.send_process = mp.Process(target= self.send_processed_img) 
        # self.send_process.start()
        self.video_reader_process = mp.Process(target= self.video_reader) 
        self.video_reader_process.start()
    def initial_extractor(self, video_path):
        # self.video_extracter_name = name
        self.video_source = video_path
        # self.video_save_path = video_save_path
        # self.detect_img_read_model = 0
        # self.detector_nums = len(gpu_arrange)
        # self.gpu_arrange = list(gpu_arrange)
        # self.video_param_dict = {}

        self.video_save_dir.value = 0
        # print(self.video_save_dir.value)
        self.video_save_queue_start.value = 1
        self.video_save_queue_end.value = 1
        self.video_read_signal.value = 1
        self.video_read_process_alive_status.value = 1
        self.video_save_process_alive_status.value = 1
        self.video_arrange_process_alive_status.value = 1
        self.detector_process_dead_nums.value = 0
        
        # self.record_array = mp.Array('i', 30)
        for i in range(len(self.record_array)):
            self.record_array[i]= -1

        # self.detection_process_status_array = mp.Array('i', self.detector_nums)
        for i in range(len(self.detection_process_status_array)):
            self.detection_process_status_array[i]= 0

        # self.input_image_queue = Queue()
        # self.queue_list = [ Queue(maxsize=5) for i in range( self.detector_nums)]
        # self.processed_img_queue = Queue(maxsize =20)
        # self.video_save_queue = Queue()
        # self.video_save_out_queue = Queue()
        # self.huahua_model = huahua_model

    def monitor_process(self):
        self.video_reader_process.join()
        # self.video_save_process.join()
        for p in self.detector_processers:
            p.join()
        self.data_arrange_process.join()
        # self.send_process.join()
        return True

if __name__ == '__main__':
    cur_pid = os.getpid()
    print('main process id is %s:'%(cur_pid))
    # tf.logging.set_verbosity(tf.logging.INFO)
    '''
        FLAGS = tf.app.flags.FLAGS
        video_path = FLAGS.video_source
        video_save_path = FLAGS.video_save_dir
        tfmodel =  FLAGS.model_dir
        athele_num_dir = FLAGS.athele_num_dir
        database_parameters = FLAGS.database_conn.split(':')

        if video_path == '':
            raise IOError(('video source path has not been input'))
        print('video source is %s'%(video_path))
        video_name = video_path
        if video_save_path == '':
            video_save_path = './'
            print('video save path has not been input ,save to "./"')
        else:
            print('video will be saved to "%s"'%(video_save_path))
        if not os.path.isdir(video_save_path):
            os.makedirs(video_save_path)
        if not os.path.isfile(tfmodel + '.meta'):
            print(tfmodel)
            raise IOError(('{:s} not found.\nDid you download the proper networks from '
                        'our server and place them properly?').format(tfmodel + '.meta'))
    '''
    def origin_video_output(video_out_queue, queue_maxsize, window_name, show_model, is_draw_box, read_status, video_param_dict = None):
        open_status = False 
        print('start')

        if show_model == 2:
            FRAME_WIDTH =   video_param_dict['FRAME_WIDTH'] 
            FRAME_HEIGHT =  video_param_dict['FRAME_HEIGHT'] 
            FRAME_FPS =     video_param_dict['FRAME_FPS']
            # FRAME_FOURCC =  self.video_param_dict['FRAME_FOURCC'] 
            FRAME_FOURCC =  cv2.VideoWriter_fourcc(*'avc1')
            video_path = './video_save.mp4'
            video_writer = cv2.VideoWriter(video_path ,FRAME_FOURCC,FRAME_FPS,(FRAME_WIDTH,FRAME_HEIGHT))


        count = 0
        start_time = 0
        time_interval = 200
        sleep_time = 0
        
        while 1:
            # 有卡顿的现象，需要设置对列长度，缓冲
            # while sleep_time == 0 and read_status.value != 0 and video_out_queue.qsize() < 100:
            #     time.sleep(2)
            # sleep_time = 1
            if video_out_queue.qsize() > queue_maxsize:
                (status,(frame_count,data_show)) = video_out_queue.get()
                if show_model >= 1:
                    show_frame = data_show
                    if is_draw_box:
                        draw_bboxes_to_image(data_show['processed_image'], data_show['bbox'])
                        cv2.imshow(window_name, data_show['processed_image'])
                        if show_model == 2:
                            video_writer.write(data_show['processed_image'])
                    else:
                        if show_model == 1:
                            show_frame = draw_text_to_image(show_frame, CLASSES[status])
                            cv2.imshow(window_name, show_frame)
                        elif show_model == 2:
                            cv2.imshow(window_name, show_frame)
                            # show the highlight
                            if status != 0:
                                open_status = True
                                cv2.imshow(window_name + '_highlight' , show_frame)
                            else:
                                if open_status: 
                                    open_status = False
                                    cv2.destroyWindow(window_name + '_highlight')
                count += 1
                if (count-1)%time_interval == 0:
                    if start_time == 0:
                        start_time = time.time()
                    else:
                        end_time = time.time()
                        cost_time = end_time - start_time
                        start_time = end_time
                        print('%s is continue, the get frames nums is %s , avg per sencond get frames is %.2f'%(window_name ,count-1, (time_interval)/cost_time))
                if cv2.waitKey(20) & 0xFF == ord('q'):
                    break      
            else:
                if read_status.value == 0:
                    print('read status is end ')
                    break
        video_writer.release()
        print(window_name,'end, the total show frame is %s , avg frame interval is %.2f'%(count-1, (frame_count-1)/(count-2)))
        cv2.destroyAllWindows()

    # video_path_list = [r'.\video\jinboyang.mp4',r'.\video\yumi.mp4']
    # gpu_arrange_list = [(0,),(1,)]
    video_path_list = [r'rtsp://admin:p@ssw0rd@192.168.3.250/h264/ch33/main/av_stream',]
    gpu_arrange_list = [(0,1),]
    video_nums = len(video_path_list)

    # show model 0: no show  1: show 2channel  2: show highlight 4channel >>>>>for origin video
    # show model 0: no show  1: show  2: save >>>>>for processed video
    show_model = 1

    head_extract = VideoExtract( name= 'head'  ,gpu_arrange = (0,1) ,  video_source= r'rtsp://admin:p@ssw0rd@192.168.3.250/h264/ch33/main/av_stream' ,pb_path=PATH_TO_CKPT, OB_Model = OBModel )
    head_extract.verify_video_source()
    head_extract.load_detector_model()
    head_extract.start_process()
    image_show_thread = threading.Thread(target= origin_video_output, name = 'processed_video_head',args=(head_extract.processed_img_queue, 0, 'Real time detection processed video _ 0', show_model, True, head_extract.video_arrange_process_alive_status, head_extract.video_param_dict)) 
    image_show_thread.start()
    image_show_thread.join()
    head_extract.monitor_process()
    print('exit')

