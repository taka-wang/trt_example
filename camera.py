import os
import ctypes
import time
import sys
import argparse

import cv2
import numpy as np
from PIL import Image
import tensorrt as trt

import utils.inference as inference_utils # TRT/TF inference wrappers
import utils.model as model_utils # UFF conversion
import utils.boxes as boxes_utils # Drawing bounding boxes
import utils.coco as coco_utils # COCO dataset descriptors
from utils.paths import PATHS # Path management

import pycuda.driver as cuda
import pycuda.autoinit

ctypes.CDLL("lib/libflattenconcat.so")
COCO_LABELS = coco_utils.COCO_CLASSES_LIST  # change this

# Confidence threshold for drawing bounding box
VISUALIZATION_THRESHOLD = 0.5

# Precision command line argument -> TRT Engine datatype
TRT_PRECISION_TO_DATATYPE = {
    16: trt.DataType.HALF,
    32: trt.DataType.FLOAT,
    8: trt.DataType.INT8
}

# Layout of TensorRT network output metadata
TRT_PREDICTION_LAYOUT = {
    "image_id": 0,
    "label": 1,
    "confidence": 2,
    "xmin": 3,
    "ymin": 4,
    "xmax": 5,
    "ymax": 6
}

def fetch_prediction_field(field_name, detection_out, pred_start_idx):
    """Fetches prediction field from prediction byte array.

    After TensorRT inference, prediction data is saved in
    byte array and returned by object detection network.
    This byte array contains several pieces of data about
    prediction - we call one such piece a prediction field.
    The prediction fields layout is described in TRT_PREDICTION_LAYOUT.

    This function, given prediction byte array returned by network,
    staring index of given prediction and field name of interest,
    returns prediction field data corresponding to given arguments.

    Args:
        field_name (str): field of interest, one of keys of TRT_PREDICTION_LAYOUT
        detection_out (array): object detection network output
        pred_start_idx (int): start index of prediction of interest in detection_out

    Returns:
        Prediction field corresponding to given data.
    """
    return detection_out[pred_start_idx + TRT_PREDICTION_LAYOUT[field_name]]

def analyze_prediction(detection_out, pred_start_idx, img_pil):
    image_id = int(fetch_prediction_field("image_id", detection_out, pred_start_idx))
    label = int(fetch_prediction_field("label", detection_out, pred_start_idx))
    confidence = fetch_prediction_field("confidence", detection_out, pred_start_idx)
    xmin = fetch_prediction_field("xmin", detection_out, pred_start_idx)
    ymin = fetch_prediction_field("ymin", detection_out, pred_start_idx)
    xmax = fetch_prediction_field("xmax", detection_out, pred_start_idx)
    ymax = fetch_prediction_field("ymax", detection_out, pred_start_idx)
    if confidence > VISUALIZATION_THRESHOLD:
        class_name = COCO_LABELS[label]
        confidence_percentage = "{0:.0%}".format(confidence)
        print("Detected {} with confidence {}".format(
            class_name, confidence_percentage))
        boxes_utils.draw_bounding_boxes_on_image(
            img_pil, np.array([[ymin, xmin, ymax, xmax]]),
            display_str_list=["{}: {}".format(
                class_name, confidence_percentage)],
            color=coco_utils.COCO_COLORS[label]
        )

def parse_commandline_arguments():
    """Parses command line arguments and adjusts internal data structures."""

    # Define script command line arguments
    parser = argparse.ArgumentParser(description='Run object detection inference on input image.')
    parser.add_argument('--input_img_path', metavar='INPUT_IMG_PATH',
        help='an image file to run inference on')
    parser.add_argument('-p', '--precision', type=int, choices=[32, 16, 8], default=32,
        help='desired TensorRT float precision to build an engine with')
    parser.add_argument('-b', '--max_batch_size', type=int, default=1,
        help='max TensorRT engine batch size')
    parser.add_argument('-d', '--calib_dataset', default='../VOCdevkit/VOC2007/JPEGImages',
        help='path to the calibration dataset')
    parser.add_argument('-c', '--camera', default=True,
        help='if True, will run webcam application')

    # Parse arguments passed
    args = parser.parse_args()

    # Fetch TensorRT engine path and datatype
    args.trt_engine_datatype = TRT_PRECISION_TO_DATATYPE[args.precision]

    return args

def main(uff_path, engine_path):

    # Parse command line arguments
    args = parse_commandline_arguments()

    # Set up all TensorRT data structures needed for inference
    trt_inference_wrapper = inference_utils.TRTInference(
        engine_path, uff_path,
        trt_engine_datatype=args.trt_engine_datatype,
        calib_dataset = args.calib_dataset,
        batch_size=args.max_batch_size)

    if args.camera == True:
        print('Running webcam:', args.camera)
        # Define the video stream
        cap = cv2.VideoCapture(0)  # Change only if you have more than one webcams
        cap.set(3, 1280)
        cap.set(4, 720)

        # Loop for running inference on frames from the webcam
        while True:
            # Read frame from camera (and expand its dimensions to fit)
            ret, image_np = cap.read()

            # Actually run inference
            detection_out, keep_count_out = trt_inference_wrapper.infer_webcam(image_np)

            # Overlay the bounding boxes on the image
            # let analyze_prediction() draw them based on model output
            img_pil = Image.fromarray(image_np)
            prediction_fields = len(TRT_PREDICTION_LAYOUT)
            for det in range(int(keep_count_out[0])):
                analyze_prediction(detection_out, det * prediction_fields, img_pil)
            final_img = np.asarray(img_pil)

            # Display output
            cv2.imshow('object detection', final_img)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

if __name__ == '__main__':
    main('tmp.uff', 'tmp.bin')
