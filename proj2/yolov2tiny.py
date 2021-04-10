import os
import sys
import pickle
import onnx
import numpy as np
import cv2 as cv2

from activation import sigmoid, softmax

#import tensorflow as tf
from dnn import DnnGraphBuilder, DnnInferenceEngine

in_width = 416
in_height = 416


def preprocessing(input_img):
    return input_img


"""
    https://github.com/simo23/tinyYOLOv2
"""


def iou(boxA, boxB):
    # boxA = boxB = [x1,y1,x2,y2]

    # Determine the coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection
    intersection_area = (xB - xA + 1) * (yB - yA + 1)

    # Compute the area of both rectangles
    boxA_area = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxB_area = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # Compute the IOU
    iou = intersection_area / float(boxA_area + boxB_area - intersection_area)

    return iou


"""
    https://github.com/simo23/tinyYOLOv2
"""


def non_maximal_suppression(thresholded_predictions, iou_threshold):

    nms_predictions = []

    # Add the best B-Box because it will never be deleted
    nms_predictions.append(thresholded_predictions[0])

    # For each B-Box (starting from the 2nd) check its iou with the higher score B-Boxes
    # thresholded_predictions[i][0] = [x1,y1,x2,y2]
    i = 1
    while i < len(thresholded_predictions):
        n_boxes_to_check = len(nms_predictions)
        #print('N boxes to check = {}'.format(n_boxes_to_check))
        to_delete = False

        j = 0
        while j < n_boxes_to_check:
            curr_iou = iou(
                thresholded_predictions[i][0], nms_predictions[j][0])
            if(curr_iou > iou_threshold):
                to_delete = True
            #print('Checking box {} vs {}: IOU = {} , To delete = {}'.format(thresholded_predictions[i][0],nms_predictions[j][0],curr_iou,to_delete))
            j = j+1

        if to_delete == False:
            nms_predictions.append(thresholded_predictions[i])
        i = i+1

    return nms_predictions


"""
    https://github.com/simo23/tinyYOLOv2
"""


def postprocessing(predictions, input_image):

    n_classes = 20
    n_grid_cells = 13
    n_b_boxes = 5
    n_b_box_coord = 4

    # Names and colors for each class
    classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
               "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    colors = [(254.0, 254.0, 254), (239.88888888888889, 211.66666666666669, 127),
              (225.77777777777777, 169.33333333333334,
               0), (211.66666666666669, 127.0, 254),
              (197.55555555555557, 84.66666666666667,
               127), (183.44444444444443, 42.33333333333332, 0),
              (169.33333333333334, 0.0,
               254), (155.22222222222223, -42.33333333333335, 127),
              (141.11111111111111, -84.66666666666664, 0), (127.0, 254.0, 254),
              (112.88888888888889, 211.66666666666669,
               127), (98.77777777777777, 169.33333333333334, 0),
              (84.66666666666667, 127.0,
               254), (70.55555555555556, 84.66666666666667, 127),
              (56.44444444444444, 42.33333333333332,
               0), (42.33333333333332, 0.0, 254),
              (28.222222222222236, -42.33333333333335,
               127), (14.111111111111118, -84.66666666666664, 0),
              (0.0, 254.0, 254), (-14.111111111111118, 211.66666666666669, 127)]

    # Pre-computed YOLOv2 shapes of the k=5 B-Boxes
    anchors = [1.08, 1.19,  3.42, 4.41,  6.63,
               11.38,  9.42, 5.11,  16.62, 10.52]

    thresholded_predictions = []

    # IMPORTANT: reshape to have shape = [ 13 x 13 x (5 B-Boxes) x (4 Coords + 1 Obj score + 20 Class scores ) ]
    predictions = np.reshape(predictions, (13, 13, 5, 25))

    # IMPORTANT: Compute the coordinates and score of the B-Boxes by considering the parametrization of YOLOv2
    for row in range(n_grid_cells):
        for col in range(n_grid_cells):
            for b in range(n_b_boxes):

                tx, ty, tw, th, tc = predictions[row, col, b, :5]

                # IMPORTANT: (416 img size) / (13 grid cells) = 32!
                # YOLOv2 predicts parametrized coordinates that must be converted to full size
                # final_coordinates = parametrized_coordinates * 32.0 ( You can see other EQUIVALENT ways to do this...)
                center_x = (float(col) + sigmoid(tx)) * 32.0
                center_y = (float(row) + sigmoid(ty)) * 32.0

                roi_w = np.exp(tw) * anchors[2*b + 0] * 32.0
                roi_h = np.exp(th) * anchors[2*b + 1] * 32.0

                final_confidence = sigmoid(tc)

                # Find best class
                class_predictions = predictions[row, col, b, 5:]
                class_predictions = softmax(class_predictions)

                class_predictions = tuple(class_predictions)
                best_class = class_predictions.index(max(class_predictions))
                best_class_score = class_predictions[best_class]

                # Flip the coordinates on both axes
                left = int(center_x - (roi_w/2.))
                right = int(center_x + (roi_w/2.))
                top = int(center_y - (roi_h/2.))
                bottom = int(center_y + (roi_h/2.))

                if((final_confidence * best_class_score) > 0.3):
                    thresholded_predictions.append(
                        [[left, top, right, bottom], final_confidence * best_class_score, classes[best_class]])

    # Sort the B-boxes by their final score
    thresholded_predictions.sort(key=lambda tup: tup[1], reverse=True)

    # Non maximal suppression
    if (len(thresholded_predictions) > 0):
        nms_predictions = non_maximal_suppression(thresholded_predictions, 0.3)
    else:
        nms_predictions = []

    # Draw final B-Boxes and label on input image
    for i in range(len(nms_predictions)):

        color = colors[classes.index(nms_predictions[i][2])]
        best_class_name = nms_predictions[i][2]

        # Put a class rectangle with B-Box coordinates and a class label on the image
        input_image = cv2.rectangle(input_image, (nms_predictions[i][0][0], nms_predictions[i][0][1]), (
            nms_predictions[i][0][2], nms_predictions[i][0][3]), color)
        cv2.putText(input_image, best_class_name, (int(min(nms_predictions[i][0][0], nms_predictions[i][0][2]) - 1), int(
            min(nms_predictions[i][0][1], nms_predictions[i][0][3])) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return input_image


class YOLO2_TINY(object):

    def __init__(self, in_shape, onnx_path, proc="cpu"):
        self.proc = proc
        self.onnx_path = onnx_path
        self.g = DnnGraphBuilder()
        self.build_graph(in_shape)
        self.sess = DnnInferenceEngine(self.g)

    def dictionarize_initializer(self, initializers):
        _initializers = {}
        for ini in initializers:
            _initializers[ini.name] = ini.float_data
        return _initializers

    def get_y2t_w(self):
        with open("../../y2t_weights.pickle", "rb") as h:
            pw = pickle.load(h, encoding='latin1')
        y2t_w = {}
        perm = [2, 3, 1, 0]
        om = onnx.load(self.onnx_path)
        initializers = self.dictionarize_initializer(om.graph.initializer)
        for i in range(0, 9):
            y2t_w[i] = {}
        y2t_w[0]["kernel"] = np.transpose(
            np.array(initializers["convolution_W"]).reshape(16, 3, 3, 3), perm)
        y2t_w[0]["biases"] = np.array(initializers["BatchNormalization_B"])
        y2t_w[0]["moving_mean"] = np.array(
            initializers["BatchNormalization_mean"])
        y2t_w[0]["moving_variance"] = np.array(
            initializers["BatchNormalization_variance"])
        y2t_w[0]["gamma"] = np.array(initializers["BatchNormalization_scale"])

        y2t_w[1]["kernel"] = np.transpose(
            np.array(initializers["convolution1_W"]).reshape(32, 16, 3, 3), perm)
        y2t_w[2]["kernel"] = np.transpose(
            np.array(initializers["convolution2_W"]).reshape(64, 32, 3, 3), perm)
        y2t_w[3]["kernel"] = np.transpose(
            np.array(initializers["convolution3_W"]).reshape(128, 64, 3, 3), perm)
        y2t_w[4]["kernel"] = np.transpose(
            np.array(initializers["convolution4_W"]).reshape(256, 128, 3, 3), perm)
        y2t_w[5]["kernel"] = np.transpose(
            np.array(initializers["convolution5_W"]).reshape(512, 256, 3, 3), perm)
        y2t_w[6]["kernel"] = np.transpose(
            np.array(initializers["convolution6_W"]).reshape(1024, 512, 3, 3), perm)
        y2t_w[7]["kernel"] = np.transpose(
            np.array(initializers["convolution7_W"]).reshape(1024, 1024, 3, 3), perm)

        for i in range(1, 8):
            y2t_w[i]["biases"] = np.array(
                initializers["BatchNormalization_B{}".format(i)])
            y2t_w[i]["moving_mean"] = np.array(
                initializers["BatchNormalization_mean{}".format(i)])
            y2t_w[i]["moving_variance"] = np.array(
                initializers["BatchNormalization_variance{}".format(i)])
            y2t_w[i]["gamma"] = np.array(
                initializers["BatchNormalization_scale{}".format(i)])

        y2t_w[8]["kernel"] = np.transpose(
            np.array(initializers["convolution8_W"]).reshape(125, 1024, 1, 1), perm)
        y2t_w[8]["biases"] = np.array(initializers["convolution8_B"])

        return y2t_w

    def build_graph(self, in_shape):
        y2t_w = self.get_y2t_w()

        inp = self.g.create_input(in_shape)

        conv0 = self.g.create_conv2d(inp, y2t_w[0]["kernel"], strides=[
                                     1, 1, 1, 1], padding='SAME')
        bn0 = self.g.create_batch_norm(
            conv0, y2t_w[0]["moving_mean"], y2t_w[0]["moving_variance"], y2t_w[0]["gamma"], 1e-5)
        bias0 = self.g.create_bias_add(bn0, y2t_w[0]["biases"])
        l1 = self.g.create_leaky_relu(bias0)
        p2 = self.g.create_max_pool2d(l1, ksize=[1, 2, 2, 1], strides=[
                                      1, 2, 2, 1], padding='SAME')

        conv3 = self.g.create_conv2d(p2, y2t_w[1]["kernel"], strides=[
                                     1, 1, 1, 1], padding='SAME')
        bn3 = self.g.create_batch_norm(
            conv3, y2t_w[1]["moving_mean"], y2t_w[1]["moving_variance"], y2t_w[1]["gamma"], 1e-5)
        bias3 = self.g.create_bias_add(bn3, y2t_w[1]["biases"])
        l4 = self.g.create_leaky_relu(bias3)
        p5 = self.g.create_max_pool2d(l4, ksize=[1, 2, 2, 1], strides=[
                                      1, 2, 2, 1], padding='SAME')

        conv6 = self.g.create_conv2d(p5, y2t_w[2]["kernel"], strides=[
                                     1, 1, 1, 1], padding='SAME')
        bn6 = self.g.create_batch_norm(
            conv6, y2t_w[2]["moving_mean"], y2t_w[2]["moving_variance"], y2t_w[2]["gamma"], 1e-5)
        bias6 = self.g.create_bias_add(bn6, y2t_w[2]["biases"])
        l7 = self.g.create_leaky_relu(bias6)
        p8 = self.g.create_max_pool2d(l7, ksize=[1, 2, 2, 1], strides=[
                                      1, 2, 2, 1], padding='SAME')

        conv9 = self.g.create_conv2d(p8, y2t_w[3]["kernel"], strides=[
                                     1, 1, 1, 1], padding='SAME')
        bn9 = self.g.create_batch_norm(
            conv9, y2t_w[3]["moving_mean"], y2t_w[3]["moving_variance"], y2t_w[3]["gamma"], 1e-5)
        bias9 = self.g.create_bias_add(bn9, y2t_w[3]["biases"])
        l10 = self.g.create_leaky_relu(bias9)
        p11 = self.g.create_max_pool2d(l10, ksize=[1, 2, 2, 1], strides=[
                                       1, 2, 2, 1], padding='SAME')

        conv12 = self.g.create_conv2d(p11, y2t_w[4]["kernel"], strides=[
                                      1, 1, 1, 1], padding='SAME')
        bn12 = self.g.create_batch_norm(
            conv12, y2t_w[4]["moving_mean"], y2t_w[4]["moving_variance"], y2t_w[4]["gamma"], 1e-5)
        bias12 = self.g.create_bias_add(bn12, y2t_w[4]["biases"])
        l13 = self.g.create_leaky_relu(bias12)
        p14 = self.g.create_max_pool2d(l13, ksize=[1, 2, 2, 1], strides=[
                                       1, 2, 2, 1], padding='SAME')

        conv15 = self.g.create_conv2d(p14, y2t_w[5]["kernel"], strides=[
                                      1, 1, 1, 1], padding='SAME')
        bn15 = self.g.create_batch_norm(
            conv15, y2t_w[5]["moving_mean"], y2t_w[5]["moving_variance"], y2t_w[5]["gamma"], 1e-5)
        bias15 = self.g.create_bias_add(bn15, y2t_w[5]["biases"])
        l16 = self.g.create_leaky_relu(bias15)
        p17 = self.g.create_max_pool2d(l16, ksize=[1, 2, 2, 1], strides=[
                                       1, 1, 1, 1], padding='SAME')

        conv18 = self.g.create_conv2d(p17, y2t_w[6]["kernel"], strides=[
                                      1, 1, 1, 1], padding='SAME')
        bn18 = self.g.create_batch_norm(
            conv18, y2t_w[6]["moving_mean"], y2t_w[6]["moving_variance"], y2t_w[6]["gamma"], 1e-5)
        bias18 = self.g.create_bias_add(bn18, y2t_w[6]["biases"])
        l19 = self.g.create_leaky_relu(bias18)

        conv20 = self.g.create_conv2d(l19, y2t_w[7]["kernel"], strides=[
                                      1, 1, 1, 1], padding='SAME')
        bn20 = self.g.create_batch_norm(
            conv20, y2t_w[7]["moving_mean"], y2t_w[7]["moving_variance"], y2t_w[7]["gamma"], 1e-5)
        bias20 = self.g.create_bias_add(bn20, y2t_w[7]["biases"])
        l21 = self.g.create_leaky_relu(bias20)

        conv22 = self.g.create_conv2d(l21, y2t_w[8]["kernel"], strides=[
                                      1, 1, 1, 1], padding='SAME')
        out = self.g.create_bias_add(conv22, y2t_w[8]["biases"])

        self.g.set_out_node(out)

    def inference(self, im):
        out = self.sess.run(im)
        return out
