import os
import sys
import onnx
import pickle
import numpy as np
import cv2 as cv2

from activation import sigmoid, softmax

import tensorflow as tf

in_width = 416
in_height = 416


def preprocessing(input_img):
    return input_img


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
        self.g = tf.Graph()
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.compat.v1.Session(config=config, graph=self.g)
        self.proc = proc
        self.onnx_path = onnx_path
        self.inp, self.nodes = self.build_graph(in_shape)

    def get_y2t_w(self):
        '''Open ONNX file from onnx_path and return the parsed weights
        '''
        #
        # Your code from here. You may clear the comments.
        #
        #raise NotImplementedError('get_y2t_w is not implemented yet')
        model = onnx.load(self.onnx_path)

        y2t_w = {}

        for initializer in model.graph.initializer:
            weight = np.array(initializer.float_data, dtype=np.float32)
            y2t_w[initializer.name] = weight.reshape(initializer.dims)

        return y2t_w

    def build_graph(self, in_shape):
        '''Build a tensor graph to be used for future inference.
        '''
        #
        # Your code from here. You may clear the comments.
        #
        #raise NotImplementedError('build_graph is not implemented yet')

        # Load weight parameters from the ONNX file.
        y2t_w = self.get_y2t_w()

        # Create an empty list for tensors.
        nodes = []

        # Use self.g as a default graph. Refer to this API.
        # https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/Graph#as_default
        # Then you need to declare which device to use for tensor computation. The device info
        # is given from the command line argument and stored somewhere in this object.
        # In this project, you may choose CPU or GPU. Consider using the following API.
        # https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/Graph#device
        # Then you are ready to add tensors to the graph. According to the Yolo v2 tiny model,
        # build a graph and append the tensors to the returning list for computing intermediate
        # values. One tip is to start adding a placeholder tensor for the first tensor.
        # (Use 1e-5 for the epsilon value of batch normalization layers.)
        with self.g.as_default() as g:
            with g.device(self.proc):
                inp = tf.compat.v1.placeholder(tf.float32, shape=in_shape)
                #image2 = inp * y2t_w['scalerPreprocessor_scale'] + tf.transpose(y2t_w['scalerPreprocessor_bias'])

                convolution2d_1_output = tf.nn.conv2d(inp, tf.transpose(y2t_w['convolution_W'], perm=[2, 3, 1, 0]), strides=[1, 1], padding='SAME')

                batchnormalization_1_output = tf.nn.batch_normalization(convolution2d_1_output, y2t_w['BatchNormalization_mean'], y2t_w['BatchNormalization_variance'], y2t_w['BatchNormalization_B'], y2t_w['BatchNormalization_scale'], variance_epsilon=1e-5)
                leakyrelu_1_output = tf.nn.leaky_relu(batchnormalization_1_output, alpha=0.10000000149011612)
                maxpooling2d_1_output = tf.nn.max_pool2d(leakyrelu_1_output, [2, 2], [2, 2], 'SAME')
                convolution2d_2_output = tf.nn.conv2d(maxpooling2d_1_output, tf.transpose(y2t_w['convolution1_W'], perm=[2, 3, 1, 0]), strides=[1, 1], padding='SAME')

                batchnormalization_2_output = tf.nn.batch_normalization(convolution2d_2_output, y2t_w['BatchNormalization_mean1'], y2t_w['BatchNormalization_variance1'], y2t_w['BatchNormalization_B1'], y2t_w['BatchNormalization_scale1'], variance_epsilon=1e-5)
                leakyrelu_2_output = tf.nn.leaky_relu(batchnormalization_2_output, alpha=0.10000000149011612)
                maxpooling2d_2_output = tf.nn.max_pool2d(leakyrelu_2_output, [2, 2], [2, 2], 'SAME')
                convolution2d_3_output = tf.nn.conv2d(maxpooling2d_2_output, tf.transpose(y2t_w['convolution2_W'], perm=[2, 3, 1, 0]), strides=[1, 1], padding='SAME')

                batchnormalization_3_output = tf.nn.batch_normalization(convolution2d_3_output, y2t_w['BatchNormalization_mean2'], y2t_w['BatchNormalization_variance2'], y2t_w['BatchNormalization_B2'], y2t_w['BatchNormalization_scale2'], variance_epsilon=1e-5)
                leakyrelu_3_output = tf.nn.leaky_relu(batchnormalization_3_output, alpha=0.10000000149011612)
                maxpooling2d_3_output = tf.nn.max_pool2d(leakyrelu_3_output, [2, 2], [2, 2], 'SAME')
                convolution2d_4_output = tf.nn.conv2d(maxpooling2d_3_output, tf.transpose(y2t_w['convolution3_W'], perm=[2, 3, 1, 0]), strides=[1, 1], padding='SAME')

                batchnormalization_4_output = tf.nn.batch_normalization(convolution2d_4_output, y2t_w['BatchNormalization_mean3'], y2t_w['BatchNormalization_variance3'], y2t_w['BatchNormalization_B3'], y2t_w['BatchNormalization_scale3'], variance_epsilon=1e-5)
                leakyrelu_4_output = tf.nn.leaky_relu(batchnormalization_4_output, alpha=0.10000000149011612)
                maxpooling2d_4_output = tf.nn.max_pool2d(leakyrelu_4_output, [2, 2], [2, 2], 'SAME')
                convolution2d_5_output = tf.nn.conv2d(maxpooling2d_4_output, tf.transpose(y2t_w['convolution4_W'], perm=[2, 3, 1, 0]), strides=[1, 1], padding='SAME')

                batchnormalization_5_output = tf.nn.batch_normalization(convolution2d_5_output, y2t_w['BatchNormalization_mean4'], y2t_w['BatchNormalization_variance4'], y2t_w['BatchNormalization_B4'], y2t_w['BatchNormalization_scale4'], variance_epsilon=1e-5)
                leakyrelu_5_output = tf.nn.leaky_relu(batchnormalization_5_output, alpha=0.10000000149011612)
                maxpooling2d_5_output = tf.nn.max_pool2d(leakyrelu_5_output, [2, 2], [2, 2], 'SAME')
                convolution2d_6_output = tf.nn.conv2d(maxpooling2d_5_output, tf.transpose(y2t_w['convolution5_W'], perm=[2, 3, 1, 0]), strides=[1, 1], padding='SAME')

                batchnormalization_6_output = tf.nn.batch_normalization(convolution2d_6_output, y2t_w['BatchNormalization_mean5'], y2t_w['BatchNormalization_variance5'], y2t_w['BatchNormalization_B5'], y2t_w['BatchNormalization_scale5'], variance_epsilon=1e-5)
                leakyrelu_6_output = tf.nn.leaky_relu(batchnormalization_6_output, alpha=0.10000000149011612)
                maxpooling2d_6_output = tf.nn.max_pool2d(leakyrelu_6_output, [2, 2], [1, 1], 'SAME')
                convolution2d_7_output = tf.nn.conv2d(maxpooling2d_6_output, tf.transpose(y2t_w['convolution6_W'], perm=[2, 3, 1, 0]), strides=[1, 1], padding='SAME')

                batchnormalization_7_output = tf.nn.batch_normalization(convolution2d_7_output, y2t_w['BatchNormalization_mean6'], y2t_w['BatchNormalization_variance6'], y2t_w['BatchNormalization_B6'], y2t_w['BatchNormalization_scale6'], variance_epsilon=1e-5)
                leakyrelu_7_output = tf.nn.leaky_relu(batchnormalization_7_output, alpha=0.10000000149011612)
                convolution2d_8_output = tf.nn.conv2d(leakyrelu_7_output, tf.transpose(y2t_w['convolution7_W'], perm=[2, 3, 1, 0]), strides=[1, 1], padding='SAME')

                batchnormalization_8_output = tf.nn.batch_normalization(convolution2d_8_output, y2t_w['BatchNormalization_mean7'], y2t_w['BatchNormalization_variance7'], y2t_w['BatchNormalization_B7'], y2t_w['BatchNormalization_scale7'], variance_epsilon=1e-5)
                leakyrelu_8_output = tf.nn.leaky_relu(batchnormalization_8_output, alpha=0.10000000149011612)
                convolution2d_9_output = tf.nn.conv2d(leakyrelu_8_output, tf.transpose(y2t_w['convolution8_W'], perm=[2, 3, 1, 0]), strides=[1, 1], padding='SAME')
                grid = tf.nn.bias_add(convolution2d_9_output, y2t_w['convolution8_B'], data_format="NHWC")

                nodes.extend([#image2,
                              convolution2d_1_output,
                              batchnormalization_1_output,
                              leakyrelu_1_output,
                              maxpooling2d_1_output,
                              convolution2d_2_output,
                              batchnormalization_2_output,
                              leakyrelu_2_output,
                              maxpooling2d_2_output,
                              convolution2d_3_output,
                              batchnormalization_3_output,
                              leakyrelu_3_output,
                              maxpooling2d_3_output,
                              convolution2d_4_output,
                              batchnormalization_4_output,
                              leakyrelu_4_output,
                              maxpooling2d_4_output,
                              convolution2d_5_output,
                              batchnormalization_5_output,
                              leakyrelu_5_output,
                              maxpooling2d_5_output,
                              convolution2d_6_output,
                              batchnormalization_6_output,
                              leakyrelu_6_output,
                              maxpooling2d_6_output,
                              convolution2d_7_output,
                              batchnormalization_7_output,
                              leakyrelu_7_output,
                              convolution2d_8_output,
                              batchnormalization_8_output,
                              leakyrelu_8_output,
                              convolution2d_9_output,
                              grid])

        # Return the start tensor and the list of all tensors.
        return inp, nodes

    def inference(self, im):
        feed_dict = {self.inp: im}
        out_tensors = self.sess.run(self.nodes, feed_dict)
        return self.nodes, out_tensors
