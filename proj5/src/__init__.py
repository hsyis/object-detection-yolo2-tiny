import sys
import numpy as np
import cv2
import time
import argparse

import yolov2tiny

np_dtype = {
    'FP32': np.float32,
    'FP16': np.float16,
    'INT8': np.int8,
}


def resize_input(im):
    imsz = cv2.resize(im, (416, 416))
    imsz = imsz / 255.0
    imsz = imsz[:, :, ::-1]
    return np.asarray(imsz, dtype=np.float32)


def image_object_detection(in_image, out_image, precision):
    frame = cv2.imread(in_image)

    y2t = yolov2tiny.YOLO2_TINY(
        [1, 416, 416, 3], "./y2t_weights.onnx", precision)

    t_end2end = time.time()

    _frame = resize_input(frame)
    _frame = np.expand_dims(_frame, axis=0)

    t_inference = time.time()
    tout = y2t.inference(_frame)
    t_inference = time.time() - t_inference

    tout = np.squeeze(tout)

    assert(tout.dtype == np_dtype[precision])

    np.save(precision, tout)

    tout = tout.astype(np.float32)
    frame = yolov2tiny.postprocessing(
        tout, cv2.resize(frame, (416, 416), interpolation=cv2.INTER_CUBIC)
    )
    t_end2end = time.time() - t_end2end

    cv2.imwrite(out_image, frame)

    print("DNN inference elapsed time: %.3f" % t_inference)
    print("End-to-end elapsed time   : %.3f" % t_end2end)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("IN_IMAGE", help="path to the input jpg")
    parser.add_argument("OUT_IMAGE", help="path to the output jpg")
    parser.add_argument("PRECISION", choice=np_dtype.keys(),
                        help="Precision used for convolution")
    args = parser.parse_args()

    image_object_detection(args.IN_IMAGE, args.OUT_IMAGE, args.PRECISION)


if __name__ == "__main__":
    main()
