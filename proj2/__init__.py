import sys
import numpy as np
import cv2
import time

import yolov2tiny


def resize_input(im):
    imsz = cv2.resize(im, (416, 416))
    imsz = imsz / 255.
    imsz = imsz[:, :, ::-1]
    return np.asarray(imsz, dtype=np.float32)


def image_object_detection(in_image, out_image):
    frame = cv2.imread(in_image)

    y2t = yolov2tiny.YOLO2_TINY([1, 416, 416, 3], "./y2t_weights.onnx")

    t_end2end = time.time()

    _frame = resize_input(frame)
    _frame = np.expand_dims(_frame, axis=0)

    t_inference = time.time()
    tout = y2t.inference(_frame)
    t_inference = time.time() - t_inference

    tout = np.squeeze(tout)
    frame = yolov2tiny.postprocessing(tout, cv2.resize(
        frame, (416, 416), interpolation=cv2.INTER_CUBIC))
    t_end2end = time.time() - t_end2end

    cv2.imwrite(out_image, frame)

    print('DNN inference elapsed time: %.3f' % t_inference)
    print('End-to-end elapsed time   : %.3f' % t_end2end)


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 __init__.py [in_image] [out_image]")
        sys.exit()
    image_in = sys.argv[1]
    image_out = sys.argv[2]

    image_object_detection(image_in, image_out)


if __name__ == "__main__":
    main()
