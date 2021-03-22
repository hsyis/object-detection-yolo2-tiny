import sys
import numpy as np
import cv2 as cv2
import time

import yolov2tiny


def open_video_with_opencv(in_video_path, out_video_path):
    #
    # This function takes input and output video path and open them.
    #
    # Your code from here. You may clear the comments.
    #
    raise NotImplementedError('open_video_with_opencv is not implemented yet')

    # Open an object of input video using cv2.VideoCapture.

    # Open an object of output video using cv2.VideoWriter.

    # Return the video objects and anything you want for further process.


def resize_input(im):
    imsz = cv2.resize(im, (yolov2tiny.in_width, yolov2tiny.in_height))
    imsz = imsz / 255.
    imsz = imsz[:, :, ::-1]
    return np.asarray(imsz, dtype=np.float32)


def video_object_detection(in_video_path, out_video_path, proc="cpu", onnx_path="./y2t_weights.onnx"):
    #
    # This function runs the inference for each frame and creates the output video.
    #
    # Your code from here. You may clear the comments.
    #
    raise NotImplementedError('video_object_detection is not implemented yet')

    # Open video using open_video_with_opencv.
    _ = open_video_with_opencv(in_video_path, out_video_path)

    # Check if video is opened. Otherwise, exit.

    # Create an instance of the YOLO_V2_TINY class. Pass the dimension of
    # the input, a path to weight file, and which device you will use as arguments.

    # Start the main loop. For each frame of the video, the loop must do the followings:
    # 1. Do the inference.
    # 2. Run postprocessing using the inference result, accumulate them through the video writer object.
    #    The coordinates from postprocessing are calculated according to resized input; you must adjust
    #    them to fit into the original video.
    # 3. Measure the end-to-end time and the time spent only for inferencing.
    # 4. Save the intermediate values for the first layer.
    # Note that your input must be adjusted to fit into the algorithm,
    # including resizing the frame and changing the dimension.

    # Check the inference peformance; end-to-end elapsed time and inferencing time.
    # Check how many frames are processed per second respectivly.

    # Release the opened videos.


def main():
    if len(sys.argv) < 3:
        print(
            "Usage: python3 __init__.py [in_video.mp4] [out_video.mp4] ([cpu|gpu])")
        sys.exit()
    in_video_path = sys.argv[1]
    out_video_path = sys.argv[2]
    if len(sys.argv) == 4:
        proc = sys.argv[3]
    else:
        proc = "cpu"

    video_object_detection(in_video_path, out_video_path, proc)


if __name__ == "__main__":
    main()
