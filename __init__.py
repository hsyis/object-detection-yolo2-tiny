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
    #raise NotImplementedError('open_video_with_opencv is not implemented yet')

    # Open an object of input video using cv2.VideoCapture.
    cap = cv2.VideoCapture(in_video_path)

    # Open an object of output video using cv2.VideoWriter.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    videoWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    videoHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    videoFPS = int(cap.get(cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter(out_video_path, fourcc, videoFPS, (videoWidth, videoHeight))

    # Return the video objects and anything you want for further process.
    return cap, out, videoWidth, videoHeight


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
    #raise NotImplementedError('video_object_detection is not implemented yet')

    # Open video using open_video_with_opencv.
    cap, out, video_width, video_height = open_video_with_opencv(in_video_path, out_video_path)

    # Check if video is opened. Otherwise, exit.
    if cap.isOpened() == False:
        exit()

    # Create an instance of the YOLO_V2_TINY class. Pass the dimension of
    # the input, a path to weight file, and which device you will use as arguments.
    input_dim = [1, yolov2tiny.in_height, yolov2tiny.in_width, 3]

    y2t = yolov2tiny.YOLO2_TINY(input_dim, onnx_path, proc)

    # Start the main loop. For each frame of the video, the loop must do the followings:
    # 1. Do the inference.
    # 2. Run postprocessing using the inference result, accumulate them through the video writer object.
    #    The coordinates from postprocessing are calculated according to resized input; you must adjust
    #    them to fit into the original video.
    # 3. Measure the end-to-end time and the time spent only for inferencing.
    # 4. Save the intermediate values for the first layer.
    # Note that your input must be adjusted to fit into the algorithm,
    # including resizing the frame and changing the dimension.
    is_first_frame = True

    elapse_end_2_end = 0.
    elapse_inference = 0.

    elapse_end_2_end_start = time.time()

    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            frame = resize_input(frame)
            expanded_frame = np.expand_dims(frame, 0)

            elapse_inference_start = time.time()

            nodes, out_tensors = y2t.inference(expanded_frame)

            elapse_inference += (time.time() - elapse_inference_start)

            frame = yolov2tiny.postprocessing(out_tensors[-1], frame)
            frame = np.uint8(frame * 255)
            frame = frame[:, :, ::-1]
            frame = cv2.resize(frame, (video_width, video_height))

            if is_first_frame:
                for i, out_tensor in enumerate(out_tensors):
                    np.save("intermediate/layer_" + str(i) + ".npy", out_tensor)
                is_first_frame = False

            out.write(frame)
        else:
             break

    elapse_end_2_end += (time.time() - elapse_end_2_end_start)

    # Check the inference peformance; end-to-end elapsed time and inferencing time.
    # Check how many frames are processed per second respectivly.
    print("end-to-end elpased time:  ", elapse_end_2_end)
    print("inferencing elapsed time: ", elapse_inference)
    print("how may FPS processed:    ", cap.get(cv2.CAP_PROP_FRAME_COUNT) / elapse_end_2_end)

    # Release the opened videos.
    cap.release()
    out.release()
    cv2.destroyAllWindows()


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
