import pyrealsense2 as rs
import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
import sys


camera_Width = 640  # 1280 # 640
camera_Heigth = 480  # 960  # 480
centerZone = 100


# GridLine color green and thickness
lineColor = (0, 255, 0)
lineThickness = 1


# message color and thickness
colorBlue = (255, 0, 0)
colorGreen = (0, 255, 0)
colorRed = (0, 0, 255)
colorYellow = (40, 255, 255)
messageThickness = 2
messageThickness1 = 1

# Filters

decimate = rs.decimation_filter(magnitude=2)
spatial = rs.spatial_filter(
    smooth_alpha=0.5, smooth_delta=28, magnitude=2, hole_fill=1)
temporal = rs.temporal_filter(
    smooth_alpha=0.05, smooth_delta=28, persistence_control=2)
depth_to_disparity = rs.disparity_transform(True)
disparity_to_depth = rs.disparity_transform(False)


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-LimiteSeguridad', '--limiteseguridad', default='1.0', type=float)
    parser.add_argument(
        '-LimiteCentral', '--limitecentral', default='2.0', type=float)
    return parser


model = cv2.dnn.readNetFromTensorflow(
    "frozen_inference_graph_human_body.pb",
    "graph.pbtxt"
)


def getClassLabel(class_id, classes):

    for key, value in classes.items():

        if class_id == key:
            return value


COCO_labels = {0: 'background',
               1: '"Person"', 2: '', 3: '', 4: '',
               5: '', 6: '', 7: '', 8: '', 9: '',
               10: '', 11: '', 13: '', 14: '',
               15: '', 16: '', 17: '', 18: '', 19: '', 20: '', 21: '', 22: '',
               23: '', 24: '', 25: '', 27: '', 28: '',
               31: '', 32: '', 33: '', 34: '', 35: '',
               36: '', 37: '', 38: '', 39: '', 40: '',
               41: '', 42: '', 43: '', 44: '',
               46: '', 47: '', 48: '', 49: '', 50: '', 51: '', 52: '',
               53: '', 54: '', 55: '', 56: '', 57: '', 58: '', 59: '',
               60: '', 61: '', 62: '', 63: '', 64: '', 65: '',
               67: '', 70: '', 72: '', 73: '',
               74: '', 75: '', 76: '', 78: '', 79: '', 80: '', 81: '',
               82: '', 84: '', 85: '', 86: '', 87: '',
               88: '', 89: '', 90: ''}


def draw_predictions(pred_img, color_img, depth_image, colorized_depth, profile):
    for detection in pred_img[0, 0, :, :]:
        score = float(detection[2])
        #
        if score > 0.5:
            class_id = detection[1]
            class_label = getClassLabel(class_id, COCO_labels)
            left = detection[3] * color_img.shape[1]
            top = detection[4] * color_img.shape[0]
            right = detection[5] * color_img.shape[1]
            bottom = detection[6] * color_img.shape[0]
            if class_id == 1:
                cv2.rectangle(color_img, (int(left), int(top)),
                              (int(right), int(bottom)), (210, 230, 23), 2)
                cv2.rectangle(colorized_depth, (int(left), int(top)),
                              (int(right), int(bottom)), (210, 230, 23), 2)
            # if class_label = 'person'
                cv2.putText(color_img, class_label, (int(left), int(
                    bottom)-5), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)

            #

                depth = depth_image[int(left):int(right), int(
                    top):int(bottom)].astype(float)
                depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
                depth = depth * depth_scale
                dist, _, _, _ = cv2.mean(depth)
            # dist = round(dist, 1)
                if dist != 0.0:
                    cv2.putText(color_img, "dist: "+str(dist)+"m", (int(left),
                                                                    int(top)-5), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
                    cv2.putText(colorized_depth, "dist:"+str(dist)[:6]+"m", (int(right)//2, int(
                        bottom)//2), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
            # cv.putText(color_img,class_label,(int(left),int(bottom)+5,cv.FONT_HERSHEY_PLAIN,1,(255, 0, 255),1) "WARNING", (250, 50), 2,1, colorRed
                    #print("Detected a " + str(dist)[:5]+"meters away.")
                    if dist > 0.18 and dist <= 1:

                        cv2.putText(color_img, 'Caution_Person',
                                    (200, 400), 2, 1, colorYellow)
                        #cv2.rectangle(color_img, (int(left), int(top)), (int(right), int(bottom)), colorRed, 3)

            # elif dist > 0.8: cv.putText(color_img, 'PASS...', (50,50), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 3)


class KalmanFilter:

    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array(
        [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)

    def Estimate(self, coordX, coordY):
        ''' This function estimates the position of the object'''
        measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
        self.kf.correct(measured)
        predicted = self.kf.predict()
        return predicted


def init_intel_pipeline():
    pipeline = rs.pipeline()

    # configure stream, get the depth stream and colour stream
    # colour stream needed for background subtraction
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # start streaming
    profile = pipeline.start(config)

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)
    return align, pipeline, depth_scale, profile


# Dilation to remove noise and objects that are too far away that appears inconsequential
def morph_mask(fg_mask):

    # Create ellipse kernel for morph ops
    elipse_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))

    # Fill holes
    close = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, elipse_kernel)

    # Remove noise
    opening = cv2.morphologyEx(close, cv2.MORPH_OPEN, elipse_kernel)

    # Dilate to merge and increase detected in mask
    dilation = cv2.dilate(opening, elipse_kernel, iterations=2)

    return opening


def get_midpoint(x, y, w, h):
    mid_x = x + (w/2)
    mid_y = y + (h/2)
    return (mid_x, mid_y)


def detect(fg_mask, min_contour_width=10, min_contour_height=10):
    matches = []

    # finding contours
    contours, _ = cv2.findContours(
        fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    # contours = filter(lambda cont: cv2.contourArea(cont) > 10 , contours)

    # if len(contours) > 5:
    #   pass
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        contour_area = w * h
        if contour_area > 2000:
            continue

        valid_contours = (w >= min_contour_width) and (h >= min_contour_height)

        if not valid_contours:
            continue

        midpoint = get_midpoint(x, y, w, h)

        matches.append(((x, y, w, h), midpoint))
    return matches


def output(current_points, current_z, frame):
    cv2.putText(frame, current_points, (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 255, 255), 1)
    cv2.putText(frame, current_z, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 255, 255), 1)


def main():

    # Create bgs with 500 frames in cache and no shadow detection
    init_bgs = False
    frameNo = 0
    fg_mask = None
    kf = KalmanFilter()
    predictedXY = np.zeros((2, 1), np.float32)
    # get an image of the background
    mog2 = cv2.createBackgroundSubtractorMOG2(history=500, detectShadows=False)

    # open camera and start streaming
    align, pipeline, depth_scale, profile = init_intel_pipeline()

    xy, output_z = (" ",) * 2

    # check arguments
    parser = make_parser()
    args = parser.parse_args()

    try:
        while True:
            matches = []
            frames = pipeline.wait_for_frames()

            # Align the depth frame to color frame
            aligned_frames = align.process(frames)

            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame().as_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            #aligned_depth_frame = decimate.process(aligned_depth_frame )
            #aligned_depth_frame = depth_to_disparity.process(aligned_depth_frame)
            #aligned_depth_frame = spatial.process(aligned_depth_frame)
            #aligned_depth_frame = temporal.process(aligned_depth_frame)
            #aligned_depth_frame = disparity_to_depth.process(aligned_depth_frame)

            if not aligned_depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # feed the image to the neural network
            model.setInput(cv2.dnn.blobFromImage(
                color_image, size=(300, 300), swapRB=True, crop=False))

            # to identify the obstacle
            pred = model.forward()
            t, _ = model.getPerfProfile()
            # get colors to the depth image
            colorized_depth = cv2.applyColorMap(cv2.convertScaleAbs(
                depth_image, alpha=0.03), cv2.COLORMAP_JET)

            # Just draw boxes around the person
            draw_predictions(pred, color_image, depth_image,
                             colorized_depth, profile)

            color_copy = color_image.copy()
            # draw a line a the middle of the image, not sure why
            cv2.line(color_copy, (int(camera_Width/2), 0),
                     (int(camera_Width/2), camera_Heigth), lineColor, lineThickness)

            #
            if frameNo != 500:
                # Init BGS for the first 500 frames to have a clear template of the background
                mog2.apply(color_image, None, 0.001)
                frameNo = frameNo + 1
            else:
                # set a higher learning rate in deployment
                # removes background and updates model
                #fg_mask = mog2.apply(color_image, None, 0.005)
                fg_mask = mog2.apply(color_image, None, 0.005)
                # apply some morphological operations
                dilated_mask = morph_mask(fg_mask)
                #cv2.imshow("obstacles ",fg_mask)
                # detect obstacles
                matches = detect(dilated_mask)
                # for each obstacle get the mean
                for items in matches:
                    x, y, w, h = items[0]
                    mid_x, mid_y = items[1]
                    # xy = "X: %d, Y:%d" % (mid_x, mid_y)

                    # angle FOV for box

                    a = (42.6/320) * mid_x - 42.6
                    xy = "X: %d, Y: %d,alpha: %d" % (mid_x, mid_y, a)

                    depth_z = aligned_depth_frame.get_distance(
                        int(mid_x), int(mid_y))

                    if (x < int(camera_Width/2) - centerZone) and (depth_z <= args.limiteseguridad):
                        cv2.putText(color_copy, " L1-LEFT ",
                                    (5, 410), 1, 1, colorGreen)
                    elif (x > int(camera_Width/2) - centerZone) and (depth_z <= args.limiteseguridad):
                        cv2.putText(color_copy, " L1-RIGHT ",
                                    (450, 410), 1, 1, colorGreen)

                    elif (x > int(camera_Width/2) - centerZone) and (depth_z > args.limiteseguridad and depth_z < args.limitecentral):
                        cv2.putText(color_copy, "L2-RIGHT",
                                    (450, 255), 1, 1, colorGreen)
                    elif (x < int(camera_Width/2) - centerZone) and (depth_z > args.limiteseguridad and depth_z < args.limitecentral):
                        cv2.putText(color_copy, " L2-LEFT ",
                                    (5, 255), 1, 1, colorGreen)
                    elif (depth_z >= args.limitecentral):
                        cv2.putText(color_copy, "L3-FGround",
                                    (220, 100), 1, 1, colorGreen)

                    # depth_z = aligned_depth_frame.get_distance(
                        # int(mid_x), int(mid_y))

                    if depth_z != 0 and depth_z < 0.4:
                        output_z = "Depth_min: %f" % depth_z
                        cv2.putText(color_copy, "WARNING",
                                    (250, 50), 2, 1, colorRed)

                    future_x_y = kf.Estimate(mid_x, mid_y)

                    cv2.putText(color_copy, 'Obstacle detected',
                                (x+w+10, y+h), 0, 0.3, (0, 255, 0))
                    cv2.circle(color_copy, (int(mid_x), int(mid_y)),
                               3, (0, 255, 0), -2)
                    cv2.circle(
                        color_copy, (future_x_y[0], future_x_y[1]), 3, (0, 0, 255), -2)

                output(xy, output_z, color_copy)

            # Show everything
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(
                depth_image, alpha=0.03), cv2.COLORMAP_JET)
            cv2.imshow("Color ", color_copy)

            if fg_mask is not None:
                cv2.imshow("Depth", colorized_depth)
                continue

            else:
                print ("--Waiting Mask--")

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()

