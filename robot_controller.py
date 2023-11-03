import cv2
import time
import numpy as np
import onnxruntime 
import spidev as SPI
import xgoscreen.LCD_2inch as LCD_2inch
from PIL import Image, ImageDraw, ImageFont
from xgolib import XGO
import threading
import random
import atexit

# Constants defining camera resolution and known parameters for distance calculation
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240
KNOWN_DISTANCE = 76.2  # Example known distance from the camera to a face
KNOWN_WIDTH = 14.3     # Example known width of the face
FOCAL_LENGTH = 500     # Preset focal length of the camera

class DogRobotController:
    def __init__(self):
        self.dog = XGO('COM4', 115200, 'xgolite')
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, CAMERA_WIDTH)
        self.cap.set(4, CAMERA_HEIGHT)
        self.session = onnxruntime.InferenceSession('/home/pi/model/Model.onnx')
        self.stop_thread = False
        self.t = None
    
    def sigmoid(self, x):
        return 1. / (1 + np.exp(-x))

    def tanh(self, x):
        return 2. / (1 + np.exp(-2 * x)) - 1

    def preprocess(self, src_img, size):
        output = cv2.resize(src_img, (size[0], size[1]), interpolation=cv2.INTER_AREA)
        output = output.transpose(2, 0, 1)
        output = output.reshape((1, 3, size[1], size[0])) / 255
        return output.astype('float32')

    def nms(self, dets, thresh=0.45):
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]
        output = []
        for i in keep:
            output.append(dets[i].tolist())
        return output

    def detection(self, session, img, input_width, input_height, thresh):
        try:
            pred = []
            H, W, _ = img.shape
            data = self.preprocess(img, [input_width, input_height])
            input_name = session.get_inputs()[0].name
            feature_map = session.run([], {input_name: data})[0][0]
            feature_map = feature_map.transpose(1, 2, 0)
            feature_map_height = feature_map.shape[0]
            feature_map_width = feature_map.shape[1]
            for h in range(feature_map_height):
                for w in range(feature_map_width):
                    data = feature_map[h][w]
                    obj_score, cls_score = data[0], data[5:].max()
                    score = (obj_score ** 0.6) * (cls_score ** 0.4)
                    if score > thresh:
                        cls_index = np.argmax(data[5:])
                        x_offset, y_offset = self.tanh(data[1]), self.tanh(data[2])
                        box_width, box_height = self.sigmoid(data[3]), self.sigmoid(data[4])
                        box_cx = (w + x_offset) / feature_map_width
                        box_cy = (h + y_offset) / feature_map_height
                        x1, y1 = box_cx - 0.5 * box_width, box_cy - 0.5 * box_height
                        x2, y2 = box_cx + 0.5 * box_width, box_cy + 0.5 * box_height
                        x1, y1, x2, y2 = int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H)
                        pred.append([x1, y1, x2, y2, score, cls_index])
            return self.nms(np.array(pred))
        except:
            return None
    
    def object_data(self, image):
        input_width, input_height = 352, 352
        bboxes = self.detection(self.session, image, input_width, input_height, 0.65)
        return bboxes

    def distance_finder(self, focal_length, real_width, width_in_rf_image):
        distance = (real_width * focal_length) / width_in_rf_image
        return distance

    def move_robot_based_on_distance(self, distance, target_distance=50, too_close_distance=10):
        """
        Moves the robot forward if the object is centered and the distance is more than the target distance.
        If the object is too close, the robot stops and turns around.

        :param distance: The current distance from the target (in cm).
        :param target_distance: The desired distance to maintain from the target (in cm).
        :param too_close_distance: The distance at which the robot considers it is too close to the object (in cm).
        """
        if distance > target_distance:
            # If the robot is further than the target distance, move forward
            self.dog.move('x', 10)  # Move forward at a speed of 10 (arbitrary units)
        elif distance < too_close_distance:
            # If the robot is closer than the too close distance, stop moving and turn around
            self.dog.stop()  # Stop the robot's movement
            time.sleep(1)  # Wait for a moment before turning
            self.dog.move('y', -5 if random.choice([True, False]) else 5)  # Turn left or right randomly
            time.sleep(1)  # Turn for a bit
            self.dog.stop()  # Stop the turning movement
        else:
            # If the robot is within the target distance and not too close, do not move
            self.dog.stop()

    def move_robot_based_on_position(self, object_center_x, frame_width):
        """
        Adjusts the robot's movement based on the object's position in the frame.
        The object should be positioned in a central area of the frame.
        """
        center_tolerance = frame_width * 0.2  # Increased tolerance range for centering
        center_range = (frame_width / 2 - center_tolerance, frame_width / 2 + center_tolerance)

        if object_center_x < center_range[0]:  # Object is left of center
            self.dog.move('y', -5)  # Move left
        elif object_center_x > center_range[1]:  # Object is right of center
            self.dog.move('y', 5)  # Move right
        else:  # Object is within central range
            # The object is centered; no lateral movement is needed
            self.dog.move('x', 10)  # Move forward
            pass  # You could potentially add a command to move forward here if necessary

        
    def move_robot_towards_person(self, object_center_x, frame_width, distance, target_distance):
        """
        Moves the robot towards the person if they are detected.
        The robot will maintain a specific distance to the person.
        """
        center_tolerance = frame_width * 0.1
        center_range = (frame_width / 2 - center_tolerance, frame_width / 2 + center_tolerance)

        # Centering logic
        if object_center_x < center_range[0]:
            self.dog.move('y', -5)  # Move left
        elif object_center_x > center_range[1]:
            self.dog.move('y', 5)  # Move right

        # Distance maintaining logic
        if distance > target_distance + 10:  # Add a margin to prevent too much forward-backward movement
            self.dog.move('x', 10)  # Move forward
        elif distance < target_distance - 10:  # Subtract a margin to prevent too much forward-backward movement
            self.dog.move('x', -10)  # Move backward
            
    def run(self):
        KNOWN_WIDTH = 14.3  # Width of the object (person's face in this case)
        focal_length_found = 500
        frame_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        target_distance = 50  # The desired distance from the object (in cm)

        while not self.stop_thread:
            _, frame = self.cap.read()

            # Use object detection instead of face recognition
            bboxes = self.object_data(frame)
            if bboxes:
                for bbox in bboxes:
                    # Assuming the first class (index 0) is 'person'
                    if int(bbox[5]) == 0:  # Check if the detected class is a person
                        x1, y1, x2, y2 = bbox[:4]
                        object_width_in_frame = x2 - x1
                        object_center_x = (x1 + x2) / 2
                        distance = self.distance_finder(focal_length_found, KNOWN_WIDTH, object_width_in_frame)
                        print(f"Distance to object: {distance} cm")

                        # Move robot based on position and distance
                        self.move_robot_based_on_position(object_center_x, frame_width)
                        self.move_robot_based_on_distance(distance, target_distance)
                        break  # Stop after processing the first person
            else:
                # If no person is detected, turn in place to search
                print("No person detected. Turning to search...")
                self.dog.move('y', 5)  # Turn right or left
                time.sleep(1)  # Turn for a bit
                self.dog.stop()


    def start(self):
        self.t = threading.Thread(target=self.run)
        self.t.start()

    def stop(self):
        self.stop_thread = True
        if self.t:
            self.t.join()
        cv2.destroyAllWindows()

controller = DogRobotController()
controller.start()
time.sleep(40)  # Let it run for e.g. 40 seconds
controller.stop()
