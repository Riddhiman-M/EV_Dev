
import cv2
import csv
import numpy as np
from tracker import *

tracker = EuclideanDistTracker()

path = "C:/Users/Riddhiman Moulick/IIT Kharagpur/pythonProject/Electric_Vehicle/Resource/Traffic - 27260.mp4"
cap = cv2.VideoCapture(path)
input_size = 320

# Detection confidence threshold
confThreshold = 0.2
nmsThreshold = 0.2

font_color = (0, 0, 255)
font_size = 0.5
font_thickness = 2

# Middle cross line position (Detection Lines)
middle_line_position = 125
up_line_position = middle_line_position - 15
down_line_position = middle_line_position + 15

# Coco Dataset to detect vehicle classes
classesFile = "coco.names"
classNames = open(classesFile).read().strip().split('\n')

# class index for our required detection classes
# indices have been found by checking the names of vehicle classes in Coco
required_class_index = [2, 3, 5, 7]

detected_classNames = []

# Model Files
modelConfiguration = 'yolov3-320.cfg'
modelWeights = 'yolov3-320.weights'

# Configuring the network model
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)

# Defining random colour (of bounding boxes) for each class of vehicle
np.random.seed(38)
colors = np.random.randint(0, 255, size=(len(classNames), 3), dtype='uint8')

# Function for finding the center of a rectangle
def find_center(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy


# List for store vehicle count information
temp_up_list = []
temp_down_list = []
up_list = [0, 0, 0, 0]
down_list = [0, 0, 0, 0]
total_list = [0, 0, 0, 0]

# Function for count vehicle
def count_vehicle(box_id, img):
    x, y, w, h, id, index = box_id  # Here index denotes type (class) of vehicle

    # Find the center of the rectangle for detection
    center = find_center(x, y, w, h)
    ix, iy = center

    # Find the current position of the vehicle
    if (iy > up_line_position) and (iy < middle_line_position):

        if id not in temp_up_list:
            temp_up_list.append(id)

    elif (iy < down_line_position) and (iy > middle_line_position):

        if id not in temp_down_list:
            temp_down_list.append(id)

    elif iy < up_line_position:
        if id in temp_down_list:
            temp_down_list.remove(id)
            up_list[index] = up_list[index] + 1

    elif iy > down_line_position:
        if id in temp_up_list:
            temp_up_list.remove(id)
            down_list[index] = down_list[index] + 1


# Function for finding the detected objects from the network output
def postProcess(outputs, img):
    global detected_classNames
    height, width = img.shape[0], img.shape[1]
    boxes = []
    classIds = []
    confidence_scores = []
    detection = []
    for output in outputs:
        for det in output:
            scores = det[5:]  # 80 total classes in COCO Dataset hence 80 output values in score
            classId = np.argmax(scores)  # stores index of class where score is maximum
            confidence = scores[classId]
            if classId in required_class_index:
                if confidence > confThreshold:
                    w, h = int(det[2] * width), int(det[3] * height)
                    x, y = int((det[0] * width) - w / 2), int((det[1] * height) - h / 2)
                    boxes.append([x, y, w, h])
                    classIds.append(classId)
                    confidence_scores.append(float(confidence))

    # Non-Max Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidence_scores, confThreshold, nmsThreshold)

    for i in indices.flatten():  # Converting matrix indices to 1-D array
        x, y, w, h = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]

        color = [int(c) for c in colors[classIds[i]]]
        name = classNames[classIds[i]]
        detected_classNames.append(name)
        
        # Draw classname and confidence score
        cv2.putText(img, f'{name.upper()} {int(confidence_scores[i] * 100)}%',
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Draw bounding rectangle
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
        detection.append([x, y, w, h, required_class_index.index(classIds[i])])

    # Update the tracker for each object
    boxes_ids = tracker.update(detection)
    for box_id in boxes_ids:
        count_vehicle(box_id, img)


def realTime():
    while cap.isOpened():

        success, img = cap.read()

        if not success:
            break

        img = cv2.resize(img, (0, 0), None, 0.5, 0.5)
        ih, iw, channels = img.shape
        blob = cv2.dnn.blobFromImage(img, 1 / 255, (input_size, input_size), [0, 0, 0], 1, crop=False)

        # Setting the input of the network
        net.setInput(blob)
        layersNames = net.getLayerNames()
        outputNames = [(layersNames[i[0] - 1]) for i in net.getUnconnectedOutLayers()]
        # Feeding data to the network
        outputs = net.forward(outputNames)

        # Detecting vehicles and updating up_list and down_list
        postProcess(outputs, img)

        # Crossing lines
        cv2.line(img, (0, middle_line_position), (iw, middle_line_position), (255, 255, 255), 2)
        cv2.line(img, (0, up_line_position), (iw, up_line_position), (0, 0, 255), 2)
        cv2.line(img, (0, down_line_position), (iw, down_line_position), (0, 0, 255), 2)

        # Vehicle Count texts in the frame
        cv2.putText(img, "Car:        " + str(up_list[0]+down_list[0]), (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, "Motorbike:  " + str(up_list[1]+down_list[1]), (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, "Bus:        " + str(up_list[2]+down_list[2]), (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, "Truck:      " + str(up_list[3]+down_list[3]), (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)

        cv2.imshow('Output', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Storing and saving the vehicle count information in a file

    for i in range(4):
        total_list[i] = up_list[i] + down_list[i]

    with open("data.csv", 'w') as f1:
        cwriter = csv.writer(f1)
        cwriter.writerow(['Cars', 'Motorbikes', 'Buses', 'Trucks'])
        cwriter.writerow(total_list)
    f1.close()
    print("Data saved at 'data.csv'")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    realTime()