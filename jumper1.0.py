# pathing
import os
# model loading
import caffe
from caffe.proto import caffe_pb2
# image processing
import cv2
# help show framerate, log frame times
import time
from time import strftime, localtime
# c++ speed computation
import numpy as np
# for version detection
import imutils
# frame_buffer data structure
if imutils.is_cv2():
    from Queue import *
else:
    from queue import *
# simplifies writting command line interface
import argparse

user = 'HAAS'

if user == 'HAAS':
  caffe.set_mode_cpu()
else:
  caffe.set_mode_gpu()

# default database file
EXAMPLE_FILE = 'classification_data.csv'
# how many frames we ignore for a given set of frames with the same ID tag
IGNORE = 3
# used in FPS calculation
FRAMES = 5
# specify image width and height for transformations to network specs
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
# enum dictionaries for useful output
CLASS_TO_NUM = {'blouse' : 0, 'cloak' : 1, 'coat' : 2, 'jacket' : 3, 'long_dress' : 4, 'polo_shirt' : 5,
                'robe' : 6, 'shirt' : 7, 'short_dress' : 8, 'suit' : 9, 'sweater' : 10,
                't_shirt' : 11, 'undergarment' : 12, 'uniform' : 13, 'vest' : 14}
NUM_TO_CLASS = { value: key for key, value in CLASS_TO_NUM.items() }

# creates the ArgumentParser object
parser = argparse.ArgumentParser()

# adds an optional tags to the video capture
parser.add_argument("-v", "--video",
	help="path to the (optional) video file")

# adds an optional tag to turn on verbose debug stream
parser.add_argument("--debug", dest='DEBUG', action='store_true', default=False,
	help="Execute with a verbose debug stream to command line")

def ResetVariables():

    frame_buffer = Queue(maxsize=0)
    ID = -1
    fps = 0
    frame_count = 0
    on_screen = False

def WriteMessage(message, frame):

    location = (10, frame.shape[0]-10)
    color = (255, 255, 255)
    cv2.putText(frame,
    message,
    location,
    cv2.FONT_HERSHEY_SIMPLEX,
    0.35,
    color,
    1)




def TransformImage(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):

    #Histogram Equalization
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

    #Image Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)

    return img

def CropImage_10(img):
    CROP_SIZE = 227
    NUM_CROPS = 10
    tensor_img = np.ndarray((NUM_CROPS, 3, CROP_SIZE, CROP_SIZE))
    CENTER_MARGIN = (IMAGE_HEIGHT-CROP_SIZE)//2

    tensor_img[0, :, : ,:] = img[:, :CROP_SIZE, :CROP_SIZE].copy()
    tensor_img[1, :, : ,:] = img[:, IMAGE_HEIGHT-CROP_SIZE:, :CROP_SIZE].copy()
    tensor_img[2, :, : ,:] = img[:, :CROP_SIZE, IMAGE_WIDTH-CROP_SIZE:].copy()
    tensor_img[3, :, : ,:] = img[:, IMAGE_HEIGHT-CROP_SIZE:, IMAGE_WIDTH-CROP_SIZE:].copy()
    tensor_img[4, :, : ,:] = img[:, CENTER_MARGIN:IMAGE_HEIGHT-CENTER_MARGIN-1, CENTER_MARGIN:IMAGE_HEIGHT-CENTER_MARGIN-1].copy()
    img = cv2.flip(img, 1)
    tensor_img[5, :, : ,:] = img[:, :CROP_SIZE, :CROP_SIZE].copy()
    tensor_img[6, :, : ,:] = img[:, IMAGE_HEIGHT-CROP_SIZE:, :CROP_SIZE].copy()
    tensor_img[7, :, : ,:] = img[:, :CROP_SIZE, IMAGE_WIDTH-CROP_SIZE:].copy()
    tensor_img[8, :, : ,:] = img[:, IMAGE_HEIGHT-CROP_SIZE:, IMAGE_WIDTH-CROP_SIZE:].copy()
    tensor_img[9, :, : ,:] = img[:, CENTER_MARGIN:IMAGE_HEIGHT-CENTER_MARGIN-1, CENTER_MARGIN:IMAGE_HEIGHT-CENTER_MARGIN-1].copy()

    return tensor_img

def MakePrediction(image):

    net.blobs['data'].data[...] = image
    out = net.forward()
    return out['prob']

def ProcessQueue(queue):

    master = []
    if args.get("DEBUG"):
        print("Parsing queue of size: ", queue.qsize())
    for count, triple in enumerate(iter(queue.get, None)):

        ID = triple[0]
        time = triple[1]
        img = triple[2]
        img = TransformImage(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
        img = img.transpose((2,0,1)) - mean_array
        tensor_img = CropImage_10(img)
        probabilities = MakePrediction(tensor_img)
        if len(master) <= ID:
            master.insert(ID, [[time, np.average(probabilities, axis=0)]])
        else:
            master[ID].append([time, np.average(probabilities, axis=0)])
        if queue.empty():
            break

    for ID, prediction_list in enumerate(master):
        if len(prediction_list) < IGNORE:
            continue
        if args.get("DEBUG"):
            print("ID: ", ID)
            print("First seen at: ", prediction_list[0][0])
        sum_prob = np.zeros((1,15))
        max_prob = 0
        max_IDX = 0
        for pred in prediction_list[IGNORE:]:
            if (max(pred[1]) > max_prob) :
                max_prob = max(pred[1])
                max_IDX = np.argmax(pred[1])
            sum_prob += pred[1]
        sum_prob /= (len(prediction_list) - IGNORE)
        sum_prob = sum_prob.flatten()
        if args.get("DEBUG"):
            print("Max Probability: ", max_prob)
            print("Max acheived with: ", NUM_TO_CLASS[int(max_IDX)])
            for count, element in enumerate(prediction_list):
                print("Frame number ", count)
                if (count < IGNORE):
                    print("NOTE: Ignored in statistics")
                print(element[1])
            print("Averaged Probability over all frames: \n", sum_prob)
        top_five = sum_prob.argsort()[-5:][::-1] #np.argpartition(sum_prob, -5)[-5:]
        top_five_classes = []
        for num in top_five:
            top_five_classes.append(NUM_TO_CLASS[num])
        print("Top five classes (nums): ", top_five)
        print("Top five classes (words): ", top_five_classes)


    return

'''
Setting up the camera and network with supplied settings
'''

#Read mean image
mean_blob = caffe_pb2.BlobProto()

if user == 'HAAS':
    path = '/Users/JRod/Desktop/Jumper/alexnet_transfer/mean_256.binaryproto'
else:
    path = '/home/wyolland/Documents/cmpt-414-project/caffe/Jumper/fashion-data/input/mean_256.binaryproto'


with open(path, 'rb') as f:
    mean_blob.ParseFromString(f.read())
mean_array = np.asarray(mean_blob.data, dtype=np.float32).reshape((mean_blob.channels, mean_blob.height, mean_blob.width))

# parse_args() parses the arguments by inspecting the command-line, converting each object to the appropriate type and invoking the appropriate action. vars() returns the dictionary attribute for the objects
args = vars(parser.parse_args())

# if a video path was not supplied, grab the reference to the 0th webcam
if not args.get("video", False):
	camera = cv2.VideoCapture(0)
# otherwise, grab a reference to the video file
else:
	camera = cv2.VideoCapture(args["video"])

if imutils.is_cv2():
  ForegroundBackground = cv2.BackgroundSubtractorMOG2(50, 16, 0)

else:
  ForegroundBackground = cv2.createBackgroundSubtractorMOG2(history=50, detectShadows=False)

#Read model architecture and trained model's weights


net = caffe.Net('/Users/JRod/Desktop/Jumper/alexnet_transfer/deploy.prototxt',
                '/Users/JRod/Desktop/Jumper/alexnet_transfer/alexnet_transfer_iter_13000.caffemodel',
                caffe.TEST)

frame_buffer = Queue(maxsize=0)
ID = -1
fps = 0
frame_count = 0
on_screen = False
message_count = 0
message = ''

while True:

    if args.get("DEBUG"):
        if frame_count == 0:
            start = time.time()
        elif frame_count % FRAMES == 0:
            end = time.time()
            seconds = end - start
            fps = FRAMES / seconds
            start = end

    # get next webcam frame
    (grabbed, frame) = camera.read()

    if not grabbed:
        print ("fail to capture frame")
        break

    flipframe = cv2.flip(frame,1)
    orig = flipframe

    # generate mask frame
    # fgmask = ForegroundBackground.apply(background)
    fgmask = ForegroundBackground.apply(flipframe, learningRate=0.01)

    # erosians and dilations remove any small remaining imperfections in the mask
    fgmask = cv2.erode(fgmask, None, iterations=1)
    fgmask = cv2.dilate(fgmask, None, iterations=1)

    # calculate moment for centroid
    moments = cv2.moments(fgmask)
    m00 = moments['m00']
    centroid_x, centroid_y = None, None
    if m00 != 0:
        centroid_x = int(moments['m10']/m00)
        centroid_y = int(moments['m01']/m00)

    # assume no centroid
    ctr = (-1,-1)

    # use centroid if it exists
    if centroid_x != None and centroid_y != None:
        ctr = (centroid_x, centroid_y)
        #Put black circle in at centroid in image
        # cv2.circle(flipframe, ctr, 40, (0,0,255))

    rowIDX, colIDX = np.nonzero(fgmask)

    if (rowIDX.any() and colIDX.any()):
        if args.get("DEBUG"):
            cv2.rectangle(fgmask, (min(colIDX), min(rowIDX)), (max(colIDX), max(rowIDX)), (255,255,255))
        else:
            cv2.rectangle(flipframe, (min(colIDX), min(rowIDX)), (max(colIDX), max(rowIDX)), (255,255,255))
        if not on_screen:
            ID += 1
            on_screen = True
        if ((frame_count % 3) == 0):
            frame_buffer.put(
                ( ID,
                  strftime("%Y-%m-%d %H:%M:%S", localtime()),
                  orig[ min(rowIDX):max(rowIDX), min(colIDX):max(colIDX) ].copy() )
                  )
    else:
        on_screen = False


    if args.get("DEBUG"):
        # show the fps on the mask
        cv2.putText(fgmask, "fps: {}".format(fps), (fgmask.shape[1] - 80, fgmask.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        #write current message to screen
        if message_count > 0:
            WriteMessage(message, flipframe)

        cv2.imshow('mask',fgmask)
        cv2.imshow('frame',flipframe)

    else:
        cv2.imshow('frame', flipframe)

    key = cv2.waitKey(1) & 0xFF

    # keep track of whether message should be displayed
    if message_count > 0:
      message_count -= 1
    frame_count += 1

    if key == ord('p'):
        message = "Please wait while we process process the detected objects"
        WriteMessage(message, flipframe)
        cv2.imshow('frame',flipframe)
        
        ProcessQueue(frame_buffer)
        ResetVariables()

    if key == ord("c"):
        message = "Process queue reset"
        message_count = 30

        ResetVariables()


    if key == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()
