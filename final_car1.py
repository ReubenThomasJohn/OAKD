from inspect import Attribute
from threading import Thread
import cv2, time, numpy as np
import time
import depthai as dai
import datetime
import os
from os import listdir
from os.path import isfile, join
import torch

class VideoStream(object):
    def __init__(self, SHAPE=600):
        self.shape = SHAPE
        self.filepath = os.getcwd() + '/output1'
        isExist = os.path.exists(self.filepath)
        if not isExist:
            os.makedirs(self.filepath)
        self.time_diff = 0
        self.start_time = datetime.datetime.now()
        self.counter = 0
        onlyfiles = [f for f in listdir(self.filepath) if isfile(join(self.filepath, f))]
        processed = []
        if len(onlyfiles) == 0:
            self.counter = 0
        else:
            for file in onlyfiles:
                splitfile = file.split('_')
                new = splitfile[2]
                new1 = new.split('.')
                processed.append(int(new1[0]))
        
            self.counter = sorted(processed)[-1] + 1
            print("Starting from file no: ", self.counter)
        self.lap = False
        self.result = cv2.VideoWriter('./output1/rgb_stream_'+str(self.counter)+'.avi', cv2.VideoWriter_fourcc(*'MJPG'), 60, (SHAPE, SHAPE)) # better to use os.join here
        self.prev_frame = 0
        self.next_frame = 0

        self.shape_NN = SHAPE
        self.filepathNN = os.getcwd() + '/NN_Output'
        isExistNN = os.path.exists(self.filepathNN)
        if not isExistNN:
            os.makedirs(self.filepathNN)
        self.counterNN = 0
        onlyfilesNN = [f for f in listdir(self.filepathNN) if isfile(join(self.filepathNN, f))]
        processed = []
        if len(onlyfilesNN) == 0:
            self.counterNN = 0
        else:
            for file in onlyfilesNN:
                splitfile = file.split('_')
                new = splitfile[2]
                new1 = new.split('.')
                processed.append(int(new1[0]))
        
            self.counterNN = sorted(processed)[-1] + 1
            print("Starting NN Output from file no: ", self.counterNN)

        self.resultNN = cv2.VideoWriter('./NN_Output/NN_Stream_'+str(self.counterNN)+'.avi', cv2.VideoWriter_fourcc(*'MJPG'), 20, (SHAPE, SHAPE)) # better to use os.join here
        # changed from 60 to 20.
        self.pipeline = dai.Pipeline()
        self.cam = self.pipeline.create(dai.node.ColorCamera)

        self.cam.setPreviewSize(SHAPE, SHAPE)
        self.cam.setBoardSocket(dai.CameraBoardSocket.RGB)
        self.cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        self.cam.setInterleaved(True)
        self.cam.setFps(30)
        self.Xout = self.pipeline.create(dai.node.XLinkOut)
        self.cam.preview.link(self.Xout.input)
        self.Xout.setStreamName("out")

        self.bestpt_model = torch.hub.load('ultralytics/yolov5', 'custom', path = 'best.pt') # Pothole
        
        # self.bestpt_model = torch.hub.load('ultralytics/yolov5', 'yolov5s') 
        time.sleep(15)

        self.thread1 = Thread(target=self.update, args=())
        self.thread1.daemon = True
        self.thread1.start()
        

        # self.thread2 = Thread(target=self.best_model, args = ())
        # self.thread2.daemon = True
        # self.thread2.start()

        self.thread3 = Thread(target=self.write_frame, args=())
        self.thread3.daemon = True
        self.thread3.start()

        
    def update(self):
        with dai.Device(self.pipeline) as device:
            self.Out = device.getOutputQueue(name="out", maxSize=4, blocking=False)
            print("Starting camera output...")
            # Read the next frame from the stream in a different thread
            while True:
                self.next_frame = time.time()      
                self.fps = 1/(self.next_frame - self.prev_frame)
                self.prev_frame = time.time()
                self.fps = str(int(self.fps))
                RGB = self.Out.get()
                self.RGB_Frame = RGB.getFrame()
                # time.sleep(.01)

    def best_model(self):
        results = self.bestpt_model(self.RGB_Frame)
        self.out = np.squeeze(results.render())
    
    def show_frame(self):
        # Display frames in main program
        copied_frame = np.copy(self.RGB_Frame)  
        cv2.putText(copied_frame, "fps: " + self.fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 1, cv2.LINE_AA)
        # mins, secs = divmod(600, 60)
        # timeformat = '{:02d}:{:02d}'.format(mins,secs)
        # cv2.putText(copied_frame, timeformat, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 1, cv2.LINE_AA)
        cv2.imshow('frame', copied_frame)
        cv2.imshow('NN', self.out)
        key = cv2.waitKey(1)
        if key == ord('q'):
            self.result.release()  # can remove this
            self.resultNN.release() # can remove this
            cv2.destroyAllWindows()
            exit(1)
    def write_frame(self):
        try:
            self.result.write(self.RGB_Frame)
            self.resultNN.write(self.out)
        except AttributeError:
            pass

    def break_video(self):        
        self.time_diff = datetime.datetime.now() - self.start_time
        if self.time_diff.total_seconds() > 300: # Change time here
            print('Creating file number {counter}'.format(counter=self.counter+1))
            self.lap = True
            self.counter += 1
            self.counterNN += 1
            self.start_time = datetime.datetime.now()
            self.result.release()
            self.resultNN.release()            
        else:
            pass

    def reset_video_write(self):
        if self.lap == True:
            self.result = cv2.VideoWriter('./output1/rgb_stream_'+str(self.counter)+'.avi', cv2.VideoWriter_fourcc(*'MJPG'), 60, (self.shape, self.shape))
            self.resultNN = cv2.VideoWriter('./NN_Output/NN_Stream_'+str(self.counter)+'.avi', cv2.VideoWriter_fourcc(*'MJPG'), 60, (self.shape, self.shape))
            self.lap = False
        else: 
            pass

if __name__ == '__main__':
    video_stream_widget = VideoStream()
    while True:
        try:
            video_stream_widget.best_model()
            video_stream_widget.show_frame()
            video_stream_widget.write_frame()
            video_stream_widget.break_video()
            video_stream_widget.reset_video_write()
        except AttributeError:
            pass
    