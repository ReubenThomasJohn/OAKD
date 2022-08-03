from threading import Thread
import cv2, time, numpy as np
import time
import depthai as dai
import datetime
import os
from os import listdir
from os.path import isfile, join

from yoloV5_modified.YOLO_Fast() import post_process

class VideoStream(object):
    def __init__(self, SHAPE=640):

        self.blob_path = ''

        self.shape = SHAPE
        self.time_diff = 0
        self.start_time = datetime.datetime.now()
        self.shape_NN = SHAPE

        self.pipeline = dai.Pipeline()
        self.cam = self.pipeline.create(dai.node.ColorCamera)
        self.cam.setPreviewSize(SHAPE, SHAPE)
        self.cam.setBoardSocket(dai.CameraBoardSocket.RGB)
        self.cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_720_P)
        self.cam.setInterleaved(True)
        self.cam.setFps(30)
        self.Xout = self.pipeline.create(dai.node.XLinkOut)
        self.cam.preview.link(self.Xout.input)
        self.Xout.setStreamName("out")

        self.nn = self.pipeline.create(dai.node.NeuralNetwork)
        self.nn.setBlobPath(self.blob_path)
        self.cam.out.link(self.nn.input)

        # Send NN out to the host via XLink
        self.nnXout = self.pipeline.create(dai.node.XLinkOut)
        self.nnXout.setStreamName("nn")
        self.nn.out.link(self.nnXout.input)

        # self.bestpt_model = torch.hub.load('ultralytics/yolov5', 'custom', path = 'best.pt') # Pothole
        
        # self.bestpt_model = torch.hub.load('ultralytics/yolov5', 'yolov5s') 
        time.sleep(5)

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

    def run_model(self):
        qNn = self.device.getOutputQueue("nn")
        nnData = qNn.get() # Blocking
        # NN can output from multiple layers. Print all layer names:
        print(nnData.getAllLayerNames())

        # Get layer named "Layer1_FP16" as FP16
        self.layer1Data = nnData.getLayerFp16("Layer1_FP16")

    def post_processing(self):
        pass
    
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
            video_stream_widget.run_model()
            video_stream_widget.show_frame()
            video_stream_widget.write_frame()
            video_stream_widget.break_video()
            video_stream_widget.reset_video_write()
        except AttributeError:
            pass
    