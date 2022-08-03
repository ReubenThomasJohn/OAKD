# Before running the script - make sure python 3 is installed, 3.10.1 preferrably
# run the command git clone https://github.com/luxonis/depthai
# Next, cd into depthai
# run python install_requirements.py
# Ensure the best.pt file is in the right path. The best.pt file is called in line

# With this, this particular script will run without any issue

import numpy as np
import depthai as dai
import torch
import cv2
import matplotlib.pyplot as plt

def compute_disparity(image, img_pair, num_disparities=6*16, block_size=11, window_size=6, matcher="stereo_sgbm", show_disparity=True):
    """
    Create a Stereo BM or Stereo SGBM Matcher
    Compute the Matching
    Display the disparity image
    Return it 
    """
    if matcher == "stereo_bm":
        new_image = cv2.StereoBM_create(numDisparities=num_disparities,blockSize=block_size)
    elif matcher == "stereo_sgbm":
        '''
        Understand parameters: https://docs.opencv.org/3.4/d2/d85/classcv_1_1StereoSGBM.html
        '''
        new_image = cv2.StereoSGBM_create(minDisparity=0, numDisparities=num_disparities, blockSize=block_size, P1=8 * 3 * window_size ** 2,
            P2=32 * 3 * window_size ** 2, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

    new_image = new_image.compute(image, img_pair).astype(np.float32)/16
    if (show_disparity==True):
        plt.figure(figsize = (40,20))
        plt.imshow(new_image, cmap="cividis")
        plt.show()
    return new_image

def calc_depth_map(disp_left):
    # Get the focal length from the K matrix
    f =882.5 # k_left[0]
    # Get the distance between the cameras from the t matrices (baseline)
    b =0.075 #abs(t_left[0] - t_right[0]) #On the setup page, you can see 0.54 as the distance between the two color cameras (http://www.cvlibs.net/datasets/kitti/setup.php)
    # Replace all instances of 0 and -1 disparity with a small minimum value (to avoid div by 0 or negatives)
    disp_left[disp_left == 0] = 0.1
    disp_left[disp_left == -1] = 0.1
    # Initialize the depth map to match the size of the disparity map
    depth_map = np.ones(disp_left.shape, np.single)
    # Calculate the depths 
    depth_map[:] = f * b / disp_left[:]
    return depth_map

def run_obstacle_detection(img, model):

  #Convert to BGR
    dets = []
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized_image = cv2.resize(img, (460, 460))
    results = model(resized_image)
    locs = (results.pandas().xywhn)[0]
    # print(locs)
    for lab, row in locs.iterrows():
            x = row['xcenter']
            y = row['ycenter']
            w = row['width']
            h = row['height']
            dets.append([x, y, w, h])
            
    return np.squeeze(results.render()), dets

def find_distances(depth_map, pred_bboxes, img, method="center"):
    """
    Go through each bounding box and take a point in the corresponding depth map. 
    It can be:
    * The Center of the box
    * The average value
    * The minimum value (closest point)
    * The median of the values
    """
    depth_list = []
    h, w = img.shape
    for box in pred_bboxes:
        # x1 = int(box[0]*w - box[2]*w*0.5) # center_x - width /2
        # y1 = int(box[1]*h-box[3]*h*0.5) # center_y - height /2
        # x2 = int(box[0]*w + box[2]*w*0.5) # center_x + width/2
        # y2 = int(box[1]*h+box[3]*h*0.5) # center_y + height/2

        x1 = int(box[0]*w - box[2]*w*0.5) # center_x - width /2
        y1 = int(box[1]*h-box[3]*h*0.5) # center_y - height /2
        x2 = int(box[0]*w + box[2]*w*0.5) # center_x + width/2
        y2 = int(box[1]*h+box[3]*h*0.5) # center_y + height/2
        #print(np.array([x1, y1, x2, y2]))
        obstacle_depth = depth_map[y1:y2, x1:x2]
        if method=="closest":
            depth_list.append(obstacle_depth.min()) # take the closest point in the box
        elif method=="average":
            depth_list.append(np.mean(obstacle_depth)) # take the average
        elif method=="median":
            depth_list.append(np.median(obstacle_depth)) # take the median
        else:
            depth_list.append(depth_map[int(box[1]*h)][int(box[0]*w)]) # take the center
    return depth_list

def add_depth(depth_list, result, pred_bboxes):
    h, w, _ = result.shape
    res = result.copy()
    for i, distance in enumerate(depth_list):
        cv2.line(res,(int(pred_bboxes[i][0]*w - pred_bboxes[i][2]*w/2),int(pred_bboxes[i][1]*h - pred_bboxes[i][3]*h*0.5)),(int(pred_bboxes[i][0]*w + pred_bboxes[i][2]*w*0.5),int(pred_bboxes[i][1]*h + pred_bboxes[i][3]*h*0.5)),(255,255,255),2)
        cv2.line(res,(int(pred_bboxes[i][0]*w + pred_bboxes[i][2]*w/2),int(pred_bboxes[i][1]*h - pred_bboxes[i][3]*h*0.5)),(int(pred_bboxes[i][0]*w - pred_bboxes[i][2]*w*0.5),int(pred_bboxes[i][1]*h + pred_bboxes[i][3]*h*0.5)),(255,255,255),2)
        cv2.putText(res, '{0:.2f} m'.format(distance), (int(pred_bboxes[i][0]*w - pred_bboxes[i][2]*w*0.5),int(pred_bboxes[i][1]*h)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 100), 2, cv2.LINE_AA)    
    return res

def perception_pipeline(img_left, img_right, rgb, model):
    "For a pair of 2 Calibrated Images"
    #Reading the Left Images
    # img_left_gray = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    # # Reading the right Images
    # img_right_gray = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
    # Compute the Disparity Map
    disparity = compute_disparity(img_left, img_pair=img_right, num_disparities=112, block_size=5, window_size=7, matcher="stereo_sgbm", show_disparity=False)
    # Get the Calibration Parameters
    #k_left, r_left, t_left = decompose_projection_matrix(p_left)
    #k_right, r_right, t_right = decompose_projection_matrix(p_right)
    # Compute the Depth Map
    depth_map = calc_depth_map(disparity)
    # Run obstacle detection in 2D
    result, pred_bboxes = run_obstacle_detection(rgb, model)
    # Find the Distance
    depth_list = find_distances(depth_map, pred_bboxes, img_left)
    # Final Image
    final = add_depth(depth_list, result, pred_bboxes)
    return final, result, depth_map


SHAPE = 640

# Creating the pipeline object
pipeline = dai.Pipeline()

# Creating the rgb camera node
cam = pipeline.create(dai.node.ColorCamera)

# configuring camera properties
cam.setPreviewSize(SHAPE, SHAPE)
cam.setBoardSocket(dai.CameraBoardSocket.RGB)
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam.setInterleaved(True)
# cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

# Send camera output to the host via XLink
Xout = pipeline.create(dai.node.XLinkOut)
cam.preview.link(Xout.input)
Xout.setStreamName("out")

monoR = pipeline.create(dai.node.MonoCamera)
monoR.setBoardSocket(dai.CameraBoardSocket.RIGHT)
monoR.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)

XoutMR = pipeline.create(dai.node.XLinkOut)
monoR.out.link(XoutMR.input)
XoutMR.setStreamName("out_mono_right")

monoL = pipeline.create(dai.node.MonoCamera)
monoL.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoL.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)

XoutML = pipeline.create(dai.node.XLinkOut)
monoL.out.link(XoutML.input)
XoutML.setStreamName("out_mono_left")

video_result = cv2.VideoWriter('C:\\Python Files\\OAK_D\\depthai-main\\potholedetection.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, (SHAPE, SHAPE))
# rgb_result = cv2.VideoWriter('C:\\Python Files\\OAK_D\\depthai-main\\rgb.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, (SHAPE, SHAPE))
counter = 0
model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:\\Python Files\\yolov5\\yolov5\\best.pt')
# model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:\\Jupyter_Notebooks\\PyTorchImplementation\\best.pt')

with dai.Device(pipeline) as device:

      # Op/Ip queues
      Out = device.getOutputQueue(name="out", maxSize=4, blocking=False)
      OutRight = device.getOutputQueue(name="out_mono_right", maxSize=4, blocking=False)
      OutLeft = device.getOutputQueue(name="out_mono_left", maxSize=4, blocking=False)

      print("Starting camera output...")
        
    #   cv2.namedWindow("Resized_Window", cv2.WINDOW_NORMAL)
  
      # Using resizeWindow()
    #   cv2.resizeWindow("Resized_Window", 1000, 800)

      while(True):
        RGB = Out.get()
        RGB_Frame = RGB.getFrame()  
        
        frame = OutRight.get()
        Right = frame.getCvFrame()
        
        frame1 = OutLeft.get()
        Left = frame1.getFrame()

        final, result, depth_map = perception_pipeline(Left, Right, RGB_Frame, model)
        video_result.write(RGB_Frame)
        # rgb_result.write(final)
        cv2.imshow("Resized_Window", final)
        key = cv2.waitKey(1)
        # if key == ord('s'):
        #     counter += 1
            # cv2.imwrite("./saved_images/frame%d.jpg" % counter, image)
        if key == ord('q'):
          break

# rgb_result.release()
video_result.release()
cv2.destroyAllWindows()
