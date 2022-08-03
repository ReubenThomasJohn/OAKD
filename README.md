## STEPS TO RUN REAL-TIME NEURAL INFERENCE ON OAK-D

### Assumptions:
1. Custom neural network is to be run on the Intel Neural compute stick of the OAK-D
2. We run a custom YOLOv5 object detection network
3. Pytorch is being used for the training

### 1.	Convert ```best.pt``` to ```best.onnx```

The trained network needs to be converted to a. blob format for it to be able to run on the OAK-D device. Since the best.pt file from Pytorch cannot be converted to. blob directly, it first needs to be converted to the ONNX format. Fortunately, Pytorch has an inbuilt module that makes this conversion simple.

Refer [documentation](https://docs.microsoft.com/en-us/windows/ai/windows-ml/tutorials/pytorch-convert-model) here. 

```
import torch.onnx 

#Function to Convert to ONNX 
def Convert_ONNX(): 

    # set the model to inference mode 
    model.eval() 

    # Let's create a dummy input tensor  
    dummy_input = torch.randn(1, input_size, requires_grad=True)  

    # Export the model   
    torch.onnx.export(model,         # model being run 
         dummy_input,       # model input (or a tuple for multiple inputs) 
         "ImageClassifier.onnx",       # where to save the model  
         export_params=True,  # store the trained parameter weights inside the model file 
         opset_version=10,    # the ONNX version to export the model to 
         do_constant_folding=True,  # whether to execute constant folding for optimization 
         input_names = ['modelInput'],   # the model's input names 
         output_names = ['modelOutput'], # the model's output names 
         dynamic_axes={'modelInput' : {0 : 'batch_size'},    # variable length axes 
                                'modelOutput' : {0 : 'batch_size'}}) 
    print(" ") 
    print('Model has been converted to ONNX')
```

### 2. Simplify ```best.onnx``` model

The conversion to onnx is not always optimized. Often, a large number of redundant and useless will be added during the conversion. It is imperative that these layers are removed. Else, they lead to reduced inference times. 

```
import onnx
from onnxsim import simplify

onnx_model = onnx.load("path/to/model.onnx")
model_simpified, check = simplify(onnx_model)
onnx.save(model_simpified, "path/to/simplified/model.onnx")
```
Refer [documentation](https://docs.luxonis.com/en/latest/pages/tutorials/creating-custom-nn-models/#run-your-own-cv-functions-on-device) here. 

### 3. Convert ```best.onnx``` to OpenVINO ```best.blob``` format

For this step, we require the blobconverter module. This is the last step in our module conversion process. After this step, we will obtain our trained network in a format that is suitable to be used on the OAK-D compute hardware. 

```
import blobconverter

blobconverter.from_onnx(
    model="/path/to/model.onnx",
    output_dir="/path/to/output/model.blob",
    data_type="FP16",
    shaves=6,
    use_cache=False,
    optimizer_params=[]
)
```
Refer [documentation](https://docs.luxonis.com/en/latest/pages/tutorials/creating-custom-nn-models/#run-your-own-cv-functions-on-device) here. 

### 4. Obtain streams from OAK camera

We now need to write the code, to obtain the streams from the camera, and have our network use these streams. We also need to make the network run on the OAK device itself, and not on our host device (computer in this case). For this purpose, we will use the ```depth.ai``` API provided by Luxonis. 

I have documented here, a basic program - to create nodes to create a communication between the OAK and the host computer. 

```
import depthai as dai
import cv2
import numpy as np

SHAPE = 500

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

result = cv2.VideoWriter('rgb_stream.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, (SHAPE, SHAPE))
with dai.Device(pipeline) as device:

  # Op/Ip queues
  Out = device.getOutputQueue(name="out", maxSize=4, blocking=False)

  print("Starting camera output...")

  while(True):
    RGB = Out.get()

    RGB_Frame = RGB.getFrame()    
    print(RGB_Frame.shape)         
    cv2.imshow("out", RGB_Frame)
    result.write(RGB_Frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
      break

cv2.destroyAllWindows()
result.release()
```

