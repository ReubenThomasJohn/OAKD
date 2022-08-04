# import depthai as dai

# BlobPath = ' '
# pipeline = dai.Pipeline()
# nn = pipeline.create(dai.node.NeuralNetwork)
# cam = pipeline.create(dai.node.ColorCamera)
# nn.setBlobPath(BlobPath)
# cam.out.link(nn.input)

# # Send NN out to the host via XLink
# nnXout = pipeline.create(dai.node.XLinkOut)
# nnXout.setStreamName("nn")
# nn.out.link(nnXout.input)

# with dai.Device(pipeline) as device:
#   qNn = device.getOutputQueue("nn")

#   nnData = qNn.get() # Blocking

#   # NN can output from multiple layers. Print all layer names:
#   print(nnData.getAllLayerNames())

#   # Get layer named "Layer1_FP16" as FP16
#   layer1Data = nnData.getLayerFp16("Layer1_FP16")

#   # You can now decode the output of your NN

from convert2onnx.tool.darknet2onnx import transform_to_onnx

cfgfile = '/home/reuben/Projects/pothole-detection/learnopencv/Pothole-Detection-using-YOLOv4-and-Darknet/jupyter_notebook/darknet/cfg/tiny-yolo.cfg'
weightfile = '/home/reuben/Projects/OAKD/backup_yolov4_tiny/yolov4-tiny-pothole_final.weights'

transform_to_onnx(cfgfile, weightfile, batch_size=1, onnx_file_name='yolov4-tiny.onnx')