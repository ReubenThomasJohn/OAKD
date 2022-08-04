import cv2
import numpy as np

configs = {'INPUT_WIDTH':640,
        'INPUT_HEIGHT':640,
        'CONFIDENCE_THRESHOLD':.4,
        'SCORE_THRESHOLD':.5,
        'NMS_THRESHOLD':.3}

classesFile = "./Tracking_DeepSORT/data/labels/coco.names"
classes = None
# A handy way to read all the classes from a file, without needed to hardcode each one
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

print("Number of Classes: ", len(classes))

def post_process(outputs, image):
    '''
    One forward pass gives us an output as an array, that has a shape=(25200, no.of classes). That is, for each image,
    25200 predictions, per class are made. We need to extract only the useful and important information from this massive
    array. We use the post_process() function for this. 
    '''
    # Lists to hold respective values while unwrapping.
    class_ids = []
    confidences = []
    boxes = []
    classes_scores = []

    # Rows.
    rows = outputs[0].shape[1]

    image_height, image_width = image.shape[:2]

    # Resizing factor.
    x_factor = image_width / configs['INPUT_WIDTH']
    y_factor =  image_height / configs['INPUT_HEIGHT']

    # Iterate through 25200 detections.
    for r in range(rows):
        row = outputs[0][0][r]
        confidence = row[4]

        # Discard bad detections and continue.
        if confidence >= configs['CONFIDENCE_THRESHOLD']:
            classes_scores = row[5:]

            # Get the index of max class score.
            class_id = np.argmax(classes_scores)

            #  Continue if the class score is above threshold.
            if (classes_scores[class_id] > configs['SCORE_THRESHOLD']):
                confidences.append(confidence)
                class_ids.append(class_id)
                classes_scores.append(classes_scores[class_id])

                cx, cy, w, h = row[0], row[1], row[2], row[3]

                left = int((cx - w/2) * x_factor) 
                top = int((cy - h/2) * y_factor)
                right = int((cx + w/2) * x_factor)
                bottom = int((cy + h/2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                boxes.append(box)

    boxes = np.array(boxes)  # nms needs tlbr not tlwh 

    return boxes, confidences

def tlwh2tlbr(boxes):
    '''
    A func that converts bboxes in the tlwh (top left, width height) format to the 
    tlbr format (top left, bottom right)
    '''
    tlbr_boxes = boxes.copy()
    try:
        tlbr_boxes[:,2:] += boxes[:,:2]
    except:
        tlbr_boxes = []

    return tlbr_boxes 

def tlbr2tlwh(boxes):
    '''
    A function to convert tlbr to tlwh bboxes
    '''
    tlwh_boxes = boxes.copy()
    tlwh_boxes[:,2:] -= boxes[:,:2]
    return tlwh_boxes


def non_max_suppression_fast(boxes):
    '''
    A quick and efficient way to perform NMS using vectorization.
    This function requires the bboxes to be in the tlbr format
    '''
    # print('Rejecting overlapping boxes...')
    boxes = tlwh2tlbr(boxes)
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    # initialize the list of picked indexes	
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
    # grab the last index in the indexes list and add the
    # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]] 

        # # Supress/hide the warning
        # np.seterr(invalid='ignore')
        
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                np.where(overlap > configs['NMS_THRESHOLD'])[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    boxes = boxes[pick].astype("int")
    boxes = tlbr2tlwh()
    class_ids = np.array(class_ids)[pick]
    assert len(boxes) == len(class_ids) # check if class_ids are available!!, classes**
    classes_scores = np.array(classes_scores)[pick]
    picks = pick 

    return picks, class_ids
    
def drawNMSBoxes(boxes, picks, class_ids, confidences, classes, image):
    boxes = tlwh2tlbr(boxes)
    for box, pick, classId in zip(boxes, picks, class_ids):
        label = '%.2f' % (confidences[pick])
        if len(class_ids)>0:
            assert(classId < len(classes))
            labeltoDraw = '%s:%s' % (classes[classId], label)
#       box = boxes[i]
        left = box[0]
        top = box[1]
        right = box[2]
        bottom = box[3]
        cv2.rectangle(image, (left, top), (right, bottom), (255,255,0), 3)

        #Display the label at the top of the bounding box
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        cv2.putText(image, labeltoDraw, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), thickness=1)
        
        # # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
        # t, _ = self.net.getPerfProfile()
        # label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
        # cv2.putText(self.image, label, (20, 40), self.FONT_FACE, self.FONT_SCALE, self.YELLOW, 1, cv2.LINE_AA)    
    
def object_detection(self, input_image, visualise=False):
    self.pre_process(input_image)
    self.post_process()      
    self.non_max_suppression_fast()
    if visualise == True:
        self.drawNMSBoxes()

    return self.boxes, np.array([self.classes_scores]), np.array([self.class_ids]), len(self.class_ids)


    # Convert yolov5s.pt to blob and check if objects are being detected well