import os
import cv2
import time
import numpy as np
import tensorflow as tf
import xml.etree.ElementTree as ET
from nms import non_max_supression_fast

FROZEN_GRAPH = '/path/to/frozenGraph'
IMGS = '/path/to/images'
ERROR_ACCEPTED = 10
MIN_PRECISSION = 0.7

# Read the graph.
with tf.gfile.GFile(FROZEN_GRAPH, 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())

with tf.Session() as sess:
    idx = 0
    acum = 0

    # Restore session
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')

    for filename in os.listdir(IMGS): 
        # Dont use .DS_Store file for evaluation
        if ".DS_Store" in filename:
            continue

        attrs = []           
        s = '{s:{c}^{n}}'.format(s=idx, n=4, c=' ') + ' | '
        s = s + '{s:{c}^{n}}'.format(s=filename, n=8, c=' ') + ' | '
        path = os.path.join(IMGS, filename)
        print(path)
        # Read one image
        img = cv2.imread(path)
        rows = img.shape[0]
        cols = img.shape[1]
        # Resize to models input size
        inp = cv2.resize(img, (300, 300))
        inp = inp[:, :, [2, 1, 0]] # BGR2RGB

        # For time metrics only
        start_prediction = time.time()
        # Run the model
        out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                        sess.graph.get_tensor_by_name('detection_scores:0'),
                        sess.graph.get_tensor_by_name('detection_boxes:0'),
                        sess.graph.get_tensor_by_name('detection_classes:0')],
                        feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})

        # Get bounding boxes.
        num_detections = int(out[0][0])

        s = s + '{s:{c}^{n}}'.format(s=num_detections, n=4, c=' ') + ' | '

        boxes = []

        # Check every detection
        for i in range(num_detections):
            score = float(out[1][0][i])
            # Discard all detections bellow MIN_PRECISSION
            if (score > MIN_PRECISSION):
                s = s + '{s:{c}^{n}}'.format(s="S: " + str(score), n=4, c=' ') + ' | '
                bbox = [float(v) for v in out[2][0][i]]
                x = bbox[1] * cols
                y = bbox[0] * rows
                right = bbox[3] * cols
                bottom = bbox[2] * rows
                bbox = np.array([x, y, right, bottom])
                boxes.append((x, y, right, bottom))

        # Get one bounding box from all valid detections
        box = non_max_supression_fast(np.array(boxes), 0.3)
        for x, y, right, bottom in box:
            # Crop the predictions from the original image
            crop_img = img[int(y):int(bottom), int(x):int(right)]
            # Draw a rectangle over the image for the prediction
            cv2.rectangle(img, (int(x), int(y)), (int(right), int(bottom)), (125, 255, 51), thickness=2)
            acum += 1
                  
        # Finish prediction time
        end_prediction = time.time()
        idx += 1
        s = s + 'T: ' + str(round((end_prediction - start_prediction), 2)) + ' | '

        print(s)
        s = s + '\n'

        # Show results
        cv2.imshow('Cropped', crop_img)
        cv2.imshow('Face detection', img)
        cv2.waitKey(0)