import numpy as np
import os
import sys
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
import pylab
import imageio
imageio.plugins.ffmpeg.download()
import skimage
sys.path.append("..")
from utils import label_map_util
from utils import visualization_utils as vis_util

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = './output_frame/mo/model/frozen_inference_graph.pb'
video_PATH = "./testing/2017_700.mp4"              # 要检测的视频
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'frame_label_map.pbtxt')

NUM_CLASSES = 1

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

vid = imageio.get_reader(video_PATH,'ffmpeg')
#cap = cv2.VideoCapture(video_PATH)
#cap = cv2.VideoCapture(0)
with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
      for num,im in enumerate(vid):
          print(im.mean())
          image_np = skimage.img_as_float(im).astype(np.float64)

          # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
          image_np_expanded = np.expand_dims(image_np, axis=0)
          image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
          # Each box represents a part of the image where a particular object was detected.
          boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
          # Each score represent how level of confidence for each of the objects.
          # Score is shown on the result image, together with the class label.
          scores = detection_graph.get_tensor_by_name('detection_scores:0')
          classes = detection_graph.get_tensor_by_name('detection_classes:0')
          num_detections = detection_graph.get_tensor_by_name('num_detections:0')
          # Actual detection.
          (boxes, scores, classes, num_detections) = sess.run(
           [boxes, scores, classes, num_detections],
           feed_dict={image_tensor: image_np_expanded})
          # Visualization of the results of a detection.
          vis_util.visualize_boxes_and_labels_on_image_array(
              image_np,
              np.squeeze(boxes),
              np.squeeze(classes).astype(np.int32),
              np.squeeze(scores),
              category_index,
              use_normalized_coordinates=True,
              line_thickness=8)
          fig = pylab.figure()
          fig.subtitle('image #{}'.format(num),fontsize = 20)
          pylab.imshow(image_np)
          #cv2.imshow('object detection', cv2.resize(image_np, (800,600)))
   #      if cv2.waitKey(25) 0xFF == ord('q'):
   #         cv2.destroyAllWindows()
  #          break