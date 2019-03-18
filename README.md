# UAV-ZJU-TF-GATE-DETECTION
Detecting the gate with Tensorflow-object-detection-API
## Dependencies
This code based on the [Tensorflow object detection API](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md). you should install this library.
## Train
The pretrained model [SSD-MobileNet-V2](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) was used for fineturn.train this model by running this cmd:
``` bash
# From tensorflow/models/research/object_detection
python model_main.py --logtostderr --train_dir=./outputing --pipeline_config_path=./ssd_mobilenet_v2_coco.config
```
## Test
``` bash
# From tensorflow/models/research/object_detection
python video_detection.py
```
