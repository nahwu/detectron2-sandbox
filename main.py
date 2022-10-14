from Detector import *

detector = Detector(modelType="OBJECT_DETECTION")
detector.onImage("images/street1.jpg")

detector = Detector(modelType="INSTANCE_SEGMENTATION")
detector.onImage("images/street2.webp")

detector = Detector(modelType="KEYPOINT")
detector.onImage("images/street2.webp")

detector = Detector(modelType="LVIS")
detector.onImage("images/street2.webp")

detector = Detector(modelType="PANOPTIC_SEGMENTATION")
detector.onImage("images/street2.webp")
