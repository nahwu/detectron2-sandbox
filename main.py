from Detector import *

detector = Detector(modelType="OBJECT_DETECTION - Faster R-CNN")
detector.onImage("images/street1.jpg")

detector = Detector(modelType="INSTANCE_SEGMENTATION - Mask R-CNN")
detector.onImage("images/street2.webp")

detector = Detector(modelType="KEYPOINT - R-CNN")
detector.onImage("images/street2.webp")

detector = Detector(modelType="LVIS - Mask R-CNN")
detector.onImage("images/street2.webp")

detector = Detector(modelType="PANOPTIC_SEGMENTATION - FPN")
detector.onImage("images/street2.webp")

detector = Detector(modelType="PANOPTIC_SEGMENTATION - FPN")
detector.onVideo("images/face_person_test_video_loop_aws.mp4")
