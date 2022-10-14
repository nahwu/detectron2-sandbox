from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2 import model_zoo

import cv2
import numpy as np
from datetime import datetime

class Detector:
    def __init__(self, modelType):
        self.cfg = get_cfg()
        self.model_type = modelType

        # Load model config and pretrained model
        if modelType == "OBJECT_DETECTION":
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
        elif modelType == "INSTANCE_SEGMENTATION":
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
        elif modelType == "KEYPOINT":
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
        elif modelType == "LVIS":
            self.cfg.merge_from_file(model_zoo.get_config_file("LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml")
        elif modelType == "PANOPTIC_SEGMENTATION":
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
        else:
            print("Unknown Model Type. Default set to OBJECT_DETECTION")

        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        self.cfg.MODEL.DEVICE = "cpu"  # cpu or cuda

        self.predictor = DefaultPredictor(self.cfg)

    def onImage(self, imagePath):
        print("Image Function called: ", self.model_type , datetime.now())

        image = cv2.imread(imagePath)

        if self.model_type != "PANOPTIC_SEGMENTATION":
            predictions = self.predictor(image)

            print("Predictions ready: ", datetime.now())
            print(predictions)

            viz = Visualizer(image[:, :, ::-1], metadata=MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), instance_mode=ColorMode.IMAGE_BW)

            output = viz.draw_instance_predictions(predictions["instances"].to("cpu"))
        else:
            predictions, segmentInfo = self.predictor(image)["panoptic_seg"]

            print("Predictions ready: ", datetime.now())
            print(predictions)

            viz = Visualizer(image[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]))
            output = viz.draw_panoptic_seg_predictions(predictions.to("cpu"), segmentInfo)

        print("Display ready: ", datetime.now())
        cv2.imshow("Result - " + self.model_type, output.get_image()[:, :, ::-1])
        cv2.waitKey(0)
        