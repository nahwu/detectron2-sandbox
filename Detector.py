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
        if modelType == "OBJECT_DETECTION - Faster R-CNN":
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
        elif modelType == "INSTANCE_SEGMENTATION - Mask R-CNN":
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
        elif modelType == "KEYPOINT - R-CNN":
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
        elif modelType == "LVIS - Mask R-CNN":
            self.cfg.merge_from_file(model_zoo.get_config_file("LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml")
        elif modelType == "PANOPTIC_SEGMENTATION - FPN":
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
        else:
            print("Unknown Model Type. Default set to OBJECT_DETECTION")

        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        self.cfg.MODEL.DEVICE = "cpu"  # cpu or cuda

        self.predictor = DefaultPredictor(self.cfg)

    def onImage(self, imagePath):
        print("Image Function called: ", self.model_type, datetime.now())

        image = cv2.imread(imagePath)

        output = self.prepareOpenCV2Display(image)

        print("Display ready: ", datetime.now())
        cv2.imshow("Result - " + self.model_type, output.get_image()[:, :, ::-1])
        cv2.waitKey(0)

    def onVideo(self, videoPath):
        currentFrameIndex = 0
        playRate = 5       # Process and display every X frames

        openedStream = cv2.VideoCapture(videoPath)

        if (openedStream.isOpened == False):
            print("Unable to open video file")
            return


        # Default resolutions of the frame are obtained.The default resolutions are system dependent.
        # We convert the resolutions from float to integer.
        frame_width = int(openedStream.get(3))
        frame_height = int(openedStream.get(4))
        # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
        myVideoWriter = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))


        # Retrieve frame from video file
        (playable, frame) = openedStream.read()

        while playable:
            if (currentFrameIndex % playRate == 0):

                output = self.prepareOpenCV2Display(frame)

                myVideoWriter.write(output.get_image()[:, :, ::-1])
                print("Written ", currentFrameIndex, " at ",  datetime.now())

                #cv2.imshow("Result - " + self.model_type,output.get_image()[:, :, ::-1])

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                # Stop processing video file if user presses "Q"
                break

            # Continue to next frame
            currentFrameIndex +=1 
            (playable, frame) = openedStream.read()
        
        # When done, release the video capture object
        openedStream.release()

        # Closes all the frames
        cv2.destroyAllWindows()

    def prepareOpenCV2Display(self, image):
        if self.model_type != "PANOPTIC_SEGMENTATION - FPN":
            # For non-Panoptic Segmentation display
            predictions = self.predictor(image)

            #print("Predictions ready: ", datetime.now())
            #print(predictions)

            viz = Visualizer(image[:, :, ::-1], metadata=MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), instance_mode=ColorMode.IMAGE_BW)

            return viz.draw_instance_predictions(predictions["instances"].to("cpu"))
        else:
            # For Panoptic Segmentation display
            predictions, segmentInfo = self.predictor(image)["panoptic_seg"]

            #print("Predictions ready: ", datetime.now())
            #print(predictions)

            viz = Visualizer(image[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]))

            return viz.draw_panoptic_seg_predictions(predictions.to("cpu"), segmentInfo)
