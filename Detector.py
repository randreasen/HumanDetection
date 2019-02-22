from YoloTorch import YoloTorch
import cv2
import numpy as np
class Detector:
	"""Detector
	Unique interface for Detectors
	Input:
	detection_thresh: min confidance threshold
	alog: the detection algo to use YOLO, MRCNN
	nms_thresh: min nms threshold default=0.5
	appearance: An appearance model to extract feaures for detection
	class_filter: filter detection by classes if the detector detects miltiple classes
	"""

	def __init__(self,detection_thresh=0.8,nms_thresh=0.5,class_filter=['person']):

		self.classes =class_filter
		self.model = YoloTorch(confidence=detection_thresh, nms_thesh=nms_thresh)
		print("Setting up a Detector !")

	def detect(self, img):
		"""
		In:
			img = array[nb_imgs,w,h,3]
		Out:
			detections= list(Detection) ,len={nb_imgs};
		"""

		cv2.cvtColor(img, cv2.COLOR_RGB2BGR, img)
		out =[]
		for d in self.model.detect(img):
			if d['label'] in self.classes:
				d['feature']=None
				out.append(d['tlwh'])

		return np.array(out).astype(int)
