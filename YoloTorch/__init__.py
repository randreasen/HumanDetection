import torch
from  torch.autograd import Variable
from .darknet import Darknet
from .util import load_classes, write_results
from .preprocess import prep_image
import pkg_resources
import numpy as np

resource_package = __name__
name_file = pkg_resources.resource_stream(resource_package, 'resources/coco.names').name
cfg_file = pkg_resources.resource_stream(resource_package, 'cfg/yolov3.cfg').name
weight_file = pkg_resources.resource_stream(resource_package, 'resources/yolov3.weights').name


class YoloTorch:
	def __init__(self, cfgfile=cfg_file, weightsfile=weight_file, reso=416, confidence=0.5, nms_thesh=0.4):
		self.CUDA = torch.cuda.is_available()
		self.classes = load_classes(name_file)
		self.nb_c = len(self.classes)
		self.Td = confidence
		self.Tnms = nms_thesh

		self.model = Darknet(cfgfile)
		self.model.load_weights(weightsfile)
		self.model.net_info["height"] = str(reso)
		self.inp_dim = reso
		assert self.inp_dim % 32 == 0
		assert self.inp_dim > 32
		if self.CUDA:
			self.model.cuda()
		self.model.eval()

	def rescale(self, out, shape):
		def tlwh(box):
			x1, y1, x2, y2 = box
			return np.array([x1 , y1, (x2 - x1), (y2 - y1)], dtype=int)

		res = write_results(out, self.Td, self.nb_c, nms=True, nms_conf=self.Tnms)
		scaling_factor = torch.min(self.inp_dim / shape)
		res[:, [1, 3]] -= (self.inp_dim - scaling_factor * shape[0]) / 2
		res[:, [2, 4]] -= (self.inp_dim - scaling_factor * shape[1]) / 2
		res[:, 1:5] /= scaling_factor
		detections = []
		for i in range(res.shape[0]):
			res[i, [1, 3]] = torch.clamp(res[i, [1, 3]], 0.0, shape[0])
			res[i, [2, 4]] = torch.clamp(res[i, [2, 4]], 0.0, shape[1])
			d = dict(tlwh=tlwh(res[i, 1:5].cpu().numpy()), label=self.classes[int(res[i, -1])], score=float(res[i, 5]),mask=None)
			detections.append(d)
		return detections

	def detect(self, orig_im):
		img, shape = prep_image(orig_im, self.inp_dim)
		shape = torch.FloatTensor(shape)
		if self.CUDA:
			img=Variable(img.cuda())
		with torch.no_grad():
			out = self.model(img, self.CUDA)
		return self.rescale(out, shape)