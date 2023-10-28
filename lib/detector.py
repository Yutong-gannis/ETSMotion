import cv2
import time
import numpy as np
import onnxruntime
from lib.utils import xywh2xyxy,  multiclass_nms


class YOLOv8:
    def __init__(self, path):
        self.session = onnxruntime.InferenceSession(path, providers=['CUDAExecutionProvider',
                                                                     'CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.conf_threshold = 0.3
        self.iou_threshold = 0.5
        self.input_shape = (300, 500)
        self.size = (640, 640)
        self.classes = ['car', 'bus', 'truck', 'tl_green', 'tl_red', 'tl_yellow']
        
    def infer(self, img):
        t0 = time.time()
        img = self.preprocess(img)
        t1 = time.time()
        # print('pre: ', t1 - t0)
        outputs = self.session.run([self.output_name], {self.input_name: img})
        t2 = time.time()
        # print('model: ', t2 - t1)
        dets = self.process_output(outputs)
        t3 = time.time()
        # print('post: ', t3 - t2)
        if dets == [] or dets is None:
            stop = 0
        else:
            stop = 1
        return stop

    def preprocess(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.size[1], self.size[0]))
        img = img / 255.0
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0).astype(np.float32)
        return img

    def process_output(self, output):
        predictions = np.squeeze(output[0]).T
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]
        if len(scores) == 0:
            return None
        class_ids = np.argmax(predictions[:, 4:], axis=1)
        boxes = self.extract_boxes(predictions)
        indices = multiclass_nms(boxes, scores, class_ids, self.iou_threshold)
        boxes, scores, class_ids = boxes[indices], scores[indices], class_ids[indices]
        objects = []
        objects_ids = []
        objects_scores = []
        for i in range(len(class_ids)):
            if class_ids[i] == 4:
                objects.append(boxes[i])
                objects_ids.append(class_ids[i])
                objects_scores.append(scores[i])
        objects = np.array(objects)
        return objects

    def extract_boxes(self, predictions):
        boxes = predictions[:, :4]
        boxes = self.rescale_boxes(boxes)
        boxes = xywh2xyxy(boxes)
        return boxes

    def rescale_boxes(self, boxes):
        input_shape = np.array([self.size[1], self.size[0], self.size[1], self.size[0]])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([self.input_shape[1], self.input_shape[0], self.input_shape[1], self.input_shape[0]])
        return boxes