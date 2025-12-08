import abc
import numpy as np
import cv2
import base64
import torch
import onnxruntime as ort
from ultralytics import YOLO
from pydantic import BaseModel

# --- Data Models ---

class Detection(BaseModel):
    id: int
    box: list[int]          # [x_min, y_min, x_max, y_max]
    score: float
    polygon: list[list[int]]  # List of [x, y] coordinate pairs

class NailImage(BaseModel):
    id: int
    score: float
    image_base64: str # Base64 encoded PNG image

# --- Abstract Base Class ---

class NailSegmenter(abc.ABC):
    @abc.abstractmethod
    def load_model(self):
        pass

    @abc.abstractmethod
    def segment(self, image: np.ndarray) -> list[Detection]:
        pass

    @abc.abstractmethod
    def extract_nails(self, image: np.ndarray) -> list[NailImage]:
        pass

# --- Ultralytics Implementation ---

class UltralyticsSegmenter(NailSegmenter):
    def __init__(self, model_path: str, conf_threshold: float = 0.25, iou_threshold: float = 0.45, img_size: int = 640):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size
        self.model = None
        self.device = None

    def load_model(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = YOLO(self.model_path)
        self.model.model.to(self.device).eval()
        print(f"✅ Ultralytics YOLOv8 Model loaded on {self.device}.")

    def _process_results(self, results):
        final_data = []
        if not results or not results[0].masks:
            return []

        result = results[0]
        H_orig, W_orig = result.orig_shape
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        masks_data = result.masks.data.cpu().numpy()

        for i in range(len(boxes)):
            box_xyxy = boxes[i].tolist()
            mask_640 = masks_data[i]
            
            # Resize mask
            mask_orig_size_float = cv2.resize(mask_640, (W_orig, H_orig), interpolation=cv2.INTER_CUBIC)
            mask_255 = (mask_orig_size_float * 255).astype(np.uint8)

            # Morphology (Original logic)
            kernel = np.ones((5, 5), np.uint8)
            mask_stable = cv2.morphologyEx(mask_255, cv2.MORPH_CLOSE, kernel)
            mask_blurred = cv2.GaussianBlur(mask_stable, (9, 9), 0)
            _, mask_binary_final = cv2.threshold(mask_blurred, 50, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

            contours, _ = cv2.findContours(mask_binary_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
            polygon_list = []
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                epsilon = 0.001 * cv2.arcLength(largest_contour, True)
                simplified_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
                polygon_list = simplified_contour.squeeze(axis=1).astype(int).tolist()

            if not polygon_list:
                continue

            final_data.append(Detection(
                id=i,
                box=[int(b) for b in box_xyxy],
                score=float(scores[i]),
                polygon=polygon_list
            ))
        return final_data

    def segment(self, image: np.ndarray) -> list[Detection]:
        if self.model is None:
            raise RuntimeError("Model not loaded.")
        
        results = self.model.predict(
            source=image,
            imgsz=self.img_size,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
            device=self.device.type,
            retina_masks=True
        )
        return self._process_results(results)

    def extract_nails(self, image: np.ndarray) -> list[NailImage]:
        if self.model is None:
            raise RuntimeError("Model not loaded.")

        results = self.model.predict(
            source=image,
            imgsz=self.img_size,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
            device=self.device.type,
            retina_masks=True
        )
        
        # Re-implementing the latest iteration logic (Iteration 3)
        extracted_nails = []
        if not results or not results[0].masks:
            return []

        result = results[0]
        H_orig, W_orig = result.orig_shape
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        masks_data = result.masks.data.cpu().numpy()

        for i in range(len(boxes)):
            box_xyxy = boxes[i].tolist()
            mask_640 = masks_data[i]
            
            # Iteration 3 Logic: INTER_LINEAR, No Morphology
            mask_orig_size_float = cv2.resize(mask_640, (W_orig, H_orig), interpolation=cv2.INTER_LINEAR)
            mask_255 = (mask_orig_size_float * 255).astype(np.uint8)
            mask_blurred = cv2.GaussianBlur(mask_255, (11, 11), 0)

            b, g, r = cv2.split(image)
            rgba = [b, g, r, mask_blurred]
            dst = cv2.merge(rgba, 4)
            
            x1, y1, x2, y2 = [int(coord) for coord in box_xyxy]
            
            # Expansion
            expansion = 25
            x1 = max(0, x1 - expansion)
            y1 = max(0, y1 - expansion)
            x2 = min(W_orig, x2 + expansion)
            y2 = min(H_orig, y2 + expansion)
            
            cropped_nail = dst[y1:y2, x1:x2]
            _, buffer = cv2.imencode('.png', cropped_nail)
            png_as_text = base64.b64encode(buffer).decode('utf-8')
            
            extracted_nails.append(NailImage(
                id=i,
                score=float(scores[i]),
                image_base64=png_as_text
            ))
            
        return extracted_nails

# --- ONNX Implementation ---

class OnnxSegmenter(NailSegmenter):
    def __init__(self, model_path: str, conf_threshold: float = 0.25, iou_threshold: float = 0.45, img_size: int = 640):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size
        self.sess = None
        self.input_name = None
        self.output_names = None
        self.num_classes = 1
        self.feather_radius = 5
        self.box_expansion = 25
        self.mask_threshold = 0.5

    def load_model(self):
        self.sess = ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.sess.get_inputs()[0].name
        self.output_names = [output.name for output in self.sess.get_outputs()]
        print(f"✅ ONNX Model loaded from {self.model_path}.")

    def _preprocess(self, image: np.ndarray):
        img_resized = cv2.resize(image, (self.img_size, self.img_size))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_transposed = np.transpose(img_rgb, (2, 0, 1))
        input_tensor = np.expand_dims(img_transposed, axis=0)
        input_tensor_float = input_tensor.astype(np.float32) / 255.0
        return input_tensor_float

    def _process_detection_output(self, output):
        # Transpose from (1, 84, 8400) to (1, 8400, 84)
        output = output.transpose(0, 2, 1)[0]
        boxes = output[:, :4]
        scores = output[:, 4:4 + self.num_classes]
        mask_coeffs = output[:, 4 + self.num_classes:]
        
        max_scores = np.max(scores, axis=1) 
        valid_indices = max_scores > self.conf_threshold

        if not np.any(valid_indices):
            return np.array([]), np.array([]), np.array([]) 

        boxes = boxes[valid_indices]
        mask_coeffs = mask_coeffs[valid_indices]
        
        # Convert boxes from (center_x, center_y, width, height) to (x1, y1, x2, y2)
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
        
        max_scores_valid = max_scores[valid_indices]
        
        boxes_int = (boxes_xyxy * 1000).astype(np.int32) 
        indices = cv2.dnn.NMSBoxes(
            boxes_int, 
            max_scores_valid.astype(np.float32), 
            self.conf_threshold, 
            self.iou_threshold
        )
        
        if len(indices) == 0:
            return np.array([]), np.array([]), np.array([])
        
        indices = indices.flatten()
        
        final_boxes = boxes_xyxy[indices]
        final_scores = max_scores_valid[indices]
        final_mask_coeffs = mask_coeffs[indices]

        return final_boxes, final_scores, final_mask_coeffs

    def _generate_masks(self, boxes_xyxy, mask_coeffs, proto_output, img_original_shape):
        if len(boxes_xyxy) == 0:
            return []

        proto = proto_output[0] 
        proto_flat = proto.reshape(proto.shape[0], -1) 
        masks_flat = mask_coeffs @ proto_flat 
        masks_flat = 1 / (1 + np.exp(-masks_flat)) # Sigmoid
        masks = masks_flat.reshape(-1, 160, 160) 

        final_masks_data = []
        H_orig, W_orig = img_original_shape[:2]

        for i, mask_160 in enumerate(masks):
            mask_640 = cv2.resize(mask_160, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            x1_640, y1_640, x2_640, y2_640 = boxes_xyxy[i].astype(int)
            
            # Mask Cropping
            cropped_soft_mask_640 = np.zeros((self.img_size, self.img_size), dtype=np.float32) 
            x1_640 = max(0, x1_640)
            y1_640 = max(0, y1_640)
            x2_640 = min(self.img_size, x2_640)
            y2_640 = min(self.img_size, y2_640)
            cropped_soft_mask_640[y1_640:y2_640, x1_640:x2_640] = mask_640[y1_640:y2_640, x1_640:x2_640]
            
            # Feathering
            alpha_mask_255 = (cropped_soft_mask_640 * 255).astype(np.uint8)
            if self.feather_radius > 0:
                alpha_mask_255 = cv2.GaussianBlur(alpha_mask_255, (self.feather_radius * 2 + 1, self.feather_radius * 2 + 1), 0)
            
            # Soft clip
            alpha_mask_255[alpha_mask_255 < (self.mask_threshold * 255 * 0.5)] = 0
            
            # Map to original size
            x1_orig = int(x1_640 * W_orig / self.img_size)
            y1_orig = int(y1_640 * H_orig / self.img_size)
            x2_orig = int(x2_640 * W_orig / self.img_size)
            y2_orig = int(y2_640 * H_orig / self.img_size)

            final_alpha_mask_orig = cv2.resize(alpha_mask_255, (W_orig, H_orig), interpolation=cv2.INTER_LINEAR)
            
            final_masks_data.append({
                'box': (x1_orig, y1_orig, x2_orig, y2_orig),
                'soft_mask': final_alpha_mask_orig,
                'score': boxes_xyxy[i] 
            })
            
        return final_masks_data

    def segment(self, image: np.ndarray) -> list[Detection]:
        if self.sess is None:
            raise RuntimeError("Model not loaded.")

        input_tensor = self._preprocess(image)
        onnx_outputs = self.sess.run(self.output_names, {self.input_name: input_tensor})
        
        boxes, scores, mask_coeffs = self._process_detection_output(onnx_outputs[0])
        final_masks_data = self._generate_masks(boxes, mask_coeffs, onnx_outputs[1], image.shape)
        
        detections = []
        for i, data in enumerate(final_masks_data):
            # Generate polygon from soft mask
            # Threshold to binary for polygon
            _, mask_binary = cv2.threshold(data['soft_mask'], 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
            polygon_list = []
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                epsilon = 0.002 * cv2.arcLength(largest_contour, True)
                simplified_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
                polygon_list = simplified_contour.squeeze(axis=1).astype(int).tolist()
            
            if not polygon_list:
                continue

            detections.append(Detection(
                id=i,
                box=list(data['box']),
                score=float(scores[i]),
                polygon=polygon_list
            ))
        return detections

    def extract_nails(self, image: np.ndarray) -> list[NailImage]:
        if self.sess is None:
            raise RuntimeError("Model not loaded.")

        input_tensor = self._preprocess(image)
        onnx_outputs = self.sess.run(self.output_names, {self.input_name: input_tensor})
        
        boxes, scores, mask_coeffs = self._process_detection_output(onnx_outputs[0])
        final_masks_data = self._generate_masks(boxes, mask_coeffs, onnx_outputs[1], image.shape)
        
        extracted_nails = []
        H_orig, W_orig = image.shape[:2]
        
        for i, data in enumerate(final_masks_data):
            x1, y1, x2, y2 = data['box']
            soft_mask = data['soft_mask']
            
            # Expansion
            expand = self.box_expansion
            x1_exp = x1 - expand
            y1_exp = y1 - expand
            x2_exp = x2 + expand
            y2_exp = y2 + expand
            
            x1_crop = max(0, x1_exp)
            y1_crop = max(0, y1_exp)
            x2_crop = min(W_orig, x2_exp)
            y2_crop = min(H_orig, y2_exp)
            
            cropped_nail_bgr = image[y1_crop:y2_crop, x1_crop:x2_crop]
            cropped_alpha_mask = soft_mask[y1_crop:y2_crop, x1_crop:x2_crop]
            
            cutout_rgba = np.zeros((cropped_nail_bgr.shape[0], cropped_nail_bgr.shape[1], 4), dtype=np.uint8)
            cutout_rgba[:, :, 0:3] = cropped_nail_bgr
            cutout_rgba[:, :, 3] = cropped_alpha_mask
            
            _, buffer = cv2.imencode('.png', cutout_rgba)
            png_as_text = base64.b64encode(buffer).decode('utf-8')
            
            extracted_nails.append(NailImage(
                id=i,
                score=float(scores[i]),
                image_base64=png_as_text
            ))
            
        return extracted_nails
