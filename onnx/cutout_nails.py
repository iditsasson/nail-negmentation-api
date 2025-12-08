import onnxruntime as ort
import numpy as np
import cv2
import os

# --- Configuration ---
ONNX_MODEL_PATH = "nails_seg_s_yolov8_v1.onnx"
IMAGE_PATH = "nail_test_2.jpeg" # Ensure this path is correct for your design image
OUTPUT_DIR = "cutout_nails_final" 
IMG_SIZE = 640
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
MASK_THRESHOLD = 0.5
NUM_CLASSES = 1 
FEATHER_RADIUS = 5               # Adjust for smoothness (e.g., 3-10)
BOX_EXPANSION_PIXELS = 25        # Adjust for safety (e.g., 20-50)

# --- Utility Functions ---

def process_detection_output(output, conf_thres, iou_thres):
    """Applies NMS and converts raw box outputs."""
    # Transpose from (1, 84, 8400) to (1, 8400, 84)
    output = output.transpose(0, 2, 1)[0]
    boxes = output[:, :4]
    scores = output[:, 4:4 + NUM_CLASSES]
    mask_coeffs = output[:, 4 + NUM_CLASSES:]
    
    max_scores = np.max(scores, axis=1) 
    valid_indices = max_scores > conf_thres

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
        conf_thres, 
        iou_thres
    )
    
    if len(indices) == 0:
        return np.array([]), np.array([]), np.array([])
    
    indices = indices.flatten()
    
    final_boxes = boxes_xyxy[indices]
    final_scores = max_scores_valid[indices]
    final_mask_coeffs = mask_coeffs[indices]

    return final_boxes, final_scores, final_mask_coeffs

def generate_segmentation_masks(boxes_xyxy, mask_coeffs, proto_output, img_original_shape, feather_radius):
    """Generates feather-edged alpha masks and original-scale bounding boxes."""
    if len(boxes_xyxy) == 0:
        return []

    proto = proto_output[0] 
    proto_flat = proto.reshape(proto.shape[0], -1) 
    masks_flat = mask_coeffs @ proto_flat 
    masks_flat = 1 / (1 + np.exp(-masks_flat)) # Sigmoid (0.0 - 1.0)
    masks = masks_flat.reshape(-1, 160, 160) 

    final_masks_data = []
    H_orig, W_orig = img_original_shape[:2]

    for i, mask_160 in enumerate(masks):
        
        mask_640 = cv2.resize(mask_160, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)

        x1_640, y1_640, x2_640, y2_640 = boxes_xyxy[i].astype(int)
        
        # 1. Mask Cropping (Confinement)
        cropped_soft_mask_640 = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32) 
        
        x1_640 = max(0, x1_640)
        y1_640 = max(0, y1_640)
        x2_640 = min(IMG_SIZE, x2_640)
        y2_640 = min(IMG_SIZE, y2_640)
        
        cropped_soft_mask_640[y1_640:y2_640, x1_640:x2_640] = mask_640[y1_640:y2_640, x1_640:x2_640]
        
        # 2. Feathering step (Gaussian Blur on 0-255 mask)
        alpha_mask_255 = (cropped_soft_mask_640 * 255).astype(np.uint8)
        
        if feather_radius > 0:
            alpha_mask_255 = cv2.GaussianBlur(alpha_mask_255, (feather_radius * 2 + 1, feather_radius * 2 + 1), 0)
            
        # Optional: Soft clip of very faint values
        alpha_mask_255[alpha_mask_255 < (MASK_THRESHOLD * 255 * 0.5)] = 0
            
        # 3. Map 640x640 coordinates to original image size
        x1_orig = int(x1_640 * W_orig / IMG_SIZE)
        y1_orig = int(y1_640 * H_orig / IMG_SIZE)
        x2_orig = int(x2_640 * W_orig / IMG_SIZE)
        y2_orig = int(y2_640 * H_orig / IMG_SIZE)

        # 4. Resize the blurred alpha mask to the original image size
        final_alpha_mask_orig = cv2.resize(alpha_mask_255, (W_orig, H_orig), interpolation=cv2.INTER_LINEAR)
        
        final_masks_data.append({
            'box': (x1_orig, y1_orig, x2_orig, y2_orig),
            'soft_mask': final_alpha_mask_orig,
            'score': boxes_xyxy[i] 
        })
        
    return final_masks_data

# --- Main Function ---

def cutout_nails():
    # Setup
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load original image
    img_bgr = cv2.imread(IMAGE_PATH)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not load image at {IMAGE_PATH}")
    
    img_original_shape = img_bgr.shape
    H_orig, W_orig = img_original_shape[:2]
    
    # --- Pre-processing ---
    img_resized = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_transposed = np.transpose(img_rgb, (2, 0, 1)) 
    input_tensor = np.expand_dims(img_transposed, axis=0)
    input_tensor_float = input_tensor.astype(np.float32) / 255.0

    # --- ONNX Inference ---
    print("Running ONNX model inference...")
    sess = ort.InferenceSession(ONNX_MODEL_PATH, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    output_names = [output.name for output in sess.get_outputs()]

    onnx_outputs = sess.run(output_names, {input_name: input_tensor_float})

    detect_output = onnx_outputs[0]
    proto_output = onnx_outputs[1]

    # --- Post-processing ---
    print("Applying post-processing (NMS and Mask Generation)...")
    boxes, scores, mask_coeffs = process_detection_output(
        detect_output, CONF_THRESHOLD, IOU_THRESHOLD
    )
    
    final_masks_data = generate_segmentation_masks(
        boxes, mask_coeffs, proto_output, img_original_shape, FEATHER_RADIUS
    )

    # --- Cutout and Save ---
    if final_masks_data:
        for i, mask_data in enumerate(final_masks_data):
            box = mask_data['box'] # Original-scale bounding box (unexpanded)
            soft_alpha_mask_orig = mask_data['soft_mask']

            x1_orig, y1_orig, x2_orig, y2_orig = box
            
            # --- EXPAND THE BOUNDING BOX ---
            expand = BOX_EXPANSION_PIXELS
            
            # 1. Expand coordinates
            x1_exp = x1_orig - expand
            y1_exp = y1_orig - expand
            x2_exp = x2_orig + expand
            y2_exp = y2_orig + expand
            
            # 2. Clip coordinates to image boundaries (safe cropping)
            x1_crop = max(0, x1_exp)
            y1_crop = max(0, y1_exp)
            x2_crop = min(W_orig, x2_exp)
            y2_crop = min(H_orig, y2_orig)
            
            # --- CROP IMAGE AND ALPHA MASK ---
            cropped_nail_bgr = img_bgr[y1_crop:y2_crop, x1_crop:x2_crop]
            cropped_alpha_mask = soft_alpha_mask_orig[y1_crop:y2_crop, x1_crop:x2_crop]

            # --- CREATE RGBA IMAGE (Transparent Background) ---
            cutout_rgba = np.zeros((cropped_nail_bgr.shape[0], cropped_nail_bgr.shape[1], 4), dtype=np.uint8)
            cutout_rgba[:, :, 0:3] = cropped_nail_bgr
            cutout_rgba[:, :, 3] = cropped_alpha_mask 

            cutout_filename = os.path.join(OUTPUT_DIR, f"nail_cutout_final_{i+1}.png")
            cv2.imwrite(cutout_filename, cutout_rgba)
            print(f"✅ Saved cutout {i+1} to: {cutout_filename}")

    else:
        print("❌ No nails detected for cutting out.")


if __name__ == "__main__":
    cutout_nails()