# noctura-vision/backend/app/api/utils.py
import torch
import torchvision
from torchvision.transforms import functional as F
import numpy as np
import cv2
from PIL import Image
import os # For environment variable check

# Hugging Face imports
from transformers import Mask2FormerForUniversalSegmentation, AutoImageProcessor

# Constants for COCO classes (for Torchvision Mask R-CNN)
# Note: Mask2Former uses its own id2label mapping, this is only for Torchvision
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# --- General Image Utilities ---

def pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
    """Converts a PIL Image to an OpenCV BGR numpy array."""
    return cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)

def cv2_to_pil(cv2_image_bgr: np.ndarray) -> Image.Image:
    """Converts an OpenCV BGR numpy array to a PIL Image."""
    return Image.fromarray(cv2.cvtColor(cv2_image_bgr, cv2.COLOR_BGR2RGB))

def resize_image_to_max_dim(image_np_rgb: np.ndarray, max_dim: int) -> np.ndarray:
    """
    Resizes an RGB image (NumPy array) if its largest dimension exceeds max_dim,
    maintaining aspect ratio.
    """
    h, w = image_np_rgb.shape[:2]
    if max_dim > 0 and max(h, w) > max_dim:
        scale_factor = max_dim / max(h, w)
        new_w, new_h = int(w * scale_factor), int(h * scale_factor)
        return cv2.resize(image_np_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image_np_rgb

# --- Torchvision (Mask R-CNN) Utilities ---

def load_torchvision_model(model_name: str, device: torch.device):
    """
    Loads a pre-trained Torchvision Mask R-CNN model.
    Checks for TORCH_HOME environment variable for cache location.
    """
    if "TORCH_HOME" not in os.environ:
        # Default path relative to the backend/ directory
        # This assumes the script is run from 'backend/' or the env is set globally
        # We need to set it for the process explicitly if it's not already
        print("TORCH_HOME environment variable not set. Setting to './models_cache/torchvision' relative to script location.")
        os.environ["TORCH_HOME"] = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models_cache/torchvision'))
    
    print(f"Loading Torchvision model: {model_name} from TORCH_HOME={os.environ['TORCH_HOME']}...")
    if model_name == 'maskrcnn_resnet50_fpn_v2':
        model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(
            weights=torchvision.models.detection.MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        )
    elif model_name == 'maskrcnn_resnet50_fpn':
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(
            weights=torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        )
    else:
        raise ValueError(f"Unknown Torchvision model name: {model_name}. Supported: 'maskrcnn_resnet50_fpn', 'maskrcnn_resnet50_fpn_v2'")
    
    model.to(device)
    model.eval() # Set to evaluation mode
    print("Torchvision model loaded successfully and set to evaluation mode.")
    return model

def preprocess_image_for_torchvision(image_np_rgb: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Converts a NumPy RGB image to a PyTorch tensor suitable for Torchvision detection models.
    Adds batch dimension and sends to device.
    """
    # Torchvision models expect a list of [C, H, W] tensors, normalized
    img_tensor = F.to_tensor(image_np_rgb) # Converts to [C, H, W] and scales to [0, 1]
    return img_tensor.unsqueeze(0).to(device) # Add batch dimension [1, C, H, W] and send to device

def get_torchvision_prediction(model: torch.nn.Module, img_tensor: torch.Tensor, threshold: float = 0.5):
    """
    Performs inference using the Torchvision model and filters results by score.
    img_tensor: A single image tensor [1, C, H, W] already on the correct device.
    """
    with torch.no_grad():
        prediction = model(img_tensor) # Pass the image tensor to the model

    # Filter out predictions below the threshold
    # prediction[0] because we are processing a single image in the batch
    pred_scores = prediction[0]['scores']
    keep_indices = pred_scores >= threshold
    
    pred_boxes = prediction[0]['boxes'][keep_indices]
    pred_masks = prediction[0]['masks'][keep_indices]
    pred_labels = prediction[0]['labels'][keep_indices]
    pred_scores_filtered = pred_scores[keep_indices]

    return pred_boxes, pred_masks, pred_labels, pred_scores_filtered

def draw_torchvision_instance_segmentation(
    image_np_bgr: np.ndarray,
    boxes: torch.Tensor,
    masks: torch.Tensor,
    labels: torch.Tensor,
    scores: torch.Tensor,
    class_names: list,
    score_thr: float = 0.5,
    draw_mask_contours: bool = True,
    contour_thickness: int = 2
) -> np.ndarray:
    """
    Draws bounding boxes, masks, labels, and optionally mask contours on the image for Torchvision output.
    image_np_bgr: Original numpy BGR image to draw on.
    boxes: [N, 4] tensor of bounding boxes (xmin, ymin, xmax, ymax) on device.
    masks: [N, 1, H, W] tensor of masks on device.
    labels: [N] tensor of class labels on device.
    scores: [N] tensor of scores on device.
    class_names: List of class names, where index corresponds to label_id.
    score_thr: Minimum score threshold to display a detection.
    draw_mask_contours: If True, draws bold contours around the masks.
    contour_thickness: Thickness of the mask contours.
    """
    img_to_draw = image_np_bgr.copy()
    num_instances = boxes.shape[0]

    for i in range(num_instances):
        current_score = scores[i].cpu().item()
        if current_score < score_thr:
            continue

        box = boxes[i].cpu().numpy().astype(int)
        label_id = labels[i].cpu().item()

        if label_id < len(class_names):
            label_name = class_names[label_id]
        else:
            label_name = "N/A"

        np.random.seed(label_id + 42) # Seed for consistent random colors per class
        color = np.random.randint(0, 255, size=3).tolist()

        cv2.rectangle(img_to_draw, (box[0], box[1]), (box[2], box[3]), color, 2)

        text = f"{label_name}: {current_score:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        text_bg_y1 = box[1] - text_height - baseline - 5
        text_bg_y2 = box[1] - baseline
        text_bg_y1 = max(text_bg_y1, 0)
        text_y_pos = max(box[1] - 5 - baseline, text_height + 5)

        cv2.rectangle(img_to_draw, (box[0], text_bg_y1), (box[0] + text_width, text_bg_y2), color, -1)
        text_color = (0,0,0) if sum(color) > 3 * 255 / 2 else (255,255,255)
        cv2.putText(img_to_draw, text, (box[0], text_y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

        mask_np = masks[i, 0].cpu().numpy()
        binary_mask = (mask_np > 0.5).astype(np.uint8)

        colored_mask_overlay = np.zeros_like(img_to_draw, dtype=np.uint8)
        colored_mask_overlay[binary_mask == 1] = color
        img_to_draw = cv2.addWeighted(img_to_draw, 1.0, colored_mask_overlay, 0.4, 0)

        if draw_mask_contours:
            try:
                contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img_to_draw, contours, -1, color, contour_thickness)
            except Exception as e:
                print(f"Warning: Could not draw mask contours for instance {i}. Error: {e}")

    return img_to_draw

# --- Hugging Face Mask2Former Utilities ---

def load_mask2former_model_and_processor(model_name: str, device: torch.device):
    """
    Loads a pre-trained Hugging Face Mask2Former model and its corresponding image processor.
    Returns the processor, the model, and its id2label mapping.
    Checks for HF_HOME environment variable for cache location.
    """
    if "HF_HOME" not in os.environ:
        # Default path relative to the backend/ directory
        print("HF_HOME environment variable not set. Setting to './models_cache/huggingface' relative to script location.")
        os.environ["HF_HOME"] = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models_cache/huggingface'))
    
    print(f"Loading image processor for {model_name} from HF_HOME={os.environ['HF_HOME']}...")
    processor = AutoImageProcessor.from_pretrained(model_name)
    
    print(f"Loading model {model_name}...")
    model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name)
    model.to(device)
    model.eval() # Set to evaluation mode

    id2label_mapping = model.config.id2label
    print(f"Hugging Face model's ID to Label mapping loaded. Example: {list(id2label_mapping.items())[:5]}...")

    print("Hugging Face model loaded successfully and set to evaluation mode.")
    return processor, model, id2label_mapping

def preprocess_image_for_mask2former(image_pil: Image.Image, processor: AutoImageProcessor, device: torch.device):
    """
    Preprocesses a PIL Image into the tensor format expected by Mask2Former.
    """
    inputs = processor(images=image_pil, return_tensors="pt").to(device)
    return inputs

def postprocess_mask2former_outputs(outputs, processor: AutoImageProcessor, original_image_size: tuple, score_threshold: float = 0.5):
    """
    Post-processes the raw outputs from Mask2Former to get instance segmentation masks, labels, and scores.
    original_image_size: Tuple (height, width) of the original image.
    """
    processed_results = processor.post_process_instance_segmentation(outputs, target_sizes=[original_image_size])[0]

    segmentation_map = processed_results['segmentation'] 
    segments_info = processed_results['segments_info']   

    all_masks = []
    all_labels = []
    all_scores = []

    for segment in segments_info:
        if segment['score'] >= score_threshold:
            segment_id = segment['id']
            label_id = segment['label_id']
            score = segment['score']

            instance_mask = (segmentation_map == segment_id)
            
            all_masks.append(instance_mask)
            all_labels.append(label_id)
            all_scores.append(score)

    if not all_masks:
        return torch.empty((0, *original_image_size), dtype=torch.bool), \
               torch.empty((0,), dtype=torch.long), \
               torch.empty((0,), dtype=torch.float)

    filtered_masks = torch.stack(all_masks)
    filtered_labels = torch.tensor(all_labels, dtype=torch.long)
    filtered_scores = torch.tensor(all_scores, dtype=torch.float)

    return filtered_masks, filtered_labels, filtered_scores

def draw_mask2former_instance_segmentation(
    image_np_bgr: np.ndarray,
    predicted_masks: torch.Tensor,
    predicted_labels: torch.Tensor,
    predicted_scores: torch.Tensor,
    id2label_mapping: dict,
    score_thr: float = 0.5,
    draw_mask_contours: bool = True,
    contour_thickness: int = 2
) -> np.ndarray:
    """
    Draws instance segmentation results from Mask2Former (Hugging Face) on the image.
    Uses the provided id2label_mapping for correct class names.
    Adds consistent coloring per class and optional mask contours.
    """
    img_to_draw = image_np_bgr.copy()
    num_instances = predicted_masks.shape[0]

    for i in range(num_instances):
        current_score = predicted_scores[i].cpu().item()
        if current_score < score_thr:
            continue

        mask_tensor = predicted_masks[i]
        mask = mask_tensor.cpu().numpy().astype(np.uint8)
        label_id = predicted_labels[i].cpu().item()

        label_name = id2label_mapping.get(label_id, f"Unknown ID {label_id}")

        np.random.seed(label_id + 30) # Different seed constant from maskrcnn for variety
        color = np.random.randint(0, 255, size=3).tolist()

        y_coords, x_coords = np.where(mask)
        if len(y_coords) == 0 or len(x_coords) == 0: continue

        xmin, xmax = np.min(x_coords), np.max(x_coords)
        ymin, ymax = np.min(y_coords), np.max(y_coords)

        cv2.rectangle(img_to_draw, (xmin, ymin), (xmax, ymax), color, 2) # Change thickness from 1 to 2 for better visibility

        text = f"{label_name}: {current_score:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        text_bg_y1 = ymin - text_height - baseline - 5
        text_bg_y2 = ymin - baseline
        text_bg_y1 = max(text_bg_y1, 0)
        text_y_pos = max(ymin - 5 - baseline, text_height + 5)
        
        cv2.rectangle(img_to_draw, (xmin, text_bg_y1), (xmin + text_width, text_bg_y2), color, -1)
        text_color_val = (0,0,0) if sum(color) > 3 * 255 / 2 else (255,255,255)
        cv2.putText(img_to_draw, text, (xmin, text_y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color_val, 1)

        colored_mask_overlay = np.zeros_like(img_to_draw, dtype=np.uint8)
        colored_mask_overlay[mask == 1] = color
        img_to_draw = cv2.addWeighted(img_to_draw, 1.0, colored_mask_overlay, 0.4, 0)

        if draw_mask_contours:
            try:
                contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img_to_draw, contours, -1, color, contour_thickness)
            except Exception as e:
                print(f"Warning: Could not draw mask contours for instance {i}. Error: {e}")

    return img_to_draw
