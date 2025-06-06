# noctura-vision/backend/app/api/routes/segmentation.py
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, Form
from fastapi.responses import Response
from typing import Dict, Any, Literal
import io
import torch
import cv2
import numpy as np
from PIL import Image

from app.api.dependencies import get_models
from app.api.utils import (
    resize_image_to_max_dim,
    preprocess_image_for_torchvision, get_torchvision_prediction, draw_torchvision_instance_segmentation, COCO_INSTANCE_CATEGORY_NAMES,
    preprocess_image_for_mask2former, postprocess_mask2former_outputs, draw_mask2former_instance_segmentation
)

router = APIRouter()

@router.post("/process_image")
async def process_image(
    file: UploadFile = File(...),
    model_type: Literal['maskrcnn', 'mask2former'] = Form(...),
    apply_enhancement: bool = Form(True),
    score_threshold: float = Form(0.5),
    input_max_dim: int = Form(1080),
    models: Dict[str, Any] = Depends(get_models),
):
    """
    Processes an image by optionally applying low-light enhancement and then
    performing instance segmentation with the selected model.
    """
    # --- 1. Read and Prepare Image ---
    try:
        contents = await file.read()
        image_pil = Image.open(io.BytesIO(contents)).convert("RGB")
        image_np_rgb = np.array(image_pil)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

    # --- 2. Apply Low-Light Enhancement (Optional) ---
    if apply_enhancement:
        if models["zero_dce_model"] is None:
            raise HTTPException(status_code=503, detail="Zero-DCE model not loaded.")
        
        image_for_inference = resize_image_to_max_dim(image_np_rgb, input_max_dim)
        img_tensor_e = torch.from_numpy(image_for_inference).float().div(255.).permute(2,0,1).unsqueeze(0).to(models["device"])
        with torch.no_grad():
            _, enhanced_tensor_e, _ = models["zero_dce_model"](img_tensor_e)
        
        enhanced_tensor_e = torch.clamp(enhanced_tensor_e, 0.0, 1.0)
        image_to_segment_rgb = (enhanced_tensor_e.squeeze(0).permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
    else:
        image_to_segment_rgb = resize_image_to_max_dim(image_np_rgb, input_max_dim)

    # --- 3. Perform Segmentation ---
    if model_type == 'maskrcnn':
        if models["maskrcnn_model"] is None:
            raise HTTPException(status_code=503, detail="Mask R-CNN model not loaded.")
        
        img_tensor = preprocess_image_for_torchvision(image_to_segment_rgb, models["device"])
        boxes, masks, labels, scores = get_torchvision_prediction(models["maskrcnn_model"], img_tensor, score_threshold)
        
        # Convert RGB processing image to BGR for drawing
        image_bgr_for_drawing = cv2.cvtColor(image_to_segment_rgb, cv2.COLOR_RGB2BGR)
        output_image_bgr = draw_torchvision_instance_segmentation(image_bgr_for_drawing, boxes, masks, labels, scores, COCO_INSTANCE_CATEGORY_NAMES, score_threshold)

    elif model_type == 'mask2former':
        if models["mask2former_model"] is None:
            raise HTTPException(status_code=503, detail="Mask2Former model not loaded.")
            
        img_pil = Image.fromarray(image_to_segment_rgb)
        inputs = preprocess_image_for_mask2former(img_pil, models["mask2former_processor"], models["device"])
        with torch.no_grad():
            outputs = models["mask2former_model"](**inputs)
        
        masks, labels, scores = postprocess_mask2former_outputs(outputs, models["mask2former_processor"], img_pil.size[::-1], score_threshold)
        
        # Convert RGB processing image to BGR for drawing
        image_bgr_for_drawing = cv2.cvtColor(image_to_segment_rgb, cv2.COLOR_RGB2BGR)
        output_image_bgr = draw_mask2former_instance_segmentation(image_bgr_for_drawing, masks, labels, scores, models["mask2former_id2label"], score_threshold)
        
    else:
        raise HTTPException(status_code=400, detail="Invalid model type specified.")

    # --- 4. Return Processed Image ---
    is_success, buffer = cv2.imencode(".jpg", output_image_bgr)
    if not is_success:
        raise HTTPException(status_code=500, detail="Could not encode output image.")
        
    return Response(content=buffer.tobytes(), media_type="image/jpeg")

@router.get("/models")
async def get_available_models():
    """Returns a list of available segmentation models."""
    return {
        "models": [
            {"value": "maskrcnn", "label": "Mask R-CNN (Torchvision)"},
            {"value": "mask2former", "label": "Mask2Former (Hugging Face)"}
        ]
    }
