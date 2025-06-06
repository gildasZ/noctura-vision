# noctura-vision/backend/app/api/routes/enhancement.py
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException
from fastapi.responses import Response
from typing import Dict, Any
import io
import torch
import cv2
import numpy as np
from PIL import Image

from app.api.dependencies import get_models
from app.api.models.zero_dce_model import DCENet
from app.api.utils import resize_image_to_max_dim # Import for image resizing if needed in API

router = APIRouter()

@router.post("/enhance_image")
async def enhance_image(
    file: UploadFile = File(...),
    models: Dict[str, Any] = Depends(get_models),
    input_max_dim: int = 1080 # Optional: resize input for inference, 0 for no resize
):
    """
    Performs low-light image enhancement using Zero-DCE (Model E).
    Accepts an image file and returns the enhanced image.
    """
    if models["zero_dce_model"] is None:
        raise HTTPException(status_code=503, detail="Zero-DCE model not loaded.")

    try:
        # Read image file
        contents = await file.read()
        image_pil = Image.open(io.BytesIO(contents)).convert("RGB")
        image_np_rgb = np.array(image_pil)
        
        # Resize input image for inference if specified
        image_np_rgb_for_inference = resize_image_to_max_dim(image_np_rgb, input_max_dim)

        # Prepare image for Zero-DCE: normalize, to tensor, add batch dim, to device
        img_tensor_e = torch.from_numpy(image_np_rgb_for_inference).float().div(255.).permute(2,0,1).unsqueeze(0).to(models["device"])

        # Perform enhancement
        with torch.no_grad():
            _, enhanced_tensor_e, _ = models["zero_dce_model"](img_tensor_e)
        
        # Convert enhanced tensor back to NumPy BGR for output
        enhanced_tensor_e = torch.clamp(enhanced_tensor_e, 0.0, 1.0)
        enhanced_np_rgb = (enhanced_tensor_e.squeeze(0).permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
        enhanced_np_bgr = cv2.cvtColor(enhanced_np_rgb, cv2.COLOR_RGB2BGR)

        # Encode the processed image back to JPEG (or PNG, etc.)
        is_success, buffer = cv2.imencode(".jpg", enhanced_np_bgr)
        if not is_success:
            raise HTTPException(status_code=500, detail="Could not encode image.")

        return Response(content=buffer.tobytes(), media_type="image/jpeg")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image enhancement failed: {e}")
