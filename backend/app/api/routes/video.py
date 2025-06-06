# noctura-vision/backend/app/api/routes/video.py
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from typing import Dict, Any
import base64
import io
import torch
import time
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

@router.websocket("/ws/process_video")
async def websocket_process_video(
    websocket: WebSocket,
    models: Dict[str, Any] = Depends(get_models)
):
    await websocket.accept()
    prev_frame_time = 0
    try:
        while True:
            # Receive configuration and image data from client
            data = await websocket.receive_json()
            
            # --- Extract parameters from the received JSON data ---
            show_fps = data.get("show_fps", False)
            model_type = data.get("model_type", "mask2former")
            apply_enhancement = data.get("apply_enhancement", True)
            score_threshold = data.get("score_threshold", 0.5)
            input_max_dim = data.get("input_max_dim", 720) # Use a smaller default for real-time video
            image_b64 = data["image_b64"]
            
            # --- Decode and prepare image ---
            img_bytes = base64.b64decode(image_b64.split(',')[1]) # Handle 'data:image/jpeg;base64,' prefix
            image_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            image_np_rgb = np.array(image_pil)
            image_np_rgb = resize_image_to_max_dim(image_np_rgb, input_max_dim)
            
            # --- Apply pipeline (similar to the HTTP endpoint) ---
            image_to_segment_rgb = image_np_rgb # Default to original
            if apply_enhancement:
                if models["zero_dce_model"]:
                    img_tensor_e = torch.from_numpy(image_np_rgb).float().div(255.).permute(2,0,1).unsqueeze(0).to(models["device"])
                    with torch.no_grad():
                        _, enhanced_tensor_e, _ = models["zero_dce_model"](img_tensor_e)
                    enhanced_tensor_e = torch.clamp(enhanced_tensor_e, 0.0, 1.0)
                    image_to_segment_rgb = (enhanced_tensor_e.squeeze(0).permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)

            # --- Perform Segmentation ---
            output_image_bgr = cv2.cvtColor(image_to_segment_rgb, cv2.COLOR_RGB2BGR) # Default to enhanced/original
            if model_type == 'maskrcnn' and models["maskrcnn_model"]:
                img_tensor = preprocess_image_for_torchvision(image_to_segment_rgb, models["device"])
                boxes, masks, labels, scores = get_torchvision_prediction(models["maskrcnn_model"], img_tensor, score_threshold)
                output_image_bgr = draw_torchvision_instance_segmentation(output_image_bgr, boxes, masks, labels, scores, COCO_INSTANCE_CATEGORY_NAMES, score_threshold)

            elif model_type == 'mask2former' and models["mask2former_model"]:
                img_pil_seg = Image.fromarray(image_to_segment_rgb)
                inputs = preprocess_image_for_mask2former(img_pil_seg, models["mask2former_processor"], models["device"])
                with torch.no_grad():
                    outputs = models["mask2former_model"](**inputs)
                masks, labels, scores = postprocess_mask2former_outputs(outputs, models["mask2former_processor"], img_pil_seg.size[::-1], score_threshold)
                output_image_bgr = draw_mask2former_instance_segmentation(output_image_bgr, masks, labels, scores, models["mask2former_id2label"], score_threshold)

            if show_fps:
                new_frame_time = time.time()
                if prev_frame_time > 0:
                    fps = 1 / (new_frame_time - prev_frame_time)
                    cv2.putText(output_image_bgr, f"FPS: {fps:.2f}", 
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                prev_frame_time = new_frame_time

            # --- Encode and send back the processed frame ---
            _, buffer = cv2.imencode(".jpg", output_image_bgr)
            processed_b64 = base64.b64encode(buffer).decode('utf-8')
            await websocket.send_text(f"data:image/jpeg;base64,{processed_b64}")

    except WebSocketDisconnect:
        print("Client disconnected.")
    except Exception as e:
        print(f"An error occurred in WebSocket: {e}")
        await websocket.close(code=1011, reason=f"Server error: {e}")
