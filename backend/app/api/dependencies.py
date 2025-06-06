# noctura-vision/backend/app/api/dependencies.py
import torch
import os

from app.api.models.zero_dce_model import DCENet
from app.api.utils import load_torchvision_model, load_mask2former_model_and_processor

# Global variables to hold models and processors
# These will be initialized once when the application starts
models = {
    "device": None,
    "zero_dce_model": None,
    "maskrcnn_model": None,
    "mask2former_processor": None,
    "mask2former_model": None,
    "mask2former_id2label": None,
}

async def load_all_models():
    """
    Loads all AI models into memory. This function should be called once at application startup.
    """
    print("Initializing models...")
    
    # Determine device
    models["device"] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {models['device']}")

    # --- Load Zero-DCE Model (Model E) ---
    zero_dce_weights_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models_cache/zero_dce/model200_dark_faces.pth'))
    
    if not os.path.exists(zero_dce_weights_path):
        print(f"Warning: Zero-DCE weights not found at {zero_dce_weights_path}. Zero-DCE model will not be loaded.")
        models["zero_dce_model"] = None
    else:
        try:
            models["zero_dce_model"] = DCENet().to(models["device"])
            models["zero_dce_model"].load_state_dict(torch.load(zero_dce_weights_path, map_location=models["device"]))
            models["zero_dce_model"].eval()
            print("Zero-DCE model loaded successfully.")
        except Exception as e:
            print(f"Error loading Zero-DCE model from {zero_dce_weights_path}: {e}")
            models["zero_dce_model"] = None

    # --- Load Mask R-CNN Model (Torchvision) ---
    # Model name 'maskrcnn_resnet50_fpn_v2' will trigger automatic download if not in TORCH_HOME
    try:
        models["maskrcnn_model"] = load_torchvision_model('maskrcnn_resnet50_fpn_v2', models["device"])
        print("Mask R-CNN model loaded successfully.")
    except Exception as e:
        print(f"Error loading Mask R-CNN model: {e}")
        models["maskrcnn_model"] = None

    # --- Load Mask2Former Model (Hugging Face) ---
    # Model name 'facebook/mask2former-swin-large-coco-instance' will trigger automatic download if not in HF_HOME
    try:
        processor, model, id2label = load_mask2former_model_and_processor('facebook/mask2former-swin-large-coco-instance', models["device"])
        models["mask2former_processor"] = processor
        models["mask2former_model"] = model
        models["mask2former_id2label"] = id2label
        print("Mask2Former model loaded successfully.")
    except Exception as e:
        print(f"Error loading Mask2Former model: {e}")
        models["mask2former_processor"] = None
        models["mask2former_model"] = None
        models["mask2former_id2label"] = None
    
    print("All models initialization attempted.")

async def get_models():
    """Dependency function to provide access to loaded models."""
    return models

# This part is just for testing the loading if run as a script.
# In a real FastAPI app, load_all_models is called during startup.
if __name__ == "__main__":
    import asyncio
    asyncio.run(load_all_models())
    print("\nModels loaded for script test:")
    print(f"Zero-DCE loaded: {models['zero_dce_model'] is not None}")
    print(f"Mask R-CNN loaded: {models['maskrcnn_model'] is not None}")
    print(f"Mask2Former loaded: {models['mask2former_model'] is not None}")
