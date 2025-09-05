from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import torch
import cv2
import numpy as np
import io
import os
from typing import List, Dict, Any
import uvicorn
import logging
from pathlib import Path

# Get the port from environment variable (Render requirement)
PORT = int(os.environ.get("PORT", 8000))

# Update model path to work on Render
BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / 'model' / 'best.pt'  # or your custom model name

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app setup
app = FastAPI(
    title='Pothole Detection API',
    description='Professional API for pothole detection using YOLOv8 computer vision model',
    version='1.0.0',
    docs_url='/docs',
    redoc_url='/redoc'
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def setup_yolo_compatibility():
    """Configure PyTorch compatibility for YOLOv8 model loading"""
    safe_classes = [
        'ultralytics.nn.tasks.DetectionModel',
        'ultralytics.nn.modules.head.Detect',
        'ultralytics.nn.modules.conv.Conv',
        'ultralytics.nn.modules.block.C2f',
        'ultralytics.nn.modules.block.SPPF',
        'ultralytics.nn.modules.block.Bottleneck'
    ]
    
    try:
        torch.serialization.add_safe_globals(safe_classes)
        logger.info("PyTorch safe globals configured successfully")
    except Exception as e:
        logger.warning(f"Failed to configure safe globals: {e}")
    
    # Fallback compatibility method
    global original_torch_load
    original_torch_load = torch.load
    
    def compatible_torch_load(*args, **kwargs):
        kwargs.setdefault('weights_only', False)
        return original_torch_load(*args, **kwargs)
    
    torch.load = compatible_torch_load

# Initialize PyTorch compatibility
setup_yolo_compatibility()

from ultralytics import YOLO

# Model configuration - Update this path to your custom model
# Change this to your custom pothole model path
SUPPORTED_FORMATS = ['.jpg', '.png', '.jpeg', '.bmp', '.tiff', '.webp']
model = None

def load_model():
    """Load custom YOLOv8 pothole detection model - NO fallback to standard model"""
    global model
    
    try:
        if os.path.exists(MODEL_PATH):
            model = YOLO(MODEL_PATH)
            logger.info(f"Custom pothole model loaded successfully from {MODEL_PATH}")
            return True
        else:
            logger.error(f"Custom pothole model NOT FOUND at {MODEL_PATH}")
            logger.error("Please ensure your trained pothole model is placed at the correct path")
            model = None
            return False
        
    except Exception as e:
        logger.error(f"Failed to load custom pothole model: {e}")
        model = None
        return False

# Load model on startup
model_loaded = load_model()

# Restore original torch.load
if 'original_torch_load' in globals():
    torch.load = original_torch_load

@app.get('/', response_model=Dict[str, Any])
async def root():
    """API root endpoint providing service status and information"""
    return {
        'service': 'Pothole Detection API',
        'status': 'operational' if model is not None else 'service unavailable - custom model not loaded',
        'model_loaded': model is not None,
        'version': '1.0.0',
        'model_type': 'Custom Pothole Detection Model Only',
        'endpoints': [
            '/docs - API documentation',
            '/health - Health check',
            '/detect - Pothole detection',
            '/detect_with_visualization - Detection with annotated image'
        ]
    }

@app.get('/health', response_model=Dict[str, Any])
async def health_check():
    """Service health check endpoint"""
    return {
        'status': 'healthy' if model is not None else 'unhealthy - custom model not loaded',
        'model_status': 'custom pothole model loaded' if model is not None else 'custom model not found',
        'pytorch_version': torch.__version__,
        'model_path': MODEL_PATH,
        'supported_formats': SUPPORTED_FORMATS,
        'service_ready': model is not None,
        'model_type': 'Custom Pothole Detection Only'
    }

@app.post('/detect', response_model=Dict[str, Any])
async def detect_potholes(file: UploadFile = File(...)):
    """
    Detect potholes in uploaded image and return detection results
    
    Args:
        file: Image file for pothole detection
        
    Returns:
        JSON response with detection results including bounding boxes and confidence scores
    """
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Service unavailable: Custom pothole model not loaded. Please ensure your trained model is at the correct path."
        )
    
    # Validate file format
    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file format. Accepted formats: {', '.join(SUPPORTED_FORMATS)}"
        )
    
    try:
        # Process uploaded image
        image_bytes = await file.read()
        image_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(
                status_code=400, 
                detail="Invalid image file or corrupted image data"
            )
        
        # Execute detection
        detection_results = model(image, verbose=False)
        
        # Process detection results
        detections = []
        total_detections = 0
        
        for result in detection_results:
            if result.boxes is not None:
                for box in result.boxes:
                    bbox_coords = box.xyxy[0].tolist()
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = result.names[class_id]
                    
                    detection_data = {
                        'detection_id': total_detections + 1,
                        'bounding_box': {
                            'x1': round(bbox_coords[0], 2),
                            'y1': round(bbox_coords[1], 2),
                            'x2': round(bbox_coords[2], 2),
                            'y2': round(bbox_coords[3], 2)
                        },
                        'confidence_score': round(confidence, 3),
                        'object_class': class_name,
                        'class_id': class_id
                    }
                    
                    detections.append(detection_data)
                    total_detections += 1
        
        response_data = {
            'processing_status': 'completed',
            'filename': file.filename,
            'image_dimensions': {
                'height': image.shape[0],
                'width': image.shape[1],
                'channels': image.shape[2]
            },
            'detection_summary': {
                'total_detections': total_detections,
                'detection_classes': list(set([d['object_class'] for d in detections]))
            },
            'detections': detections,
            'model_info': 'Custom Pothole Detection Model'
        }
        
        logger.info(f"Pothole detection completed for {file.filename}: {total_detections} objects detected")
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Detection processing failed: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Internal processing error: {str(e)}"
        )

@app.post('/detect_with_visualization')
async def detect_with_visualization(file: UploadFile = File(...)):
    """
    Detect potholes and return annotated image with bounding boxes
    
    Args:
        file: Image file for pothole detection
        
    Returns:
        Annotated image with detection bounding boxes and labels
    """
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Service unavailable: Custom pothole model not loaded. Please ensure your trained model is at the correct path."
        )
    
    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file format. Accepted formats: {', '.join(SUPPORTED_FORMATS)}"
        )
    
    try:
        # Process image
        image_bytes = await file.read()
        image_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(
                status_code=400, 
                detail="Invalid image file or corrupted image data"
            )
        
        # Execute detection
        detection_results = model(image, verbose=False)
        
        # Create annotated image
        annotated_image = image.copy()
        detection_count = 0
        
        for result in detection_results:
            if result.boxes is not None:
                for box in result.boxes:
                    detection_count += 1
                    
                    # Extract detection data
                    bbox_coords = list(map(int, box.xyxy[0].tolist()))
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = result.names[class_id]
                    
                    # Draw bounding box
                    cv2.rectangle(
                        annotated_image, 
                        (bbox_coords[0], bbox_coords[1]), 
                        (bbox_coords[2], bbox_coords[3]), 
                        (0, 255, 0), 
                        2
                    )
                    
                    # Add detection label
                    label_text = f'{class_name}: {confidence:.2f}'
                    label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    
                    # Draw label background
                    cv2.rectangle(
                        annotated_image, 
                        (bbox_coords[0], bbox_coords[1] - label_size[1] - 10), 
                        (bbox_coords[0] + label_size[0], bbox_coords[1]), 
                        (0, 255, 0), 
                        -1
                    )
                    
                    # Draw label text
                    cv2.putText(
                        annotated_image, 
                        label_text, 
                        (bbox_coords[0], bbox_coords[1] - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, 
                        (0, 0, 0), 
                        2
                    )
        
        # Encode annotated image
        encode_success, image_buffer = cv2.imencode('.jpg', annotated_image)
        if not encode_success:
            raise HTTPException(status_code=500, detail="Failed to encode annotated image")
        
        image_stream = io.BytesIO(image_buffer)
        
        logger.info(f"Pothole visualization completed for {file.filename}: {detection_count} objects annotated")
        
        return StreamingResponse(
            image_stream,
            media_type="image/jpeg",
            headers={
                "X-Detection-Count": str(detection_count),
                "X-Processing-Status": "completed",
                "X-Model-Type": "Custom-Pothole-Detection",
                "Content-Disposition": f"inline; filename=annotated_{file.filename}"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Visualization processing failed: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Internal processing error: {str(e)}"
        )

@app.get('/model_info', response_model=Dict[str, Any])
async def get_model_info():
    """Get information about the loaded model"""
    if model is None:
        return {
            'model_status': 'not loaded',
            'error': 'Custom pothole model not available',
            'solution': 'Please ensure your trained pothole model is placed at the correct MODEL_PATH',
            'current_model_path': MODEL_PATH
        }
    
    return {
        'model_status': 'loaded',
        'model_type': 'Custom YOLOv8 Pothole Detection',
        'model_path': MODEL_PATH,
        'model_classes': list(model.names.values()) if hasattr(model, 'names') else [],
        'total_classes': len(model.names) if hasattr(model, 'names') else 0,
        'input_requirements': {
            'supported_formats': SUPPORTED_FORMATS,
            'max_file_size': '10MB recommended'
        },
        'note': 'This API uses ONLY your custom pothole detection model - no fallback to standard models'
    }

if __name__ == "__main__":
    if model_loaded:
        logger.info("Pothole Detection API starting - Custom pothole model ready for inference")
        logger.info(f"Model classes available: {list(model.names.values()) if model and hasattr(model, 'names') else 'Unknown'}")
    else:
        logger.error("Pothole Detection API starting - CUSTOM MODEL NOT LOADED!")
        logger.error(f"Please ensure your trained pothole model is located at: {MODEL_PATH}")
        logger.error("Detection endpoints will be unavailable until the custom model is loaded")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=PORT,  # Use PORT from environment
        log_level="info",
        workers=1  # Single worker for Render's starter plan
    )