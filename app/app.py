from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import os
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import shutil

# Initialize FastAPI app
app = FastAPI(
    title="Pothole Detection API",
    description="YOLOv8 Pothole Detection Service",
    version="1.0.0"
)

# Load YOLO model
MODEL_PATH = "model/yolov8n.pt"
model = None

def load_model():
    """Load the YOLO model"""
    global model
    try:
        if os.path.exists(MODEL_PATH):
            model = YOLO(MODEL_PATH)
            print(f"Model loaded successfully from {MODEL_PATH}")
        else:
            print(f"Model file not found at {MODEL_PATH}")
            # Fallback to default YOLOv8n model
            model = YOLO('yolov8n.pt')
            print("Using default YOLOv8n model")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None

# Load model on startup
load_model()

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    load_model()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Pothole Detection API is running!",
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_status": "loaded" if model is not None else "not loaded",
        "model_path": MODEL_PATH
    }

@app.post("/detect")
async def detect_potholes(file: UploadFile = File(...)):
    """
    Detect potholes in uploaded image
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            # Save uploaded file
            shutil.copyfileobj(file.file, tmp_file)
            tmp_path = tmp_file.name
        
        # Run inference
        results = model(tmp_path)
        
        # Process results
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Extract box coordinates and confidence
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    
                    detections.append({
                        "class": class_name,
                        "confidence": round(confidence, 3),
                        "bbox": {
                            "x1": round(x1, 2),
                            "y1": round(y1, 2),
                            "x2": round(x2, 2),
                            "y2": round(y2, 2)
                        }
                    })
        
        # Clean up temporary file
        os.unlink(tmp_path)
        
        return {
            "filename": file.filename,
            "detections_count": len(detections),
            "detections": detections,
            "status": "success"
        }
    
    except Exception as e:
        # Clean up temporary file if it exists
        if 'tmp_path' in locals():
            try:
                os.unlink(tmp_path)
            except:
                pass
        
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@app.post("/detect-batch")
async def detect_batch(files: list[UploadFile] = File(...)):
    """
    Detect potholes in multiple images
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    results = []
    
    for file in files:
        if not file.content_type.startswith('image/'):
            results.append({
                "filename": file.filename,
                "error": "File must be an image",
                "status": "failed"
            })
            continue
        
        try:
            # Process each file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                shutil.copyfileobj(file.file, tmp_file)
                tmp_path = tmp_file.name
            
            # Run inference
            detections = []
            model_results = model(tmp_path)
            
            for result in model_results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        class_name = model.names[class_id]
                        
                        detections.append({
                            "class": class_name,
                            "confidence": round(confidence, 3),
                            "bbox": {
                                "x1": round(x1, 2),
                                "y1": round(y1, 2),
                                "x2": round(x2, 2),
                                "y2": round(y2, 2)
                            }
                        })
            
            results.append({
                "filename": file.filename,
                "detections_count": len(detections),
                "detections": detections,
                "status": "success"
            })
            
            # Clean up
            os.unlink(tmp_path)
            
        except Exception as e:
            if 'tmp_path' in locals():
                try:
                    os.unlink(tmp_path)
                except:
                    pass
            
            results.append({
                "filename": file.filename,
                "error": str(e),
                "status": "failed"
            })
    
    return {
        "total_files": len(files),
        "results": results
    }

# For Render deployment - important!
if __name__ == "__main__":
    # Get port from environment variable (Render provides this)
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)