from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import os
from pathlib import Path
import tempfile
import shutil
import asyncio

# Initialize FastAPI app FIRST
app = FastAPI(
    title="Pothole Detection API",
    description="YOLOv8 Pothole Detection Service",
    version="1.0.0"
)

# Global model variable
model = None

async def load_model_async():
    """Load model asynchronously to avoid startup timeout"""
    global model
    try:
        # Try to import ultralytics
        from ultralytics import YOLO
        
        # Check if custom model exists
        model_path = "../model/yolov8n.pt"
        if os.path.exists(model_path):
            print(f"Loading custom model from {model_path}")
            model = YOLO(model_path)
        else:
            print("Custom model not found, using default YOLOv8n")
            model = YOLO('yolov8n.pt')  # This will download if needed
        
        print("Model loaded successfully!")
        return True
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("API will run without model (model loading on first request)")
        return False

def load_model_sync():
    """Synchronous model loading for first request"""
    global model
    if model is not None:
        return model
    
    try:
        from ultralytics import YOLO
        model_path = "../model/yolov8n.pt"
        
        if os.path.exists(model_path):
            model = YOLO(model_path)
        else:
            model = YOLO('yolov8n.pt')
        
        print("Model loaded on first request!")
        return model
    except Exception as e:
        print(f"Failed to load model: {e}")
        raise HTTPException(status_code=500, detail="Model loading failed")

@app.on_event("startup")
async def startup_event():
    """Non-blocking startup - don't wait for model loading"""
    print("API starting up...")
    # Start model loading in background but don't wait for it
    asyncio.create_task(load_model_async())

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
        "model_status": "loaded" if model is not None else "loading/not loaded",
        "api_ready": True
    }

@app.post("/detect")
async def detect_potholes(file: UploadFile = File(...)):
    """Detect potholes in uploaded image"""
    
    # Load model if not already loaded (lazy loading)
    current_model = model if model is not None else load_model_sync()
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            tmp_path = tmp_file.name
        
        # Run inference
        results = current_model(tmp_path)
        
        # Process results
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = current_model.names[class_id]
                    
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
        if 'tmp_path' in locals():
            try:
                os.unlink(tmp_path)
            except:
                pass
        
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@app.post("/detect-batch")
async def detect_batch(files: list[UploadFile] = File(...)):
    """Detect potholes in multiple images"""
    
    # Load model if not already loaded
    current_model = model if model is not None else load_model_sync()
    
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
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                shutil.copyfileobj(file.file, tmp_file)
                tmp_path = tmp_file.name
            
            detections = []
            model_results = current_model(tmp_path)
            
            for result in model_results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        class_name = current_model.names[class_id]
                        
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

# For Render deployment
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)