"""
Standalone script to test YOLOv8 loading
Run this first to isolate the issue
"""

import os
import sys
import torch

def test_yolo_loading():
    print("=== YOLO LOADING TEST ===")
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Working directory: {os.getcwd()}")
    
    # Test 1: Check ultralytics installation
    print("\n1. Testing ultralytics import...")
    try:
        import ultralytics
        print(f" Ultralytics version: {ultralytics.__version__}")
    except ImportError as e:
        print(f" Ultralytics import failed: {e}")
        print("Fix: pip install ultralytics")
        return False
    
    # Test 2: Setup PyTorch compatibility
    print("\n2. Setting up PyTorch compatibility...")
    try:
        # Method 1: Add safe globals
        torch.serialization.add_safe_globals([
            'ultralytics.nn.tasks.DetectionModel',
            'ultralytics.nn.modules.head.Detect',
            'ultralytics.nn.modules.conv.Conv',
            'ultralytics.nn.modules.block.C2f',
            'ultralytics.nn.modules.block.SPPF'
        ])
        print(" Safe globals added")
    except Exception as e:
        print(f"  Safe globals failed: {e}")
    
    # Method 2: Patch torch.load
    original_load = torch.load
    def patched_load(*args, **kwargs):
        kwargs.setdefault('weights_only', False)
        return original_load(*args, **kwargs)
    torch.load = patched_load
    print(" Patched torch.load")
    
    # Test 3: Try loading models
    from ultralytics import YOLO
    
    models_to_test = [
        ("/model/yolov8n.pt", "Custom model path"),
        ("yolov8n.pt", "Default yolov8n"),
        ("yolov8s.pt", "yolov8s variant"),
    ]
    
    for model_path, description in models_to_test:
        print(f"\n3. Testing {description} ({model_path})...")
        
        if model_path.startswith('/model/'):
            if not os.path.exists(model_path):
                print(f" File not found: {model_path}")
                continue
            else:
                print(f" File exists: {os.path.getsize(model_path)} bytes")
        
        try:
            model = YOLO(model_path)
            print(f"SUCCESS: {description} loaded")
            print(f"   Model type: {type(model)}")
            
            # Test a simple prediction
            print("   Testing prediction on dummy data...")
            import numpy as np
            dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            results = model(dummy_img, verbose=False)
            print(f"Prediction successful: {len(results)} results")
            
            # Restore torch.load and return success
            torch.load = original_load
            return True
            
        except Exception as e:
            print(f" FAILED: {description}")
            print(f"   Error: {e}")
            print(f"   Error type: {type(e).__name__}")
    
    # Restore torch.load
    torch.load = original_load
    print("\nAll model loading attempts failed")
    return False

def suggest_fixes():
    print("\n=== SUGGESTED FIXES ===")
    
    fixes = [
        "1. Update ultralytics: pip install --upgrade ultralytics",
        "2. Reinstall ultralytics: pip uninstall ultralytics && pip install ultralytics",
        "3. Check model file: ls -la /model/yolov8n.pt",
        "4. Download fresh weights: wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
        "5. Use conda environment: conda create -n yolo python=3.9 && conda activate yolo && pip install ultralytics",
        "6. Check PyTorch compatibility: pip install torch==2.1.0 torchvision==0.16.0",
        "7. Clear pip cache: pip cache purge",
        "8. Check permissions: chmod 644 /model/yolov8n.pt"
    ]
    
    for fix in fixes:
        print(fix)

if __name__ == "__main__":
    success = test_yolo_loading()
    
    if success:
        print("\nYOLOv8 loading test PASSED!")
        print("Your model should work in FastAPI now.")
    else:
        print("\nYOLOv8 loading test FAILED!")
        suggest_fixes()