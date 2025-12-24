# Webcam Uniform Detection
# Simple real-time uniform detection using your trained model

from ultralytics import YOLO
import cv2

# ============================================
# CONFIGURATION
# ============================================

MODEL_PATH = 'runs/detect/uniform_detector2/weights/best.pt'
CONFIDENCE_THRESHOLD = 0.25  # Adjust this (0.1 = more detections, 0.5 = only confident ones)
WEBCAM_ID = 0 # 0 = default camera, try 1 or 2 if not working

# ============================================
# MAIN WEBCAM DETECTION
# ============================================

def run_webcam_detection():
    """Run real-time uniform detection on webcam"""
    
    print("="*60)
    print("WEBCAM UNIFORM DETECTION")
    print("="*60)
    print(f"Loading model from: {MODEL_PATH}")
    
    # Load your trained model
    model = YOLO(MODEL_PATH)
    
    print("Opening webcam...")
    cap = cv2.VideoCapture(WEBCAM_ID)
    
    if not cap.isOpened():
        print("ERROR: Could not open webcam!")
        print("Try changing WEBCAM_ID to 1 or 2")
        return
    
    print("✓ Webcam opened successfully!")
    print("\nControls:")
    print("  - Press 'Q' to quit")
    print("  - Press 'S' to take a screenshot")
    print(f"  - Current confidence threshold: {CONFIDENCE_THRESHOLD}")
    print("="*60)
    
    screenshot_count = 0
    
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        
        if not ret:
            print("ERROR: Failed to grab frame")
            break
        
        # Run detection on the frame
        results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        
        # Get annotated frame with boxes
        annotated_frame = results[0].plot()
        
        # Count detections
        num_detections = len(results[0].boxes)
        
        # Add text overlay with detection count
        cv2.putText(
            annotated_frame,
            f"Uniforms detected: {num_detections}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        # Add instructions
        cv2.putText(
            annotated_frame,
            "Press 'Q' to quit | 'S' for screenshot",
            (10, annotated_frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
        
        # Display the frame
        cv2.imshow('Uniform Detection - Press Q to quit', annotated_frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == ord('Q'):
            print("\nQuitting...")
            break
        elif key == ord('s') or key == ord('S'):
            screenshot_count += 1
            filename = f'screenshot_{screenshot_count}.jpg'
            cv2.imwrite(filename, annotated_frame)
            print(f"Screenshot saved: {filename}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam closed")

# ============================================
# RUN
# ============================================

if __name__ == "__main__":
    try:
        run_webcam_detection()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError occurred: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure your model path is correct")
        print("2. Try changing WEBCAM_ID to 1 or 2")
        print("3. Make sure opencv-python is installed: pip install opencv-python")
