# Webcam Uniform Detection - ULTRA STRICT MODE
# Maximum filtering to eliminate false positives

from ultralytics import YOLO
import cv2
import numpy as np

# ============================================
# CONFIGURATION
# ============================================

MODEL_PATH = 'runs/detect/uniform_detector2/weights/best.pt'
CONFIDENCE_THRESHOLD = 0.45
WEBCAM_ID = 0

# ULTRA STRICT FILTERING
STRICT_MODE = True
CLASS_MIN_CONFIDENCE = {
    'brown_dress': 0.70,      # ULTRA strict - only very confident detections
    'brown_pants': 0.70,      # ULTRA strict
    'white_polo': 0.40,       
    'green_polo': 0.40,       
}

# Minimum detection size
MIN_DETECTION_AREA = 8000  # Increased - must be reasonably large

# Color verification for brown items
ENABLE_COLOR_CHECK = True  # Set to False to disable color checking

# ============================================
# COLOR VERIFICATION
# ============================================

def check_brown_color(frame, box):
    """Verify if the detected region actually contains brown colors"""
    
    # Extract region
    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
    roi = frame[y1:y2, x1:x2]
    
    if roi.size == 0:
        return False
    
    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Brown color ranges in HSV
    # Brown is typically low saturation orange/red
    lower_brown1 = np.array([0, 30, 30])      # Reddish brown
    upper_brown1 = np.array([20, 200, 180])
    
    lower_brown2 = np.array([10, 40, 40])     # Orange brown
    upper_brown2 = np.array([30, 200, 150])
    
    # Create masks
    mask1 = cv2.inRange(hsv, lower_brown1, upper_brown1)
    mask2 = cv2.inRange(hsv, lower_brown2, upper_brown2)
    brown_mask = cv2.bitwise_or(mask1, mask2)
    
    # Calculate percentage of brown pixels
    brown_pixels = cv2.countNonZero(brown_mask)
    total_pixels = roi.shape[0] * roi.shape[1]
    brown_percentage = (brown_pixels / total_pixels) * 100
    
    # Require at least 15% brown pixels for brown items
    return brown_percentage > 15

# ============================================
# FILTERING FUNCTIONS
# ============================================

def filter_detections(results, frame, frame_shape, strict_mode, enable_color):
    """Apply ultra strict filtering to detections"""
    
    if not strict_mode:
        return None  # Will use original results
    
    filtered_boxes = []
    filtered_classes = []
    filtered_confs = []
    
    boxes = results[0].boxes
    
    if len(boxes) == 0:
        return [], [], [], results[0].names
    
    frame_height, frame_width = frame_shape[:2]
    frame_area = frame_height * frame_width
    
    for i, box in enumerate(boxes):
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = results[0].names[cls_id]
        
        # Get box coordinates
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        box_width = x2 - x1
        box_height = y2 - y1
        box_area = box_width * box_height
        
        # 1. Apply class-specific confidence threshold
        min_conf = CLASS_MIN_CONFIDENCE.get(class_name, CONFIDENCE_THRESHOLD)
        if conf < min_conf:
            continue
        
        # 2. Filter out tiny detections
        if box_area < MIN_DETECTION_AREA:
            continue
        
        # 3. Filter out detections too small relative to frame
        if box_area < (frame_area * 0.02):  # Must be at least 2% of frame
            continue
        
        # 4. Filter unrealistic aspect ratios
        aspect_ratio = box_width / box_height if box_height > 0 else 0
        
        if class_name == 'brown_dress':
            # Dresses should be vertical (height > width)
            if aspect_ratio > 1.2:  # Too wide for a dress
                continue
            if aspect_ratio < 0.3:  # Too narrow
                continue
                
        if class_name == 'brown_pants':
            # Pants should not be too wide
            if aspect_ratio > 1.8 or aspect_ratio < 0.4:
                continue
        
        if class_name in ['white_polo', 'green_polo']:
            # Polos should be roughly rectangular, slightly vertical
            if aspect_ratio > 1.5 or aspect_ratio < 0.5:
                continue
        
        # 5. COLOR VERIFICATION for brown items
        if enable_color and class_name in ['brown_dress', 'brown_pants']:
            if not check_brown_color(frame, box):
                continue  # Skip this detection - not actually brown
        
        # 6. Position check - uniforms typically appear in center region
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Allow detections in central 80% of frame
        if center_x < frame_width * 0.1 or center_x > frame_width * 0.9:
            continue
        if center_y < frame_height * 0.1 or center_y > frame_height * 0.9:
            continue
        
        # All checks passed - keep this detection
        filtered_boxes.append(box)
        filtered_classes.append(cls_id)
        filtered_confs.append(conf)
    
    return filtered_boxes, filtered_classes, filtered_confs, results[0].names

def draw_filtered_detections(frame, filtered_data):
    """Draw filtered detections on frame"""
    
    if filtered_data is None:
        return frame
    
    filtered_boxes, filtered_classes, filtered_confs, names = filtered_data
    annotated_frame = frame.copy()
    
    for box, cls_id, conf in zip(filtered_boxes, filtered_classes, filtered_confs):
        # Get coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        
        # Get class name
        class_name = names[cls_id]
        
        # Choose color based on class
        if 'brown' in class_name:
            color = (0, 102, 204)  # Brown-ish
        elif 'green' in class_name:
            color = (0, 255, 0)    # Green
        elif 'white' in class_name:
            color = (255, 255, 255)  # White
        else:
            color = (0, 255, 255)  # Yellow for others
        
        # Draw thicker box for emphasis
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
        
        # Draw label with confidence
        label = f"{class_name}: {conf:.2f}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(annotated_frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
        cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    return annotated_frame

# ============================================
# MAIN WEBCAM DETECTION
# ============================================

def run_webcam_detection():
    """Run real-time uniform detection on webcam with ultra strict filtering"""
    
    print("="*60)
    print("WEBCAM UNIFORM DETECTION - ULTRA STRICT MODE")
    print("="*60)
    print(f"Loading model from: {MODEL_PATH}")
    
    model = YOLO(MODEL_PATH)
    
    print("Opening webcam...")
    cap = cv2.VideoCapture(WEBCAM_ID)
    
    if not cap.isOpened():
        print("ERROR: Could not open webcam!")
        print("Try changing WEBCAM_ID to 1 or 2")
        return
    
    print("✓ Webcam opened successfully!")
    print("\nULTRA STRICT FILTERING ENABLED:")
    print(f"  - Base confidence: {CONFIDENCE_THRESHOLD}")
    for cls, conf in CLASS_MIN_CONFIDENCE.items():
        print(f"  - {cls}: {conf}")
    print(f"  - Min detection area: {MIN_DETECTION_AREA} pixels")
    print(f"  - Color verification: {'ENABLED' if ENABLE_COLOR_CHECK else 'DISABLED'}")
    print(f"  - Position filtering: ENABLED (center 80% only)")
    print(f"  - Aspect ratio filtering: ENABLED")
    print("\nControls:")
    print("  - Press 'Q' to quit")
    print("  - Press 'S' to take a screenshot")
    print("  - Press 'T' to toggle strict mode")
    print("  - Press 'C' to toggle color verification")
    print("="*60)
    
    screenshot_count = 0
    strict_mode = STRICT_MODE
    color_check = ENABLE_COLOR_CHECK
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("ERROR: Failed to grab frame")
            break
        
        # Run detection
        results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        
        # Apply strict filtering
        if strict_mode and len(results[0].boxes) > 0:
            filtered_data = filter_detections(results, frame, frame.shape, strict_mode, color_check)
            annotated_frame = draw_filtered_detections(frame, filtered_data)
            num_detections = len(filtered_data[0]) if filtered_data else 0
        else:
            annotated_frame = results[0].plot()
            num_detections = len(results[0].boxes)
        
        # Add info overlay
        mode_text = "ULTRA STRICT" if strict_mode else "NORMAL"
        color_text = "COLOR: ON" if color_check else "COLOR: OFF"
        
        cv2.putText(annotated_frame, f"Detections: {num_detections} | {mode_text} | {color_text}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(annotated_frame, "Q=Quit | S=Screenshot | T=Toggle Strict | C=Color Check",
                    (10, annotated_frame.shape[0] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('Uniform Detection - ULTRA STRICT MODE', annotated_frame)
        
        # Handle keyboard
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == ord('Q'):
            print("\nQuitting...")
            break
        elif key == ord('s') or key == ord('S'):
            screenshot_count += 1
            filename = f'screenshot_{screenshot_count}.jpg'
            cv2.imwrite(filename, annotated_frame)
            print(f"Screenshot saved: {filename}")
        elif key == ord('t') or key == ord('T'):
            strict_mode = not strict_mode
            print(f"Strict mode: {'ON' if strict_mode else 'OFF'}")
        elif key == ord('c') or key == ord('C'):
            color_check = not color_check
            print(f"Color verification: {'ON' if color_check else 'OFF'}")
    
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
        import traceback
        traceback.print_exc()