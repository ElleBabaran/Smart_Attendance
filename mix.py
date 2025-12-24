import cv2
from deepface import DeepFace
from ultralytics import YOLO
import os
import pickle
import numpy as np

# ============================================
# CONFIGURATION
# ============================================
UNIFORM_MODEL_PATH = 'runs/detect/uniform_detector2/weights/best.pt'
UNIFORM_CONFIDENCE = 0.25
FACE_EMBEDDINGS_PATH = "face_embeddings.pkl"
FACE_THRESHOLD = 10
PROCESS_EVERY_N_FRAMES = 5
WEBCAM_ID = 0

# ============================================
# STEP 1: Generate Face Embeddings (Run First)
# ============================================
def generate_embeddings(db_path="students"):
    """Generate and save embeddings for all student images"""
    print("\n" + "="*50)
    print("GENERATING FACE EMBEDDINGS")
    print("="*50)
    
    if not os.path.exists(db_path):
        print(f"Error: {db_path} folder not found!")
        return
    
    image_data = []
    
    # Check root folder
    for f in os.listdir(db_path):
        full_path = os.path.join(db_path, f)
        if os.path.isfile(full_path) and f.lower().endswith(('.jpg', '.jpeg', '.png')):
            name = os.path.splitext(f)[0]
            image_data.append((full_path, name))
    
    # Check subfolders
    for folder in os.listdir(db_path):
        folder_path = os.path.join(db_path, folder)
        if os.path.isdir(folder_path):
            student_name = folder
            for f in os.listdir(folder_path):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(folder_path, f)
                    image_data.append((img_path, student_name))
    
    if not image_data:
        print(f"No images found in {db_path}")
        return
    
    print(f"Found {len(image_data)} images\n")
    
    embeddings_cache = {}
    
    for i, (img_path, name) in enumerate(image_data, 1):
        print(f"[{i}/{len(image_data)}] Processing: {name}...", end=" ")
        
        try:
            result = DeepFace.represent(
                img_path=img_path,
                model_name="Facenet512",
                detector_backend="opencv",
                enforce_detection=True
            )
            
            embedding = result[0]["embedding"]
            
            if name in embeddings_cache:
                existing = np.array(embeddings_cache[name])
                new_emb = np.array(embedding)
                embeddings_cache[name] = ((existing + new_emb) / 2).tolist()
                print("✓ (averaged)")
            else:
                embeddings_cache[name] = embedding
                print("✓")
            
        except Exception as e:
            print(f"✗ Error: {e}")
    
    if embeddings_cache:
        with open(FACE_EMBEDDINGS_PATH, 'wb') as f:
            pickle.dump(embeddings_cache, f)
        print(f"\n✓ Successfully saved {len(embeddings_cache)} embeddings!")
        print("="*50 + "\n")
    else:
        print("\n✗ No embeddings generated. Check your images.")

# ============================================
# STEP 2: Integrated System
# ============================================
class IntegratedSystem:
    def __init__(self):
        # Load face embeddings
        if not os.path.exists(FACE_EMBEDDINGS_PATH):
            print("Error: No face embeddings found!")
            print("Run generate_embeddings() first!")
            return
        
        with open(FACE_EMBEDDINGS_PATH, 'rb') as f:
            self.face_embeddings = pickle.load(f)
        
        print(f"✓ Loaded {len(self.face_embeddings)} students")
        
        # Load face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Load uniform detection model
        print(f"Loading uniform model: {UNIFORM_MODEL_PATH}")
        self.uniform_model = YOLO(UNIFORM_MODEL_PATH)
        print("✓ Uniform model loaded")
    
    def find_face_match(self, face_embedding, threshold):
        """Find best match for face embedding"""
        min_dist = float('inf')
        best_match = "Unknown"
        
        for name, db_emb in self.face_embeddings.items():
            dist = np.linalg.norm(np.array(face_embedding) - np.array(db_emb))
            if dist < min_dist:
                min_dist = dist
                best_match = name
        
        if min_dist < threshold:
            return best_match
        return "Unknown"
    
    def run(self):
        """Run face recognition + uniform parts detection"""
        cap = cv2.VideoCapture(WEBCAM_ID)
        
        if not cap.isOpened():
            print("Error: Cannot access webcam")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("\n" + "="*60)
        print("FACE RECOGNITION + UNIFORM PARTS DETECTION")
        print("="*60)
        print(f"Students loaded: {len(self.face_embeddings)}")
        print("\nControls:")
        print("  Q - Quit")
        print("  S - Screenshot")
        print("  + - Increase threshold (more lenient)")
        print("  - - Decrease threshold (stricter)")
        print("="*60 + "\n")
        
        frame_count = 0
        last_face_detections = []
        face_threshold = FACE_THRESHOLD
        screenshot_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # ======== FACE RECOGNITION ========
            if frame_count % PROCESS_EVERY_N_FRAMES == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                
                current_face_detections = []
                
                for (x, y, w, h) in faces:
                    pad = 20
                    y1, y2 = max(0, y-pad), min(frame.shape[0], y+h+pad)
                    x1, x2 = max(0, x-pad), min(frame.shape[1], x+w+pad)
                    face_img = frame[y1:y2, x1:x2]
                    
                    try:
                        result = DeepFace.represent(
                            img_path=face_img,
                            model_name="Facenet512",
                            detector_backend="skip",
                            enforce_detection=False
                        )
                        
                        if result:
                            embedding = result[0]["embedding"]
                            name = self.find_face_match(embedding, face_threshold)
                            current_face_detections.append((x, y, w, h, name))
                    except:
                        current_face_detections.append((x, y, w, h, "Detecting..."))
                
                last_face_detections = current_face_detections
            
            # Draw face boxes with names only
            for detection in last_face_detections:
                x, y, w, h, name = detection
                
                # Blue for recognized, red for unknown
                color = (255, 0, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                # Draw name label
                label_width = len(name) * 10 + 10
                cv2.rectangle(frame, (x, y-30), (x+label_width, y), color, -1)
                cv2.putText(frame, name, (x+5, y-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # ======== UNIFORM PARTS DETECTION ========
            uniform_results = self.uniform_model(frame, conf=UNIFORM_CONFIDENCE, verbose=False)
            
            # Count uniform parts
            uniform_parts = {}
            
            # Draw boxes for each uniform part
            for box in uniform_results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                # Get class name (uniform part name)
                class_name = uniform_results[0].names[cls]
                
                # Count parts
                if class_name not in uniform_parts:
                    uniform_parts[class_name] = 0
                uniform_parts[class_name] += 1
                
                # Green boxes for uniform parts
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                label = f"{class_name} {conf:.2f}"
                label_width = len(label) * 8 + 10
                cv2.rectangle(frame, (x1, y1-25), (x1+label_width, y1), (0, 255, 0), -1)
                cv2.putText(frame, label, (x1+5, y1-8),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # ======== INFO OVERLAY ========
            num_faces = len(last_face_detections)
            
            info = f"Faces: {num_faces} | Threshold: {face_threshold}"
            cv2.putText(frame, info, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Show uniform parts count
            y_offset = 60
            for part_name, count in uniform_parts.items():
                part_info = f"{part_name}: {count}"
                cv2.putText(frame, part_info, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                y_offset += 25
            
            cv2.putText(frame, "Q=Quit | S=Screenshot | +/- Adjust",
                       (10, frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Face Recognition + Uniform Parts Detection', frame)
            
            # ======== KEYBOARD CONTROLS ========
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                break
            elif key == ord('s') or key == ord('S'):
                screenshot_count += 1
                filename = f'screenshot_{screenshot_count}.jpg'
                cv2.imwrite(filename, frame)
                print(f"Screenshot saved: {filename}")
            elif key == ord('+') or key == ord('='):
                face_threshold += 1
                print(f"Face threshold increased to {face_threshold}")
            elif key == ord('-') or key == ord('_'):
                face_threshold = max(1, face_threshold - 1)
                print(f"Face threshold decreased to {face_threshold}")
        
        cap.release()
        cv2.destroyAllWindows()
        print("\nStopped.")

# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    students_path = os.path.join(script_dir, "students")
    
    if not os.path.exists(students_path):
        students_path = os.path.join(os.path.dirname(script_dir), "students")
    
    print("\nFace Recognition + Uniform Parts Detection System")
    print("="*60)
    print(f"Looking for students in: {students_path}")
    
    if os.path.exists(students_path):
        num_images = 0
        for root, dirs, files in os.walk(students_path):
            num_images += len([f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        print(f"Found {num_images} student images")
    else:
        print("⚠ Warning: Students folder not found!")
    
    print("="*60)
    print("2. Start recognition system")
    print("="*60)
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        generate_embeddings(students_path)
    elif choice == "2":
        system = IntegratedSystem()
        system.run()
    else:
        print("Invalid choice!")