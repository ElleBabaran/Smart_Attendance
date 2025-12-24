import cv2
from deepface import DeepFace
import os
import pickle
import numpy as np

# ============================================
# STEP 1: Generate Embeddings (Run this first!)
# ============================================
def generate_embeddings(db_path="students"):
    """Generate and save embeddings for all student images"""
    print("\n" + "="*50)
    print("GENERATING FACE EMBEDDINGS")
    print("="*50)
    
    if not os.path.exists(db_path):
        print(f"Error: {db_path} folder not found!")
        return
    
    # Collect all image files (including subfolders)
    image_data = []
    
    # Check root folder
    for f in os.listdir(db_path):
        full_path = os.path.join(db_path, f)
        if os.path.isfile(full_path) and f.lower().endswith(('.jpg', '.jpeg', '.png')):
            name = os.path.splitext(f)[0]
            image_data.append((full_path, name))
    
    # Check subfolders (each subfolder is a student)
    for folder in os.listdir(db_path):
        folder_path = os.path.join(db_path, folder)
        if os.path.isdir(folder_path):
            # Use folder name as student name
            student_name = folder
            for f in os.listdir(folder_path):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(folder_path, f)
                    image_data.append((img_path, student_name))
                    # No break - include ALL images per student
    
    if not image_data:
        print(f"No images found in {db_path}")
        print("\nExpected structure:")
        print("  students/")
        print("    StudentName1/")
        print("      photo.jpg")
        print("    StudentName2/")
        print("      photo.jpg")
        print("\nOR")
        print("  students/")
        print("    StudentName1.jpg")
        print("    StudentName2.jpg")
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
            
            # If student already has embeddings, average them
            if name in embeddings_cache:
                # Average the embeddings for better accuracy
                existing = np.array(embeddings_cache[name])
                new_emb = np.array(embedding)
                embeddings_cache[name] = ((existing + new_emb) / 2).tolist()
                print("✓ (averaged)")
            else:
                embeddings_cache[name] = embedding
                print("✓")
            
        except Exception as e:
            print(f"✗ Error: {e}")
    
    # Save embeddings
    if embeddings_cache:
        with open("face_embeddings.pkl", 'wb') as f:
            pickle.dump(embeddings_cache, f)
        print(f"\n✓ Successfully saved {len(embeddings_cache)} embeddings!")
        print("="*50 + "\n")
    else:
        print("\n✗ No embeddings generated. Check your images.")

# ============================================
# STEP 2: Real-time Recognition (Fast & Light)
# ============================================
class FastFaceRecognition:
    def __init__(self):
        # Load embeddings
        if not os.path.exists("face_embeddings.pkl"):
            print("Error: No embeddings found!")
            print("Run generate_embeddings() first!")
            return
        
        with open("face_embeddings.pkl", 'rb') as f:
            self.embeddings = pickle.load(f)
        
        print(f"✓ Loaded {len(self.embeddings)} students")
        
        # Load face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
    
    def find_match(self, face_embedding, threshold=10):
        """Find best match for face embedding"""
        min_dist = float('inf')
        best_match = "Unknown"
        
        for name, db_emb in self.embeddings.items():
            dist = np.linalg.norm(np.array(face_embedding) - np.array(db_emb))
            if dist < min_dist:
                min_dist = dist
                best_match = name
        
        if min_dist < threshold:
            return best_match, min_dist
        return "Unknown", min_dist
    
    def run(self, process_every=5, threshold=10):
        """Run real-time recognition
        
        Args:
            process_every: Process every N frames (higher = faster but less responsive)
            threshold: Recognition threshold (lower = stricter, higher = more lenient)
        """
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Cannot access webcam")
            return
        
        # Lower resolution for speed
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("\n" + "="*50)
        print("FACE RECOGNITION RUNNING")
        print("="*50)
        print(f"Students loaded: {len(self.embeddings)}")
        print(f"Processing every {process_every} frames")
        print("\nControls:")
        print("  q - Quit")
        print("  + - Increase threshold (more lenient)")
        print("  - - Decrease threshold (stricter)")
        print("="*50 + "\n")
        
        frame_count = 0
        last_detections = []  # Store last detections to avoid flickering
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Only process every Nth frame
            if frame_count % process_every == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                
                current_detections = []
                
                for (x, y, w, h) in faces:
                    # Extract face with padding
                    pad = 20
                    y1, y2 = max(0, y-pad), min(frame.shape[0], y+h+pad)
                    x1, x2 = max(0, x-pad), min(frame.shape[1], x+w+pad)
                    face_img = frame[y1:y2, x1:x2]
                    
                    try:
                        # Get embedding
                        result = DeepFace.represent(
                            img_path=face_img,
                            model_name="Facenet512",
                            detector_backend="skip",
                            enforce_detection=False
                        )
                        
                        if result:
                            embedding = result[0]["embedding"]
                            name, dist = self.find_match(embedding, threshold)
                            current_detections.append((x, y, w, h, name, dist))
                    
                    except:
                        current_detections.append((x, y, w, h, "Detecting...", 0))
                
                last_detections = current_detections
            
            # Draw stored detections
            for detection in last_detections:
                x, y, w, h, name, dist = detection
                
                color = (0, 255, 0) if name != "Unknown" and name != "Detecting..." else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                # Label
                if name != "Detecting...":
                    label = f"{name} ({dist:.1f})"
                else:
                    label = name
                
                cv2.rectangle(frame, (x, y-30), (x+200, y), color, -1)
                cv2.putText(frame, label, (x+5, y-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Display info
            info = f"Threshold: {threshold} | Press +/- to adjust"
            cv2.putText(frame, info, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            cv2.imshow('Face Recognition', frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('+') or key == ord('='):
                threshold += 1
                print(f"Threshold increased to {threshold}")
            elif key == ord('-') or key == ord('_'):
                threshold = max(1, threshold - 1)
                print(f"Threshold decreased to {threshold}")
        
        cap.release()
        cv2.destroyAllWindows()
        print("\nStopped.")

# ============================================
# MAIN - Choose what to run
# ============================================
if __name__ == "__main__":
    import sys
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    students_path = os.path.join(script_dir, "students")
    
    # Check if students folder exists in script directory
    if not os.path.exists(students_path):
        # Try parent directory
        students_path = os.path.join(os.path.dirname(script_dir), "students")
    
    print("\nFace Recognition System")
    print("="*50)
    print(f"Looking for students in: {students_path}")
    
    if os.path.exists(students_path):
        # Count images in root and subfolders
        num_images = 0
        for root, dirs, files in os.walk(students_path):
            num_images += len([f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        print(f"Found {num_images} student images")
        
        # Show folder structure
        subfolders = [d for d in os.listdir(students_path) 
                     if os.path.isdir(os.path.join(students_path, d))]
        if subfolders:
            print(f"Student folders: {', '.join(subfolders[:5])}" + 
                  (f" and {len(subfolders)-5} more" if len(subfolders) > 5 else ""))
    else:
        print("⚠ Warning: Students folder not found!")
        students_path = input("Enter path to students folder: ").strip()
    
    print("="*50)
    print("1. Generate embeddings (run this first!)")
    print("2. Start face recognition")
    print("="*50)
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        generate_embeddings(students_path)
    elif choice == "2":
        system = FastFaceRecognition()
        # Adjust these for performance:
        # process_every: higher = faster but less smooth (try 3-7)
        # threshold: higher = more lenient matching (try 8-12)
        system.run(process_every=5, threshold=10)
    else:
        print("Invalid choice!")