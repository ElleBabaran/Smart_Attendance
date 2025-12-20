import tkinter as tk
from tkinter import messagebox, ttk
import cv2
from PIL import Image, ImageTk
from roboflow import Roboflow
import sqlite3
from datetime import datetime
import threading
import time
import queue
import os
from deepface import DeepFace
import winsound
import pickle
import numpy as np

class AttendanceSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Attendance System")
        self.root.geometry("600x500")
        
        self.bg_primary = "#1a1a2e"
        self.bg_secondary = "#16213e"
        self.accent_color = "#0f3460"
        self.highlight = "#e94560"
        self.success = "#06ffa5"
        self.warning = "#ffd700"
        self.text_light = "#eaeaea"
        self.root.configure(bg=self.bg_primary)

        self.cap = None
        self.camera_active = False
        self.selected_camera_index = 0
        self.model = None

        self.female_items = ['brown dress', 'socks', 'black shoes', 'belt', 'id', 'ribbon']
        self.male_items = ['white polo', 'id', 'socks', 'black shoes', 'brown pants']
        self.current_gender = None
        self.last_predictions = []
        self.detected_items_history = set()
        self.last_save_time = 0

        self.frame_queue = queue.Queue(maxsize=1)
        self.result_queue = queue.Queue(maxsize=1)
        self.face_queue = queue.Queue(maxsize=1)
        self.face_result_queue = queue.Queue(maxsize=1)
        self.detection_running = False
        self.face_recognition_running = False

        self.current_last_name = "Unknown"
        self.current_first_name = ""
        self.face_box = None

        self.detection_complete = False
        self.sound_played = False

        self.CLASS_COLORS = {
            "white polo": (0, 255, 0), "brown pants": (0, 0, 255), "brown dress": (255, 0, 0),
            "ribbon": (255, 165, 0), "socks": (255, 255, 0), "black shoes": (128, 0, 128),
            "id": (0, 255, 255), "belt": (255, 20, 147),
        }

        self.students_dir = "students"
        self.embeddings_data = self.load_embeddings()
        self.init_database()

        API_KEY = "2ypTreeDqECVWLmZgvXe"
        try:
            rf = Roboflow(api_key=API_KEY)
            self.model = rf.workspace("elle-5vs9o").project("uniformmm-ubp4u").version(3).model
            print("✓ Model loaded!")
        except Exception as e:
            print(f"✗ Model error: {e}")
            self.model = None

        self.show_main_menu()

    def load_embeddings(self):
        if os.path.exists("embeddings.pkl"):
            try:
                with open("embeddings.pkl", "rb") as f:
                    data = pickle.load(f)
                print(f"✓ Loaded embeddings for {len(data)} students")
                return data
            except Exception as e:
                print(f"⚠ Could not load embeddings: {e}")
        else:
            print("⚠ embeddings.pkl not found - run generate_embeddings.py first!")
        return {}

    def init_database(self):
        self.conn = sqlite3.connect('attendance.db', check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT, last_name TEXT NOT NULL, first_name TEXT,
            uniform_status TEXT NOT NULL, missing_items TEXT, time TEXT NOT NULL,
            status TEXT NOT NULL, date TEXT NOT NULL)''')
        self.conn.commit()
        print("✓ Database initialized")

    def save_to_database(self, last_name, first_name, uniform_status, missing_items, detected_items=None):
        current_time = datetime.now()
        date_str = current_time.strftime("%Y-%m-%d")
        time_str = current_time.strftime("%I:%M %p")
        cutoff_time = current_time.replace(hour=7, minute=1, second=0, microsecond=0)
        status = "Late" if current_time > cutoff_time else "On Time"
        missing_str = ", ".join(missing_items) if missing_items else "None"

        try:
            self.cursor.execute('''SELECT id, missing_items FROM attendance
                WHERE last_name = ? AND first_name = ? AND date = ? ORDER BY id DESC LIMIT 1''',
                (last_name, first_name, date_str))
            existing_record = self.cursor.fetchone()

            if existing_record and last_name != "Unknown":
                existing_id, existing_missing_str = existing_record
                existing_missing = set(existing_missing_str.split(", ")) if existing_missing_str != "None" else set()
                updated_missing = existing_missing - detected_items if detected_items else existing_missing - set(missing_items)
                updated_missing_str = ", ".join(sorted(updated_missing)) if updated_missing else "None"
                updated_uniform_status = "Complete" if not updated_missing or updated_missing_str == "None" else "Incomplete"
                self.cursor.execute('''UPDATE attendance SET uniform_status = ?, missing_items = ?, time = ? WHERE id = ?''',
                    (updated_uniform_status, updated_missing_str, time_str, existing_id))
                self.conn.commit()
                print(f"✓ UPDATED: {last_name}, {first_name} - {updated_uniform_status}")
            else:
                self.cursor.execute('''INSERT INTO attendance (last_name, first_name, uniform_status, missing_items, time, status, date)
                    VALUES (?, ?, ?, ?, ?, ?, ?)''', (last_name, first_name, uniform_status, missing_str, time_str, status, date_str))
                self.conn.commit()
                print(f"✓ SAVED: {last_name}, {first_name} - {uniform_status}")
            return True
        except Exception as e:
            print(f"✗ Database error: {e}")
            return False

    def play_success_sound(self):
        try:
            winsound.Beep(1000, 200)
            time.sleep(0.1)
            winsound.Beep(1200, 200)
        except:
            pass

    def show_main_menu(self):
        for widget in self.root.winfo_children():
            widget.destroy()
        self.root.geometry("600x500")
        main_frame = tk.Frame(self.root, bg=self.bg_primary)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=40, pady=40)

        tk.Label(main_frame, text="👤", font=("Arial", 40), bg=self.bg_secondary, fg=self.success).pack(pady=(0, 20))
        tk.Label(main_frame, text="Smart Attendance", font=("Segoe UI", 36, "bold"),
                 bg=self.bg_primary, fg=self.text_light).pack(pady=(0, 5))
        tk.Label(main_frame, text="Uniform & Face Detection System", font=("Segoe UI", 14),
                 bg=self.bg_primary, fg="#888888").pack(pady=(0, 30))

        button_font = ("Segoe UI", 14, "bold")
        tk.Button(main_frame, text="START DETECTION", font=button_font, bg=self.success, fg=self.bg_primary,
                  width=25, height=2, cursor="hand2", border=0, relief=tk.FLAT, command=self.start_camera).pack(pady=12)
        tk.Button(main_frame, text="ADMIN PANEL", font=button_font, bg=self.accent_color, fg=self.text_light,
                  width=25, height=2, cursor="hand2", border=0, relief=tk.FLAT, command=self.open_admin).pack(pady=12)
        tk.Label(main_frame, text="Powered by Roboflow & DeepFace • v2.1", font=("Segoe UI", 9),
                 bg=self.bg_primary, fg="#666666").pack(side=tk.BOTTOM, pady=20)

    def start_camera(self):
        if self.model is None:
            messagebox.showerror("Error", "Roboflow model not loaded!")
            return
        self.open_camera_window()
        
        def init_camera():
            try:
                self.cap = cv2.VideoCapture(self.selected_camera_index)
                if not self.cap.isOpened():
                    raise Exception("Camera not accessible")
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                time.sleep(0.5)
                
                self.camera_active = True
                self.detection_running = True
                self.face_recognition_running = True
                self.last_predictions = []
                self.last_save_time = 0
                self.frame_count = 0
                self.current_last_name = "Unknown"
                self.current_first_name = ""
                self.face_box = None
                self.detection_complete = False
                self.sound_played = False
                self.current_gender = None
                self.detected_items_history = set()
                
                while not self.frame_queue.empty():
                    try: self.frame_queue.get_nowait()
                    except: break
                while not self.result_queue.empty():
                    try: self.result_queue.get_nowait()
                    except: break
                while not self.face_queue.empty():
                    try: self.face_queue.get_nowait()
                    except: break
                while not self.face_result_queue.empty():
                    try: self.face_result_queue.get_nowait()
                    except: break
                
                threading.Thread(target=self.detection_worker, daemon=True).start()
                threading.Thread(target=self.face_recognition_worker, daemon=True).start()
                print("✓ Camera initialized!")
                self.root.after(30, self.update_camera)
            except Exception as e:
                print(f"✗ Camera error: {e}")
                self.root.after(100, lambda: messagebox.showerror("Error", f"Failed to open camera: {e}"))
        
        threading.Thread(target=init_camera, daemon=True).start()

    def detection_worker(self):
        while self.detection_running:
            try:
                frame = self.frame_queue.get(timeout=1.0)
                temp_file = "temp_frame.jpg"
                cv2.imwrite(temp_file, frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                result = self.model.predict(temp_file, confidence=50, overlap=30)
                predictions = result.json()
                try: os.remove(temp_file)
                except: pass
                
                if 'predictions' in predictions:
                    filtered = [p for p in predictions['predictions'] if int(p['width']) >= 50 and int(p['height']) >= 50 and p['confidence'] >= 0.5]
                    try:
                        if self.result_queue.empty():
                            self.result_queue.put_nowait(filtered)
                    except queue.Full:
                        pass
                else:
                    try:
                        if self.result_queue.empty():
                            self.result_queue.put_nowait([])
                    except queue.Full:
                        pass
            except queue.Empty:
                continue
            except Exception as e:
                print(f"✗ Detection error: {e}")
                time.sleep(0.5)

    def face_recognition_worker(self):
        """Ultra-relaxed face recognition - will detect almost anything"""
        print("🚀 Face recognition worker started!")
        frame_count = 0
        
        while self.face_recognition_running:
            try:
                frame = self.face_queue.get(timeout=0.5)
                frame_count += 1
                print(f"\n{'='*60}")
                print(f"📸 Processing frame #{frame_count} - Shape: {frame.shape}")
                
                try:
                    # Try multiple backends
                    backends = ['retinaface', 'opencv', 'ssd', 'mtcnn']
                    faces = None
                    used_backend = None
                    
                    for backend in backends:
                        try:
                            print(f"   Trying {backend} detector...")
                            faces = DeepFace.extract_faces(
                                frame, 
                                detector_backend=backend,
                                enforce_detection=False,
                                align=True
                            )
                            if faces and len(faces) > 0:
                                used_backend = backend
                                print(f"   ✓ {backend} found {len(faces)} face(s)!")
                                break
                            else:
                                print(f"   ✗ {backend} found nothing")
                        except Exception as e:
                            print(f"   ✗ {backend} error: {e}")
                            continue
                    
                    if not faces or len(faces) == 0:
                        print("❌ NO FACES DETECTED BY ANY BACKEND")
                        continue
                    
                    print(f"\n✅ SUCCESS! Using {used_backend}")
                    print(f"   Total faces found: {len(faces)}")
                    
                    # Process ALL detected faces and show info
                    for idx, face_data in enumerate(faces):
                        print(f"\n   --- Face #{idx+1} ---")
                        confidence = face_data.get('confidence', 0)
                        x = face_data['facial_area']['x']
                        y = face_data['facial_area']['y']
                        w = face_data['facial_area']['w']
                        h = face_data['facial_area']['h']
                        aspect_ratio = w / h if h > 0 else 0
                        
                        print(f"   Confidence: {confidence:.3f}")
                        print(f"   Position: ({x}, {y})")
                        print(f"   Size: {w}x{h}")
                        print(f"   Aspect ratio: {aspect_ratio:.2f}")
                    
                    # Use the first face
                    face_data = faces[0]
                    confidence = face_data.get('confidence', 0)
                    x = face_data['facial_area']['x']
                    y = face_data['facial_area']['y']
                    w = face_data['facial_area']['w']
                    h = face_data['facial_area']['h']
                    
                    # NO VALIDATION - Accept everything
                    print(f"\n   ✓ ACCEPTING THIS FACE (no validation)")
                    
                    # Crop with padding
                    padding = 20
                    y1 = max(0, y - padding)
                    y2 = min(frame.shape[0], y + h + padding)
                    x1 = max(0, x - padding)
                    x2 = min(frame.shape[1], x + w + padding)
                    face_crop = frame[y1:y2, x1:x2]
                    
                    print(f"   Cropped face size: {face_crop.shape}")
                    
                    if face_crop.size == 0:
                        print("   ✗ SKIP: Empty crop")
                        continue
                    
                    print("   🔄 Attempting recognition...")
                    
                    # Try recognition
                    last_name, first_name, distance = self.recognize_student_with_embeddings(face_crop)
                    
                    print(f"\n   🎯 RESULT: {last_name}, {first_name}")
                    print(f"   Distance: {distance:.3f}")
                    
                    # Accept EVERYTHING - even Unknown
                    result = {
                        'last_name': last_name if last_name else "Unknown",
                        'first_name': first_name if first_name else "",
                        'box': (x, y, w, h),
                        'confidence': confidence,
                        'distance': distance
                    }
                    
                    try:
                        # Clear old results
                        while not self.face_result_queue.empty():
                            try:
                                self.face_result_queue.get_nowait()
                            except:
                                break
                        
                        self.face_result_queue.put_nowait(result)
                        print(f"   ✅ SENT TO QUEUE: {last_name}, {first_name}")
                    except queue.Full:
                        print("   ⚠ Queue full, clearing...")
                        
                except Exception as e:
                    print(f"❌ Face extraction error: {e}")
                    import traceback
                    traceback.print_exc()
                    
            except queue.Empty:
                # This is normal, no frame to process
                pass
            except Exception as e:
                print(f"❌ Face recognition worker error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.3)
        
        print("🛑 Face recognition worker stopped!")

    def recognize_student_with_embeddings(self, face_crop):
        """Ultra-detailed recognition"""
        print(f"\n   {'─'*50}")
        print(f"   🔍 RECOGNITION ATTEMPT")
        print(f"   {'─'*50}")
        
        if not self.embeddings_data:
            print("   ❌ No embeddings data!")
            print(f"   Embeddings file exists: {os.path.exists('embeddings.pkl')}")
            return "Unknown", "", float('inf')
        
        print(f"   ✓ Embeddings loaded: {len(self.embeddings_data)} students")
        for name in list(self.embeddings_data.keys())[:5]:
            print(f"      - {name}")
        if len(self.embeddings_data) > 5:
            print(f"      ... and {len(self.embeddings_data) - 5} more")
        
        try:
            print("   Generating embedding...")
            detected_embedding = DeepFace.represent(
                face_crop,
                model_name='Facenet',
                enforce_detection=False
            )
            
            if not detected_embedding:
                print("   ❌ Could not generate embedding")
                return "Unknown", "", float('inf')
            
            detected_vector = np.array(detected_embedding[0]['embedding'])
            print(f"   ✓ Embedding generated!")
            print(f"      Vector length: {len(detected_vector)}")
            print(f"      Vector norm: {np.linalg.norm(detected_vector):.3f}")
            print(f"      First 5 values: {detected_vector[:5]}")
            
            # Validate
            if np.all(detected_vector == 0):
                print("   ❌ Embedding is all zeros!")
                return "Unknown", "", float('inf')
            
            if np.any(np.isnan(detected_vector)):
                print("   ❌ Embedding contains NaN!")
                return "Unknown", "", float('inf')
            
            print("\n   📊 Comparing with database...")
            
            best_match = None
            best_distance = float('inf')
            all_results = []
            
            for student_name, embeddings_list in self.embeddings_data.items():
                print(f"      Checking {student_name} ({len(embeddings_list)} embeddings)...")
                
                distances = []
                for emb_data in embeddings_list:
                    stored_vector = np.array(emb_data[0]['embedding'])
                    distance = np.linalg.norm(detected_vector - stored_vector)
                    distances.append(distance)
                
                min_dist = np.min(distances)
                avg_dist = np.mean(distances)
                
                all_results.append({
                    'name': student_name,
                    'min': min_dist,
                    'avg': avg_dist
                })
                
                if min_dist < best_distance:
                    best_distance = min_dist
                    best_match = student_name
            
            # Sort and show top 5
            all_results.sort(key=lambda x: x['min'])
            print(f"\n   🏆 TOP 5 MATCHES:")
            for i, result in enumerate(all_results[:5]):
                marker = "👉" if result['name'] == best_match else "  "
                print(f"      {marker} {i+1}. {result['name']}")
                print(f"         Min: {result['min']:.4f}, Avg: {result['avg']:.4f}")
            
            # NO THRESHOLD - return best match always
            if best_match:
                if ',' in best_match:
                    parts = best_match.split(',')
                    last_name = parts[0].strip()
                    first_name = parts[1].strip() if len(parts) > 1 else ""
                else:
                    last_name = best_match.strip()
                    first_name = ""
                
                print(f"\n   ✅ BEST MATCH: {last_name}, {first_name}")
                print(f"      Distance: {best_distance:.4f}")
                return last_name, first_name, best_distance
            else:
                print(f"\n   ❌ No match found")
                return "Unknown", "", float('inf')
                
        except Exception as e:
            print(f"   ❌ Recognition exception: {e}")
            import traceback
            traceback.print_exc()
            return "Unknown", "", float('inf')

    def open_camera_window(self):
        for widget in self.root.winfo_children():
            widget.destroy()
        self.root.geometry("1100x750")

        header_frame = tk.Frame(self.root, bg=self.bg_secondary, height=80)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        tk.Label(header_frame, text="🎥  Live Detection", font=("Segoe UI", 22, "bold"),
                 bg=self.bg_secondary, fg=self.text_light).pack(pady=20)

        status_card = tk.Frame(self.root, bg=self.bg_secondary, height=100)
        status_card.pack(fill=tk.X, padx=30, pady=15)
        status_card.pack_propagate(False)
        self.status_label = tk.Label(status_card, text="Initializing camera...", font=("Segoe UI", 16, "bold"),
                                     bg=self.bg_secondary, fg=self.warning)
        self.status_label.pack(pady=10)
        self.missing_label = tk.Label(status_card, text="Please wait...", font=("Segoe UI", 11),
                                      bg=self.bg_secondary, fg="#ff6b6b", wraplength=1000)
        self.missing_label.pack(pady=5)

        camera_frame = tk.Frame(self.root, bg=self.bg_primary)
        camera_frame.pack(pady=10, padx=30, fill=tk.BOTH, expand=True)
        self.camera_label = tk.Label(camera_frame, bg=self.bg_primary, text="Starting camera...",
                                     font=("Segoe UI", 14), fg=self.text_light)
        self.camera_label.pack(fill=tk.BOTH, expand=True)

        back_button = tk.Button(camera_frame, text="← BACK TO MENU", font=("Segoe UI", 11, "bold"),
                                bg=self.highlight, fg=self.text_light, width=18, height=2,
                                cursor="hand2", border=0, relief=tk.FLAT, command=self.stop_camera)
        back_button.place(relx=1.0, rely=1.0, anchor=tk.SE, x=-20, y=-20)

    def check_uniform_completeness(self, predictions):
        detected_items = set(p['class'].lower().strip().replace('_', ' ') for p in predictions)
        self.detected_items_history.update(detected_items)

        if 'brown dress' in self.detected_items_history or 'ribbon' in self.detected_items_history:
            self.current_gender = "Female"
            required_items = self.female_items
        elif 'white polo' in self.detected_items_history or 'brown pants' in self.detected_items_history:
            self.current_gender = "Male"
            required_items = self.male_items
        else:
            female_missing = sum(1 for item in self.female_items if item.lower() not in self.detected_items_history)
            male_missing = sum(1 for item in self.male_items if item.lower() not in self.detected_items_history)
            self.current_gender = "Female" if female_missing <= male_missing else "Male"
            required_items = self.female_items if self.current_gender == "Female" else self.male_items

        missing_items = [item for item in required_items if item.lower() not in self.detected_items_history]
        is_complete = len(missing_items) == 0
        return is_complete, missing_items, detected_items

    def update_camera(self):
        try:
            if not self.camera_active or not self.cap or not self.cap.isOpened():
                return
            ret, frame = self.cap.read()
            if not ret:
                self.root.after(30, self.update_camera)
                return
            
            display_frame = frame.copy()
            current_time = time.time()
            self.frame_count += 1

            if self.frame_count % 5 == 0:
                try:
                    if self.frame_queue.empty():
                        self.frame_queue.put_nowait(frame.copy())
                except queue.Full:
                    pass

            if self.frame_count % 3 == 0:
                try:
                    while not self.face_queue.empty():
                        try:
                            self.face_queue.get_nowait()
                        except:
                            break
                    self.face_queue.put_nowait(frame.copy())
                    print(f"📤 Sent frame {self.frame_count} to face queue")
                except queue.Full:
                    pass

            try:
                new_predictions = self.result_queue.get_nowait()
                self.last_predictions = new_predictions
            except queue.Empty:
                pass

            try:
                face_result = self.face_result_queue.get_nowait()
                self.current_last_name = face_result['last_name']
                self.current_first_name = face_result['first_name']
                self.face_box = face_result['box']
                print(f"✅ DISPLAY UPDATE: {self.current_last_name}, {self.current_first_name}")
            except queue.Empty:
                pass

            is_complete = False
            missing = []
            detected = set()

            if self.last_predictions:
                is_complete, missing, detected = self.check_uniform_completeness(self.last_predictions)
                for pred in self.last_predictions:
                    w, h = int(pred['width']), int(pred['height'])
                    if w < 20 or h < 20:
                        continue
                    x = int(pred['x'] - w / 2)
                    y = int(pred['y'] - h / 2)
                    color = self.CLASS_COLORS.get(pred['class'].lower(), (255, 255, 255))
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
                    label = f"{pred['class']} {pred['confidence']:.0%}"
                    cv2.rectangle(display_frame, (x, y-20), (x + len(label)*8, y), color, -1)
                    cv2.putText(display_frame, label, (x+3, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            if self.face_box:
                x, y, w, h = self.face_box
                if self.current_last_name and self.current_last_name != "Unknown":
                    box_color = (0, 255, 0)
                    name_text = f"{self.current_last_name}, {self.current_first_name}"
                else:
                    box_color = (255, 0, 0)
                    name_text = "Unknown"
                
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), box_color, 3)
                text_size = cv2.getTextSize(name_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.rectangle(display_frame, (x, y-30), (x + text_size[0] + 10, y), box_color, -1)
                cv2.putText(display_frame, name_text, (x+5, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if self.last_predictions:
                gender_text = f"({self.current_gender})" if self.current_gender else ""
                if is_complete:
                    self.status_label.config(text=f"✓ COMPLETE UNIFORM {gender_text}", fg=self.success)
                    self.missing_label.config(text="All required items present ✓", fg=self.success)
                else:
                    self.status_label.config(text=f"⚠ INCOMPLETE UNIFORM {gender_text}", fg=self.highlight)
                    self.missing_label.config(text=f"Missing: {', '.join(missing)}", fg=self.highlight)
            else:
                self.status_label.config(text="👁 Scanning for uniform...", fg=self.warning)
                self.missing_label.config(text="Position yourself in front of the camera", fg=self.warning)

            if (current_time - self.last_save_time >= 5.0):
                if self.last_predictions or self.face_box:
                    self.last_save_time = current_time
                    if self.last_predictions:
                        uniform_status = "Complete" if is_complete else "Incomplete"
                    else:
                        uniform_status = "Not Detected"
                        missing = []
                    save_last_name = self.current_last_name if self.current_last_name else "Unknown"
                    save_first_name = self.current_first_name if self.current_first_name else ""
                    success = self.save_to_database(save_last_name, save_first_name, uniform_status, missing, detected)
                    if success and not self.detection_complete:
                        self.detection_complete = True
                        if not self.sound_played:
                            threading.Thread(target=self.play_success_sound, daemon=True).start()
                            self.sound_played = True
                        threading.Timer(3.0, self.reset_detection_state).start()

            try:
                frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                frame_rgb = cv2.resize(frame_rgb, (640, 480))
                img = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
                self.camera_label.config(image=img, text="")
                self.camera_label.image = img
            except Exception as e:
                print(f"✗ Display error: {e}")
        except Exception as e:
            print(f"✗ Update error: {e}")
        finally:
            self.root.after(30, self.update_camera)

    def reset_detection_state(self):
        self.detection_complete = False
        self.sound_played = False
        self.current_last_name = "Unknown"
        self.current_first_name = ""
        self.face_box = None
        self.detected_items_history.clear()
        print("\n🔄 Ready for next student...\n")

    def stop_camera(self):
        self.camera_active = False
        self.detection_running = False
        self.face_recognition_running = False
        time.sleep(0.3)
        if self.cap:
            self.cap.release()
            self.cap = None
        self.show_main_menu()

    def open_admin(self):
        for widget in self.root.winfo_children():
            widget.destroy()
        self.root.geometry("1200x700")
        
        header_frame = tk.Frame(self.root, bg=self.bg_secondary, height=80)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        tk.Label(header_frame, text="📊  Admin Panel - Attendance Records", font=("Segoe UI", 22, "bold"),
                 bg=self.bg_secondary, fg=self.text_light).pack(pady=20)
        
        controls_frame = tk.Frame(self.root, bg=self.bg_primary)
        controls_frame.pack(fill=tk.X, padx=30, pady=15)

        self.search_var = tk.StringVar()
        self.search_entry = tk.Entry(controls_frame, textvariable=self.search_var, font=("Segoe UI", 11), width=30,
                                     bg=self.bg_secondary, fg=self.text_light, insertbackground=self.text_light)
        self.search_entry.pack(side=tk.LEFT, padx=5)
        self.search_entry.bind('<KeyRelease>', lambda e: self.refresh_admin_table())

        tk.Button(controls_frame, text="🔍 SEARCH", font=("Segoe UI", 11, "bold"),
                  bg=self.accent_color, fg=self.text_light, width=10, height=1,
                  cursor="hand2", border=0, relief=tk.FLAT, command=self.refresh_admin_table).pack(side=tk.LEFT, padx=5)
        tk.Button(controls_frame, text="🔄 REFRESH", font=("Segoe UI", 11, "bold"),
                  bg=self.success, fg=self.bg_primary, width=12, height=1,
                  cursor="hand2", border=0, relief=tk.FLAT, command=self.refresh_admin_table).pack(side=tk.LEFT, padx=5)
        tk.Button(controls_frame, text="← BACK TO MENU", font=("Segoe UI", 11, "bold"),
                  bg=self.highlight, fg=self.text_light, width=15, height=1,
                  cursor="hand2", border=0, relief=tk.FLAT, command=self.show_main_menu).pack(side=tk.RIGHT, padx=5)
        
        table_frame = tk.Frame(self.root, bg=self.bg_primary)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=30, pady=(0, 30))
        
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Treeview", background=self.bg_secondary, foreground=self.text_light,
                        fieldbackground=self.bg_secondary, rowheight=30, font=("Segoe UI", 10))
        style.configure("Treeview.Heading", background=self.accent_color, foreground=self.text_light,
                        font=("Segoe UI", 11, "bold"))
        style.map("Treeview", background=[("selected", self.highlight)])
        
        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        columns = ("ID", "Last Name", "First Name", "Uniform", "Missing Items", "Time", "Status", "Date")
        self.admin_table = ttk.Treeview(table_frame, columns=columns, show="headings",
                                        yscrollcommand=scrollbar.set, height=20)
        scrollbar.config(command=self.admin_table.yview)
        
        for col in columns:
            self.admin_table.heading(col, text=col)
        self.admin_table.column("ID", width=50, anchor=tk.CENTER)
        self.admin_table.column("Last Name", width=120, anchor=tk.W)
        self.admin_table.column("First Name", width=120, anchor=tk.W)
        self.admin_table.column("Uniform", width=120, anchor=tk.CENTER)
        self.admin_table.column("Missing Items", width=200, anchor=tk.W)
        self.admin_table.column("Time", width=100, anchor=tk.CENTER)
        self.admin_table.column("Status", width=100, anchor=tk.CENTER)
        self.admin_table.column("Date", width=120, anchor=tk.CENTER)
        
        self.admin_table.pack(fill=tk.BOTH, expand=True)
        self.refresh_admin_table()

    def refresh_admin_table(self):
        for item in self.admin_table.get_children():
            self.admin_table.delete(item)
        search_text = self.search_var.get().strip().lower()
        try:
            self.cursor.execute("SELECT * FROM attendance ORDER BY id DESC")
            records = self.cursor.fetchall()
            filtered_records = []
            for record in records:
                last_name = record[1].lower() if record[1] else ""
                first_name = record[2].lower() if record[2] else ""
                if not search_text or search_text in last_name or search_text in first_name:
                    filtered_records.append(record)
            for record in filtered_records:
                self.admin_table.insert("", tk.END, values=record)
            print(f"✓ Admin panel loaded: {len(filtered_records)} records")
        except Exception as e:
            print(f"✗ Error loading records: {e}")
            messagebox.showerror("Error", f"Failed to load attendance records: {e}")

    def on_closing(self):
        self.camera_active = False
        self.detection_running = False
        self.face_recognition_running = False
        if self.cap:
            self.cap.release()
        self.conn.close()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = AttendanceSystem(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()