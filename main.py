import tkinter as tk
from tkinter import messagebox, ttk, filedialog
import cv2
from PIL import Image, ImageTk
import sqlite3
from datetime import datetime
import threading
import time
import csv
from deepface import DeepFace
from ultralytics import YOLO
import os
import pickle
import numpy as np
import winsound
from collections import defaultdict
import xlsxwriter
from queue import Queue

class AttendanceSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Attendance System")
        self.root.geometry("600x500")
        
        # Color scheme
        self.bg_primary = "#1a1a2e"
        self.bg_secondary = "#16213e"
        self.accent_color = "#0f3460"
        self.highlight = "#e94560"
        self.success = "#06ffa5"
        self.warning = "#ffd700"
        self.text_light = "#eaeaea"
        self.root.configure(bg=self.bg_primary)

        # Camera variables
        self.cap = None
        self.camera_active = False
        
        # Detection configuration
        self.UNIFORM_MODEL_PATH = 'runs/detect/uniform_detector2/weights/best.pt'
        self.UNIFORM_CONFIDENCE = 0.25
        self.FACE_EMBEDDINGS_PATH = "face_embeddings.pkl"
        self.FACE_THRESHOLD = 10
        self.RECOGNITION_EVERY_N_FRAMES = 30  # Reduced frequency
        self.UNIFORM_EVERY_N_FRAMES = 3  # Process uniform detection less often
        
        # Detection variables
        self.face_embeddings = {}
        self.face_cascade = None
        self.uniform_model = None
        self.frame_count = 0
        
        # Separate tracking for detection (fast) vs recognition (slow)
        self.current_face_boxes = []
        self.recognized_faces = {}
        self.face_threshold = self.FACE_THRESHOLD
        
        # Attendance tracking
        self.logged_today = set()
        self.last_logged_time = {}
        self.cooldown_seconds = 5
        
        # Performance optimization
        self.recognition_in_progress = False
        self.uniform_detection_in_progress = False
        
        # Uniform accumulation
        self.uniform_accumulator = {}
        self.accumulation_frames = 15
        self.current_uniform_parts = {}  # Cache for display
        
        # Database queue for async operations
        self.db_queue = Queue()
        self.db_thread = threading.Thread(target=self.process_db_queue, daemon=True)
        self.db_thread.start()
        
        # UI update queue - prevent UI flooding
        self.ui_update_queue = Queue(maxsize=1)
        self.last_ui_update = 0
        self.ui_update_interval = 0.1  # Update UI max 10 times per second

        # Database initialization
        self.init_database()

        # Load detection models in background
        threading.Thread(target=self.load_models, daemon=True).start()

        # Show main menu
        self.show_main_menu()

    def init_database(self):
        self.conn = sqlite3.connect('attendance.db', check_same_thread=False)
        self.cursor = self.conn.cursor()
        
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='attendance'")
        table_exists = self.cursor.fetchone()
        
        if table_exists:
            self.cursor.execute("PRAGMA table_info(attendance)")
            columns = [col[1] for col in self.cursor.fetchall()]
            
            if 'full_name' not in columns:
                self.cursor.execute("ALTER TABLE attendance RENAME TO attendance_old")
                
                self.cursor.execute('''CREATE TABLE attendance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, 
                    full_name TEXT NOT NULL,
                    uniform_status TEXT NOT NULL, 
                    missing_items TEXT, 
                    time TEXT NOT NULL,
                    status TEXT NOT NULL, 
                    date TEXT NOT NULL)''')
                
                try:
                    self.cursor.execute('''INSERT INTO attendance (full_name, uniform_status, missing_items, time, status, date)
                                         SELECT name, uniform_status, missing_items, time, status, date FROM attendance_old''')
                    self.cursor.execute("DROP TABLE attendance_old")
                except:
                    self.cursor.execute("DROP TABLE IF EXISTS attendance_old")
        else:
            self.cursor.execute('''CREATE TABLE attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT, 
                full_name TEXT NOT NULL,
                uniform_status TEXT NOT NULL, 
                missing_items TEXT, 
                time TEXT NOT NULL,
                status TEXT NOT NULL, 
                date TEXT NOT NULL)''')
        
        # Add index for faster queries
        self.cursor.execute('''CREATE INDEX IF NOT EXISTS idx_full_name ON attendance(full_name)''')
        self.cursor.execute('''CREATE INDEX IF NOT EXISTS idx_date ON attendance(date)''')
        
        self.conn.commit()

    def process_db_queue(self):
        """Background thread to process database operations asynchronously"""
        while True:
            try:
                operation = self.db_queue.get()
                if operation is None:
                    break
                
                action, data = operation
                
                if action == "INSERT":
                    name, uniform_status, missing_items, time_str, status, date = data
                    conn = sqlite3.connect('attendance.db')
                    cursor = conn.cursor()
                    cursor.execute('''INSERT INTO attendance 
                        (full_name, uniform_status, missing_items, time, status, date)
                        VALUES (?, ?, ?, ?, ?, ?)''',
                        (name, uniform_status, missing_items, time_str, status, date))
                    conn.commit()
                    conn.close()
                    
                    # Schedule UI update
                    self.schedule_ui_update("success", f"✓ Attendance Logged: {name}")
                    
                    # Play sound in background
                    threading.Thread(target=self.play_success_sound, daemon=True).start()
                
                self.db_queue.task_done()
                
            except Exception as e:
                print(f"DB Queue Error: {e}")

    def schedule_ui_update(self, update_type, message):
        """Schedule UI update without blocking"""
        try:
            self.ui_update_queue.put((update_type, message), block=False)
        except:
            pass  # Queue full, skip this update

    def load_models(self):
        """Load face recognition and uniform detection models"""
        try:
            if os.path.exists(self.FACE_EMBEDDINGS_PATH):
                with open(self.FACE_EMBEDDINGS_PATH, 'rb') as f:
                    self.face_embeddings = pickle.load(f)
            
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            
            if os.path.exists(self.UNIFORM_MODEL_PATH):
                self.uniform_model = YOLO(self.UNIFORM_MODEL_PATH)
                
        except Exception as e:
            pass

    def find_face_match(self, face_embedding, threshold):
        """Find best match for face embedding"""
        if not self.face_embeddings:
            return "Unknown"
        
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
    
    def play_success_sound(self):
        """Play success sound"""
        try:
            winsound.Beep(1000, 200)
            time.sleep(0.05)
            winsound.Beep(1200, 200)
        except:
            pass
    
    def is_brown_color(self, frame, x1, y1, x2, y2):
        """Check if the detected region is actually brown colored"""
        try:
            roi = frame[y1:y2, x1:x2]
            
            if roi.size == 0:
                return False
            
            # Downsample for faster processing
            roi_small = cv2.resize(roi, (50, 50))
            hsv = cv2.cvtColor(roi_small, cv2.COLOR_BGR2HSV)
            
            lower_brown1 = np.array([5, 60, 30])
            upper_brown1 = np.array([20, 255, 180])
            lower_brown2 = np.array([0, 50, 25])
            upper_brown2 = np.array([12, 255, 140])
            
            mask1 = cv2.inRange(hsv, lower_brown1, upper_brown1)
            mask2 = cv2.inRange(hsv, lower_brown2, upper_brown2)
            brown_mask = cv2.bitwise_or(mask1, mask2)
            
            brown_pixels = cv2.countNonZero(brown_mask)
            total_pixels = roi_small.shape[0] * roi_small.shape[1]
            brown_percentage = (brown_pixels / total_pixels) * 100
            
            return brown_percentage >= 40
            
        except Exception as e:
            return False
    
    def boxes_overlap(self, box1, box2, threshold=0.5):
        """Check if two bounding boxes overlap significantly"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return False
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        box1_area = w1 * h1
        box2_area = w2 * h2
        
        iou = intersection_area / min(box1_area, box2_area)
        return iou > threshold
    
    def get_face_name_for_box(self, box):
        """Get the recognized name for a face box"""
        for recognized_box, name in self.recognized_faces.items():
            if self.boxes_overlap(box, recognized_box, threshold=0.4):
                return name
        return "Detecting..."
    
    def log_attendance(self, name, uniform_parts):
        """Log attendance to database with cooldown"""
        try:
            current_time = datetime.now()
            current_date = current_time.strftime("%B %d, %Y")
            current_time_str = current_time.strftime("%I:%M %p")
            
            # Quick cooldown check
            if name in self.last_logged_time:
                time_diff = (current_time - self.last_logged_time[name]).total_seconds()
                if time_diff < self.cooldown_seconds:
                    return False
            
            detected_parts = [p.lower() for p in uniform_parts.keys()]
            
            # Determine time status
            hour = current_time.hour
            minute = current_time.minute
            
            if (hour >= 4 and hour < 7) or (hour == 7 and minute <= 15):
                status = "On Time"
            elif hour == 7 and 16 <= minute <= 30:
                status = "Tardy"
            else:
                status = "Late"
            
            # Case 1: No uniform parts detected
            if not detected_parts or len(detected_parts) == 0:
                uniform_status = "Not Detected"
                missing_items = "No uniform parts detected"
                
                self.db_queue.put(("INSERT", (name, uniform_status, missing_items, 
                                              current_time_str, status, current_date)))
                self.last_logged_time[name] = current_time
                return True
            
            # Case 2: Determine gender
            has_brown_dress = any(part in ["brown_dress", "brown dress"] for part in detected_parts)
            has_brown_pants = any(part in ["brown_pants", "brown pants", "brown_pant"] for part in detected_parts)
            has_ribbon = any("ribbon" in part for part in detected_parts)
            has_polo = any("polo" in part for part in detected_parts)
            
            if has_ribbon or has_brown_dress:
                is_female = True
            elif has_polo or has_brown_pants:
                is_female = False
            else:
                return False
            
            # Set required parts based on gender
            if is_female:
                required_parts = {
                    "ribbon": ["ribbon"],
                    "black shoes": ["black_shoes", "shoes", "black shoes"],
                    "school id": ["school_id", "id", "school id"],
                    "dress": ["brown_dress", "brown dress"],
                    "socks": ["socks"],
                    "belt": ["belt"]
                }
            else:
                required_parts = {
                    "white polo": ["polo", "white polo", "white_polo"],
                    "brown pants": ["brown_pants", "brown pants", "brown_pant"],
                    "socks": ["socks"],
                    "black shoes": ["black_shoes", "shoes", "black shoes"],
                    "school id": ["school_id", "id", "school id"]
                }
            
            # Check for missing parts
            missing_parts = []
            for item_name, variations in required_parts.items():
                found = False
                for variation in variations:
                    if any(detected == variation for detected in detected_parts):
                        found = True
                        break
                if not found:
                    missing_parts.append(item_name)
            
            # Set uniform status
            if missing_parts:
                uniform_status = "Incomplete"
                missing_items = ", ".join(missing_parts)
            else:
                uniform_status = "Complete"
                missing_items = "None"
            
            # Add to queue
            self.db_queue.put(("INSERT", (name, uniform_status, missing_items, 
                                          current_time_str, status, current_date)))
            
            self.last_logged_time[name] = current_time
            return True
            
        except Exception as e:
            print(f"Error logging attendance: {e}")
            return False

    def show_main_menu(self):
        for widget in self.root.winfo_children():
            widget.destroy()
        self.root.geometry("600x500")
        
        main_frame = tk.Frame(self.root, bg=self.bg_primary)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=40, pady=40)

        tk.Label(main_frame, text="👤", font=("Arial", 40), 
                 bg=self.bg_secondary, fg=self.success).pack(pady=(0, 20))
        tk.Label(main_frame, text="Smart Attendance", font=("Segoe UI", 36, "bold"),
                 bg=self.bg_primary, fg=self.text_light).pack(pady=(0, 5))
        tk.Label(main_frame, text="Uniform & Face Detection System", font=("Segoe UI", 14),
                 bg=self.bg_primary, fg="#888888").pack(pady=(0, 30))

        button_font = ("Segoe UI", 14, "bold")
        tk.Button(main_frame, text="START DETECTION", font=button_font, 
                  bg=self.success, fg=self.bg_primary,
                  width=25, height=2, cursor="hand2", border=0, relief=tk.FLAT, 
                  command=self.start_camera).pack(pady=12)
        tk.Button(main_frame, text="ADMIN PANEL", font=button_font, 
                  bg=self.accent_color, fg=self.text_light,
                  width=25, height=2, cursor="hand2", border=0, relief=tk.FLAT, 
                  command=self.open_admin).pack(pady=12)
        tk.Label(main_frame, text="Powered by DeepFace & YOLO • v2.6 Optimized", 
                 font=("Segoe UI", 9),
                 bg=self.bg_primary, fg="#666666").pack(side=tk.BOTTOM, pady=20)

    def start_camera(self):
        if self.cap:
            self.cap.release()
            self.cap = None
            cv2.destroyAllWindows()
        
        self.open_camera_window()
        
        def init_camera():
            try:
                time.sleep(0.3)
                
                self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                if not self.cap.isOpened():
                    self.cap = cv2.VideoCapture(0)
                
                if not self.cap.isOpened():
                    self.root.after(0, lambda: messagebox.showerror("Error", 
                        "Failed to open camera. Please check if camera is in use."))
                    return
                
                # Optimized camera settings
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                ret, _ = self.cap.read()
                if not ret:
                    self.cap.release()
                    self.cap = None
                    self.root.after(0, lambda: messagebox.showerror("Error", 
                        "Camera opened but failed to read frame."))
                    return
                
                self.camera_active = True
                self.frame_count = 0
                self.current_face_boxes = []
                self.recognized_faces = {}
                self.uniform_accumulator = {}
                self.current_uniform_parts = {}
                self.root.after(10, self.update_camera)
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to open camera: {e}"))
        
        threading.Thread(target=init_camera, daemon=True).start()

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
        self.status_label = tk.Label(status_card, text="Initializing camera...", 
                                     font=("Segoe UI", 16, "bold"),
                                     bg=self.bg_secondary, fg=self.warning)
        self.status_label.pack(pady=10)
        self.missing_label = tk.Label(status_card, text="Please wait...", 
                                      font=("Segoe UI", 11),
                                      bg=self.bg_secondary, fg="#ff6b6b", wraplength=1000)
        self.missing_label.pack(pady=5)

        camera_frame = tk.Frame(self.root, bg=self.bg_primary)
        camera_frame.pack(pady=10, padx=30, fill=tk.BOTH, expand=True)
        self.camera_label = tk.Label(camera_frame, bg=self.bg_primary, 
                                     text="Starting camera...",
                                     font=("Segoe UI", 14), fg=self.text_light)
        self.camera_label.pack(fill=tk.BOTH, expand=True)

        threshold_frame = tk.Frame(camera_frame, bg=self.bg_primary)
        threshold_frame.place(relx=0.0, rely=1.0, anchor=tk.SW, x=20, y=-20)
        
        tk.Label(threshold_frame, text="Face Threshold:", font=("Segoe UI", 10, "bold"),
                bg=self.bg_primary, fg=self.text_light).pack(side=tk.LEFT, padx=5)
        
        self.threshold_label = tk.Label(threshold_frame, text=str(self.face_threshold), 
                                       font=("Segoe UI", 12, "bold"),
                                       bg=self.bg_secondary, fg=self.success, 
                                       width=4, relief=tk.SOLID, borderwidth=1)
        self.threshold_label.pack(side=tk.LEFT, padx=5)
        
        tk.Button(threshold_frame, text="−", font=("Segoe UI", 12, "bold"),
                 bg=self.highlight, fg=self.text_light, width=3, height=1,
                 cursor="hand2", border=0, relief=tk.FLAT,
                 command=self.decrease_threshold).pack(side=tk.LEFT, padx=2)
        
        tk.Button(threshold_frame, text="+", font=("Segoe UI", 12, "bold"),
                 bg=self.success, fg=self.bg_primary, width=3, height=1,
                 cursor="hand2", border=0, relief=tk.FLAT,
                 command=self.increase_threshold).pack(side=tk.LEFT, padx=2)

        back_button = tk.Button(camera_frame, text="← BACK TO MENU", 
                                font=("Segoe UI", 11, "bold"),
                                bg=self.highlight, fg=self.text_light, 
                                width=18, height=2,
                                cursor="hand2", border=0, relief=tk.FLAT, 
                                command=self.stop_camera)
        back_button.place(relx=1.0, rely=1.0, anchor=tk.SE, x=-20, y=-20)

    def process_face_recognition(self, frame, faces):
        """Process face recognition in background thread"""
        new_recognized_faces = {}
        
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
                    name = self.find_face_match(embedding, self.face_threshold)
                    new_recognized_faces[(x, y, w, h)] = name
            except:
                new_recognized_faces[(x, y, w, h)] = "Detecting..."
        
        self.recognized_faces = new_recognized_faces
        self.recognition_in_progress = False

    def process_uniform_detection(self, frame):
        """Process uniform detection in background thread"""
        uniform_parts = {}
        
        if self.uniform_model:
            uniform_results = self.uniform_model(frame, conf=self.UNIFORM_CONFIDENCE, verbose=False)
            
            for box in uniform_results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                class_name = uniform_results[0].names[cls]
                
                should_include = True
                
                if "dress" in class_name.lower() or "pants" in class_name.lower() or "pant" in class_name.lower():
                    is_brown = self.is_brown_color(frame, x1, y1, x2, y2)
                    if not is_brown:
                        should_include = False
                
                if should_include:
                    if class_name not in uniform_parts:
                        uniform_parts[class_name] = []
                    uniform_parts[class_name].append((x1, y1, x2, y2, float(box.conf[0])))
        
        self.current_uniform_parts = uniform_parts
        self.uniform_detection_in_progress = False

    def update_camera(self):
        if not self.camera_active:
            return
        
        try:
            if not self.cap or not self.cap.isOpened():
                return
            
            ret, frame = self.cap.read()
            if not ret:
                if self.camera_active:
                    self.root.after(10, self.update_camera)
                return
            
            self.frame_count += 1
            
            # Face detection (every frame, but fast)
            if self.face_cascade:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Downscale for faster detection
                scale = 2
                gray_small = cv2.resize(gray, (gray.shape[1]//scale, gray.shape[0]//scale))
                faces_small = self.face_cascade.detectMultiScale(gray_small, 1.3, 5)
                # Scale back up
                faces = [(x*scale, y*scale, w*scale, h*scale) for (x, y, w, h) in faces_small]
                self.current_face_boxes = faces
                
                # Face recognition (much less frequent)
                if self.frame_count % self.RECOGNITION_EVERY_N_FRAMES == 0 and len(faces) > 0 and not self.recognition_in_progress:
                    self.recognition_in_progress = True
                    threading.Thread(target=self.process_face_recognition, 
                                   args=(frame.copy(), faces), daemon=True).start()
            
            # Uniform detection (less frequent)
            if self.frame_count % self.UNIFORM_EVERY_N_FRAMES == 0 and not self.uniform_detection_in_progress:
                self.uniform_detection_in_progress = True
                threading.Thread(target=self.process_uniform_detection, 
                               args=(frame.copy(),), daemon=True).start()
            
            # Draw face boxes (from cached data)
            for box in self.current_face_boxes:
                x, y, w, h = box
                name = self.get_face_name_for_box(box)
                
                color = (255, 0, 0) if name != "Unknown" and name != "Detecting..." else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                label_width = len(name) * 10 + 10
                cv2.rectangle(frame, (x, y-30), (x+label_width, y), color, -1)
                cv2.putText(frame, name, (x+5, y-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Draw uniform boxes (from cached data)
            uniform_counts = {}
            for class_name, boxes in self.current_uniform_parts.items():
                uniform_counts[class_name] = len(boxes)
                for x1, y1, x2, y2, conf in boxes:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{class_name} {conf:.2f}"
                    label_width = len(label) * 8 + 10
                    cv2.rectangle(frame, (x1, y1-25), (x1+label_width, y1), (0, 255, 0), -1)
                    cv2.putText(frame, label, (x1+5, y1-8),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Update status (throttled)
            current_time = time.time()
            if current_time - self.last_ui_update >= self.ui_update_interval:
                self.last_ui_update = current_time
                
                # Process any queued UI updates
                try:
                    while not self.ui_update_queue.empty():
                        update_type, message = self.ui_update_queue.get_nowait()
                        if update_type == "success":
                            self.status_label.config(text=message, fg=self.success)
                except:
                    pass
                
                # Update face status
                num_faces = len(self.current_face_boxes)
                recognized_names = [name for name in self.recognized_faces.values() 
                                  if name != "Unknown" and name != "Detecting..."]
                
                if num_faces > 0:
                    if recognized_names:
                        self.status_label.config(
                            text=f"✓ Recognized: {', '.join(set(recognized_names))}", 
                            fg=self.success
                        )
                    else:
                        self.status_label.config(
                            text="⚠ Face detected but not recognized", 
                            fg=self.warning
                        )
                else:
                    self.status_label.config(
                        text="👁 Scanning for faces...", 
                        fg=self.text_light
                    )
                
                # Update uniform status
                if uniform_counts:
                    parts_text = " | ".join([f"{k}: {v}" for k, v in uniform_counts.items()])
                    self.missing_label.config(text=f"Uniform parts: {parts_text}", fg=self.success)
                else:
                    self.missing_label.config(text="No uniform parts detected", fg=self.warning)
            
            # Accumulate uniform parts for logging
            if recognized_names:
                for face_name in set(recognized_names):
                    if face_name not in self.uniform_accumulator:
                        self.uniform_accumulator[face_name] = {}
                    
                    for part, boxes in self.current_uniform_parts.items():
                        if part not in self.uniform_accumulator[face_name]:
                            self.uniform_accumulator[face_name][part] = 0
                        self.uniform_accumulator[face_name][part] += len(boxes)
            
            # Auto-log (less frequent)
            if self.frame_count % self.accumulation_frames == 0 and recognized_names:
                for face_name in set(recognized_names):
                    current_time = datetime.now()
                    can_log = True
                    
                    if face_name in self.last_logged_time:
                        time_diff = (current_time - self.last_logged_time[face_name]).total_seconds()
                        if time_diff < self.cooldown_seconds:
                            can_log = False
                    
                    if can_log and face_name in self.uniform_accumulator:
                        accumulated_parts = self.uniform_accumulator[face_name]
                        threshold = self.accumulation_frames * 0.3
                        
                        consistent_parts = {}
                        for part, count in accumulated_parts.items():
                            if count >= threshold:
                                consistent_parts[part] = 1
                        
                        threading.Thread(target=self.log_attendance, 
                                       args=(face_name, consistent_parts.copy()), 
                                       daemon=True).start()
                    
                    if face_name in self.uniform_accumulator:
                        self.uniform_accumulator[face_name] = {}

            # Display frame (optimized)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize if needed for display
            display_width = 800
            h, w = frame_rgb.shape[:2]
            if w > display_width:
                scale = display_width / w
                frame_rgb = cv2.resize(frame_rgb, (display_width, int(h * scale)))
            
            img = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
            self.camera_label.config(image=img, text="")
            self.camera_label.image = img
                
        except Exception as e:
            pass
        finally:
            if self.camera_active:
                self.root.after(10, self.update_camera)

    def stop_camera(self):
        self.camera_active = False
        self.root.after(200, self._release_camera_and_return)

    def _release_camera_and_return(self):
        if self.cap:
            self.cap.release()
            self.cap = None
        cv2.destroyAllWindows()
        self.root.after(100, self.show_main_menu)

    def increase_threshold(self):
        self.face_threshold += 1
        self.threshold_label.config(text=str(self.face_threshold))

    def decrease_threshold(self):
        if self.face_threshold > 1:
            self.face_threshold -= 1
            self.threshold_label.config(text=str(self.face_threshold))

    def open_admin(self):
        for widget in self.root.winfo_children():
            widget.destroy()
        self.root.geometry("1200x700")
        
        header_frame = tk.Frame(self.root, bg=self.bg_secondary, height=80)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        tk.Label(header_frame, text="📊  Admin Panel - Attendance Records", 
                 font=("Segoe UI", 22, "bold"),
                 bg=self.bg_secondary, fg=self.text_light).pack(pady=20)
        
        controls_frame = tk.Frame(self.root, bg=self.bg_primary)
        controls_frame.pack(fill=tk.X, padx=30, pady=15)

        self.search_var = tk.StringVar()
        self.search_entry = tk.Entry(controls_frame, textvariable=self.search_var, 
                                     font=("Segoe UI", 11), width=30,
                                     bg=self.bg_secondary, fg=self.text_light, 
                                     insertbackground=self.text_light)
        self.search_entry.pack(side=tk.LEFT, padx=5)
        self.search_entry.bind('<KeyRelease>', lambda e: self.refresh_admin_table())

        tk.Button(controls_frame, text="🔍 SEARCH", font=("Segoe UI", 11, "bold"),
                  bg=self.accent_color, fg=self.text_light, width=10, height=1,
                  cursor="hand2", border=0, relief=tk.FLAT, 
                  command=self.refresh_admin_table).pack(side=tk.LEFT, padx=5)
        tk.Button(controls_frame, text="🔄 REFRESH", font=("Segoe UI", 11, "bold"),
                  bg=self.success, fg=self.bg_primary, width=12, height=1,
                  cursor="hand2", border=0, relief=tk.FLAT, 
                  command=self.refresh_admin_table).pack(side=tk.LEFT, padx=5)
        tk.Button(controls_frame, text="🗑 DELETE", font=("Segoe UI", 11, "bold"),
                  bg=self.highlight, fg=self.text_light, width=12, height=1,
                  cursor="hand2", border=0, relief=tk.FLAT, 
                  command=self.delete_selected_record).pack(side=tk.LEFT, padx=5)
        tk.Button(controls_frame, text="📊 EXPORT EXCEL", font=("Segoe UI", 11, "bold"),
                  bg="#2ecc71", fg="white", width=15, height=1,
                  cursor="hand2", border=0, relief=tk.FLAT, 
                  command=self.export_to_excel).pack(side=tk.LEFT, padx=5)
        tk.Button(controls_frame, text="← BACK TO MENU", font=("Segoe UI", 11, "bold"),
                  bg=self.accent_color, fg=self.text_light, width=15, height=1,
                  cursor="hand2", border=0, relief=tk.FLAT, 
                  command=self.show_main_menu).pack(side=tk.RIGHT, padx=5)
        
        table_frame = tk.Frame(self.root, bg=self.bg_primary)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=30, pady=(0, 15))
        
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Treeview", background=self.bg_secondary, 
                        foreground=self.text_light,
                        fieldbackground=self.bg_secondary, rowheight=30, 
                        font=("Segoe UI", 10))
        style.configure("Treeview.Heading", background=self.accent_color, 
                        foreground=self.text_light,
                        font=("Segoe UI", 11, "bold"))
        style.map("Treeview", background=[("selected", self.highlight)])
        
        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        columns = ("ID", "Full Name", "Uniform", "Missing Items", "Time", "Status", "Date")
        self.admin_table = ttk.Treeview(table_frame, columns=columns, show="headings",
                                        yscrollcommand=scrollbar.set, height=20)
        scrollbar.config(command=self.admin_table.yview)
        
        for col in columns:
            self.admin_table.heading(col, text=col)
        self.admin_table.column("ID", width=50, anchor=tk.CENTER)
        self.admin_table.column("Full Name", width=180, anchor=tk.W)
        self.admin_table.column("Uniform", width=100, anchor=tk.CENTER)
        self.admin_table.column("Missing Items", width=200, anchor=tk.W)
        self.admin_table.column("Time", width=100, anchor=tk.CENTER)
        self.admin_table.column("Status", width=100, anchor=tk.CENTER)
        self.admin_table.column("Date", width=120, anchor=tk.CENTER)
        
        self.admin_table.pack(fill=tk.BOTH, expand=True)
        
        self.refresh_admin_table()

    def delete_selected_record(self):
        selected = self.admin_table.selection()
        if not selected:
            messagebox.showwarning("Warning", "Please select a record to delete")
            return
        
        confirm = messagebox.askyesno("Confirm Delete", 
                                      "Are you sure you want to delete this record?")
        if confirm:
            try:
                item = selected[0]
                values = self.admin_table.item(item, 'values')
                record_id = values[0]
                
                self.cursor.execute("DELETE FROM attendance WHERE id = ?", (record_id,))
                self.conn.commit()
                
                self.admin_table.delete(item)
                messagebox.showinfo("Success", "Record deleted successfully")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to delete record: {e}")

    def export_to_excel(self):
        try:
            self.cursor.execute("SELECT * FROM attendance ORDER BY id DESC")
            records = self.cursor.fetchall()
            
            if not records:
                messagebox.showinfo("Info", "No records to export")
                return
            
            file_path = filedialog.asksaveasfilename(
                defaultextension=".xlsx",
                filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")],
                initialfile=f"attendance_records_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            )
            
            if not file_path:
                return
            
            workbook = xlsxwriter.Workbook(file_path)
            worksheet = workbook.add_worksheet("Attendance Records")
            
            header_format = workbook.add_format({
                'bold': True,
                'bg_color': '#0F3460',
                'font_color': 'white',
                'align': 'center',
                'valign': 'vcenter',
                'border': 1
            })
            
            cell_format = workbook.add_format({
                'align': 'left',
                'valign': 'vcenter',
                'border': 1
            })
            
            headers = ["ID", "Full Name", "Uniform Status", "Missing Items", "Time", "Status", "Date"]
            for col, header in enumerate(headers):
                worksheet.write(0, col, header, header_format)
            
            for row, record in enumerate(records, start=1):
                for col, value in enumerate(record):
                    worksheet.write(row, col, value, cell_format)
            
            worksheet.set_column(0, 0, 8)
            worksheet.set_column(1, 1, 25)
            worksheet.set_column(2, 2, 18)
            worksheet.set_column(3, 3, 35)
            worksheet.set_column(4, 4, 15)
            worksheet.set_column(5, 5, 12)
            worksheet.set_column(6, 6, 20)
            
            workbook.close()
            
            messagebox.showinfo("Success", f"Records exported successfully to:\n{file_path}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export: {e}")

    def refresh_admin_table(self):
        for item in self.admin_table.get_children():
            self.admin_table.delete(item)
        
        search_text = self.search_var.get().strip().lower()
        
        try:
            self.cursor.execute("SELECT * FROM attendance ORDER BY id DESC")
            records = self.cursor.fetchall()
            
            filtered_records = []
            for record in records:
                full_name = record[1].lower() if record[1] else ""
                if not search_text or search_text in full_name:
                    filtered_records.append(record)
            
            for record in filtered_records:
                self.admin_table.insert("", tk.END, values=record)
                
        except:
            pass

    def on_closing(self):
        self.camera_active = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.db_queue.put(None)
        self.conn.close()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = AttendanceSystem(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()