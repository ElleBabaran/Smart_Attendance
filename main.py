import tkinter as tk
from tkinter import font, messagebox, ttk
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
import winsound  # For Windows sound

class AttendanceSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Attendance System")
        self.root.geometry("600x500")
        
        # Colors
        self.bg_primary = "#1a1a2e"
        self.bg_secondary = "#16213e"
        self.accent_color = "#0f3460"
        self.highlight = "#e94560"
        self.success = "#06ffa5"
        self.warning = "#ffd700"
        self.text_light = "#eaeaea"
        self.root.configure(bg=self.bg_primary)

        # Camera & model
        self.cap = None
        self.camera_active = False
        self.model = None

        # Uniform detection
        self.female_items = ['brown dress', 'socks', 'black shoes', 'belt', 'id', 'ribbon']
        self.male_items = ['white polo', 'id', 'socks', 'black shoes', 'brown pants']
        self.current_gender = None
        self.last_predictions = []
        self.last_save_time = 0

        # Thread queues
        self.frame_queue = queue.Queue(maxsize=1)
        self.result_queue = queue.Queue(maxsize=1)
        self.face_queue = queue.Queue(maxsize=1)
        self.face_result_queue = queue.Queue(maxsize=1)
        self.detection_running = False
        self.face_recognition_running = False

        # Face recognition results
        self.current_last_name = "Unknown"
        self.current_first_name = ""
        self.face_box = None

        # Detection state
        self.detection_complete = False
        self.sound_played = False

        # Bounding box colors
        self.CLASS_COLORS = {
            "white polo": (0, 255, 0),
            "brown pants": (0, 0, 255),
            "brown dress": (255, 0, 0),
            "ribbon": (255, 165, 0),
            "socks": (255, 255, 0),
            "black shoes": (128, 0, 128),
            "id": (0, 255, 255),
            "belt": (255, 20, 147),
        }

        # Directory for student images
        self.students_dir = "students"

        # Initialize database
        self.init_database()

        # Initialize Roboflow
        API_KEY = "2ypTreeDqECVWLmZgvXe"
        WORKSPACE_NAME = "elle-5vs9o"
        PROJECT_NAME = "uniformmm-ubp4u"
        VERSION = 5
        try:
            rf = Roboflow(api_key=API_KEY)
            workspace = rf.workspace(WORKSPACE_NAME)
            project = workspace.project(PROJECT_NAME)
            self.model = project.version(VERSION).model
            print("✓ Model loaded!")
        except Exception as e:
            print(f"✗ Model error: {e}")
            self.model = None

        self.show_main_menu()

    # --- Database ---
    def init_database(self):
        """Initialize SQLite database with attendance table"""
        self.conn = sqlite3.connect('attendance.db', check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                last_name TEXT NOT NULL,
                first_name TEXT,
                uniform_status TEXT NOT NULL,
                missing_items TEXT,
                time TEXT NOT NULL,
                status TEXT NOT NULL,
                date TEXT NOT NULL
            )
        ''')
        self.conn.commit()
        print("✓ Database initialized")

    def save_to_database(self, last_name, first_name, uniform_status, missing_items):
        """
        Save attendance record to database
        
        Database columns match admin panel:
        - Last Name: "Apostol" (from folder "Apostol, John")
        - First Name: "John" (from folder "Apostol, John")
        - Uniform Status: "Complete" or "Incomplete"
        - Missing Items: "white polo, id" (male) OR "brown dress, ribbon" (female) OR "None" if complete
        - Time: "02:30 PM" (actual detection time)
        - Status: "Late" (after 7:01 AM) or "On Time"
        - Date: "2024-12-09" (today's date)
        """
        current_time = datetime.now()
        date_str = current_time.strftime("%Y-%m-%d")
        time_str = current_time.strftime("%I:%M %p")
        
        # Late if after 7:01 AM
        cutoff_time = current_time.replace(hour=7, minute=1, second=0, microsecond=0)
        status = "Late" if current_time > cutoff_time else "On Time"
        
        # Format missing items: "item1, item2" or "None"
        missing_str = ", ".join(missing_items) if missing_items else "None"
        
        try:
            self.cursor.execute('''
                INSERT INTO attendance (last_name, first_name, uniform_status, missing_items, time, status, date)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (last_name, first_name, uniform_status, missing_str, time_str, status, date_str))
            self.conn.commit()
            
            print(f"✓ SAVED TO DATABASE:")
            print(f"  │ Last Name: {last_name}")
            print(f"  │ First Name: {first_name}")
            print(f"  │ Uniform: {uniform_status}")
            print(f"  │ Missing: {missing_str}")
            print(f"  │ Time: {time_str}")
            print(f"  │ Status: {status}")
            print(f"  │ Date: {date_str}")
            print(f"  └─────────────────────")
            
            return True
        except Exception as e:
            print(f"✗ Database save error: {e}")
            return False

    def play_success_sound(self):
        """Play a beep sound when detection is complete"""
        try:
            winsound.Beep(1000, 200)  # 1000 Hz for 200ms
            time.sleep(0.1)
            winsound.Beep(1200, 200)  # 1200 Hz for 200ms
        except:
            print("⚠ Sound not available on this system")

    # --- GUI ---
    def show_main_menu(self):
        for widget in self.root.winfo_children():
            widget.destroy()
        
        self.root.geometry("600x500")
        main_frame = tk.Frame(self.root, bg=self.bg_primary)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=40, pady=40)

        tk.Label(main_frame, text="👤", font=("Arial", 40),
                 bg=self.bg_secondary, fg=self.success).pack(pady=(0, 20))
        tk.Label(main_frame, text="Smart Attendance",
                 font=("Segoe UI", 36, "bold"),
                 bg=self.bg_primary, fg=self.text_light).pack(pady=(0, 5))
        tk.Label(main_frame, text="Uniform & Face Detection System",
                 font=("Segoe UI", 14),
                 bg=self.bg_primary, fg="#888888").pack(pady=(0, 50))

        button_font = ("Segoe UI", 14, "bold")
        tk.Button(main_frame, text="START DETECTION", font=button_font,
                  bg=self.success, fg=self.bg_primary, width=25, height=2,
                  cursor="hand2", border=0, relief=tk.FLAT,
                  command=self.start_camera).pack(pady=12)

        tk.Button(main_frame, text="ADMIN PANEL", font=button_font,
                  bg=self.accent_color, fg=self.text_light, width=25, height=2,
                  cursor="hand2", border=0, relief=tk.FLAT,
                  command=self.open_admin).pack(pady=12)

        tk.Label(main_frame, text="Powered by Roboflow & DeepFace • v1.0",
                 font=("Segoe UI", 9),
                 bg=self.bg_primary, fg="#666666").pack(side=tk.BOTTOM, pady=20)

    # --- Camera & Detection ---
    def start_camera(self):
        if self.model is None:
            messagebox.showerror("Error", "Roboflow model not loaded!")
            return
        
        self.open_camera_window()
        
        def init_camera():
            try:
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    raise Exception("Camera not accessible")
                
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                time.sleep(0.5)
                
                # Reset all state
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
                
                # Clear all queues
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
                
                # Start worker threads
                threading.Thread(target=self.detection_worker, daemon=True).start()
                threading.Thread(target=self.face_recognition_worker, daemon=True).start()
                
                print("✓ Camera initialized, starting detection...")
                self.root.after(30, self.update_camera)
            except Exception as e:
                print(f"✗ Camera error: {e}")
                self.root.after(100, lambda: messagebox.showerror("Error", f"Failed to open camera: {e}"))
        
        threading.Thread(target=init_camera, daemon=True).start()

    def detection_worker(self):
        """Separate thread for uniform detection"""
        while self.detection_running:
            try:
                frame = self.frame_queue.get(timeout=1.0)
                temp_file = "temp_frame.jpg"
                cv2.imwrite(temp_file, frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                
                # Run Roboflow prediction
                result = self.model.predict(temp_file, confidence=40, overlap=30)
                predictions = result.json()
                
                try: 
                    os.remove(temp_file)
                except: 
                    pass
                
                # Send predictions to main thread
                if 'predictions' in predictions:
                    pred_list = predictions['predictions']
                    print(f"🔍 Detected {len(pred_list)} items: {[p['class'] for p in pred_list]}")
                    try:
                        if self.result_queue.empty():
                            self.result_queue.put_nowait(pred_list)
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
        """Separate thread for face recognition"""
        while self.face_recognition_running:
            try:
                frame = self.face_queue.get(timeout=1.0)
                try:
                    # Extract faces using RetinaFace
                    faces = DeepFace.extract_faces(frame, detector_backend='retinaface', 
                                                   enforce_detection=False, align=True)
                    
                    if faces and len(faces) > 0:
                        face_data = faces[0]
                        x = face_data['facial_area']['x']
                        y = face_data['facial_area']['y']
                        w = face_data['facial_area']['w']
                        h = face_data['facial_area']['h']
                        
                        # Extract face with padding
                        padding = 20
                        y1 = max(0, y - padding)
                        y2 = min(frame.shape[0], y + h + padding)
                        x1 = max(0, x - padding)
                        x2 = min(frame.shape[1], x + w + padding)
                        face_crop = frame[y1:y2, x1:x2]
                        
                        # Recognize student
                        last_name, first_name = self.recognize_student(face_crop)
                        
                        # Send results
                        result = {
                            'last_name': last_name,
                            'first_name': first_name,
                            'box': (x, y, w, h)
                        }
                        try:
                            if self.face_result_queue.empty():
                                self.face_result_queue.put_nowait(result)
                        except queue.Full:
                            pass
                except Exception as e:
                    print(f"⚠ Face extraction error: {e}")
            except queue.Empty:
                continue
            except Exception as e:
                print(f"✗ Face recognition error: {e}")
                time.sleep(0.5)

    def open_camera_window(self):
        for widget in self.root.winfo_children():
            widget.destroy()
        self.root.geometry("1100x750")

        header_frame = tk.Frame(self.root, bg=self.bg_secondary, height=80)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        tk.Label(header_frame, text="🎥  Live Detection",
                 font=("Segoe UI", 22, "bold"),
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
                                     text="Starting camera...", font=("Segoe UI", 14),
                                     fg=self.text_light)
        self.camera_label.pack(fill=tk.BOTH, expand=True)

        back_button = tk.Button(camera_frame, text="← BACK TO MENU",
                                font=("Segoe UI", 11, "bold"),
                                bg=self.highlight, fg=self.text_light, width=18, height=2,
                                cursor="hand2", border=0, relief=tk.FLAT,
                                command=self.stop_camera)
        back_button.place(relx=1.0, rely=1.0, anchor=tk.SE, x=-20, y=-20)

    # --- Uniform check ---
    def check_uniform_completeness(self, predictions):
        """
        Check uniform completeness and determine gender
        
        Logic:
        - If detects "white polo" or "brown pants" → MALE uniform
          Required: white polo, brown pants, id, socks, black shoes
        - If detects "brown dress" or "ribbon" → FEMALE uniform
          Required: brown dress, ribbon, belt, id, socks, black shoes
        
        Returns:
            tuple: (is_complete, missing_items, detected_items)
            - is_complete: True if all items present
            - missing_items: List of missing items (e.g., ["white polo", "id"])
            - detected_items: Set of detected items
        """
        detected_items = set(p['class'].lower().strip() for p in predictions)
        
        # Determine gender based on detected items
        if 'brown dress' in detected_items or 'ribbon' in detected_items:
            self.current_gender = "Female"
            required_items = self.female_items
            print(f"👗 Detected FEMALE uniform")
        elif 'white polo' in detected_items or 'brown pants' in detected_items:
            self.current_gender = "Male"
            required_items = self.male_items
            print(f"👔 Detected MALE uniform")
        else:
            # If unclear, check which has fewer missing items
            female_missing = sum(1 for item in self.female_items if item.lower() not in detected_items)
            male_missing = sum(1 for item in self.male_items if item.lower() not in detected_items)
            self.current_gender = "Female" if female_missing <= male_missing else "Male"
            required_items = self.female_items if self.current_gender == "Female" else self.male_items
            print(f"❓ Unclear - assuming {self.current_gender} uniform")
        
        # Find missing items from required list
        missing_items = [item for item in required_items if item.lower() not in detected_items]
        
        is_complete = len(missing_items) == 0
        
        if is_complete:
            print(f"✅ COMPLETE uniform - all items present")
        else:
            print(f"⚠️ INCOMPLETE uniform - missing: {', '.join(missing_items)}")
        
        return is_complete, missing_items, detected_items

    # --- Face recognition ---
    def recognize_student(self, face_crop):
        """
        Recognize student from face crop
        
        Args:
            face_crop: Cropped face image
            
        Returns:
            tuple: (last_name, first_name)
        """
        if not os.path.exists(self.students_dir):
            print(f"⚠ Students directory not found: {self.students_dir}")
            return "Unknown", ""
        
        for student_folder in os.listdir(self.students_dir):
            student_path = os.path.join(self.students_dir, student_folder)
            if not os.path.isdir(student_path):
                continue
            
            # Parse folder name: "LastName, FirstName"
            if ',' in student_folder:
                parts = student_folder.split(',')
                last_name = parts[0].strip()
                first_name = parts[1].strip() if len(parts) > 1 else ""
            else:
                last_name = student_folder.strip()
                first_name = ""
            
            # Check all images in student folder
            for img_file in os.listdir(student_path):
                if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                
                img_path = os.path.join(student_path, img_file)
                try:
                    result = DeepFace.verify(face_crop, img_path, 
                                           model_name='Facenet',
                                           enforce_detection=False,
                                           detector_backend='retinaface')
                    
                    if result['verified']:
                        print(f"✓ Face recognized: {last_name}, {first_name}")
                        return last_name, first_name
                except Exception as e:
                    continue
        
        print("⚠ Face not recognized - saving as Unknown")
        return "Unknown", ""

    # --- Update loop ---
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

            # Send frame for uniform detection every 10 frames
            if self.frame_count % 10 == 0:
                try:
                    if self.frame_queue.empty():
                        self.frame_queue.put_nowait(frame.copy())
                except queue.Full:
                    pass

            # Send frame for face recognition every 30 frames
            if self.frame_count % 30 == 0:
                try:
                    if self.face_queue.empty():
                        self.face_queue.put_nowait(frame.copy())
                except queue.Full:
                    pass

            # Get uniform detection results
            try:
                new_predictions = self.result_queue.get_nowait()
                self.last_predictions = new_predictions
            except queue.Empty:
                pass

            # Get face recognition results
            try:
                face_result = self.face_result_queue.get_nowait()
                self.current_last_name = face_result['last_name']
                self.current_first_name = face_result['first_name']
                self.face_box = face_result['box']
            except queue.Empty:
                pass

            is_complete = False
            missing = []

            # Draw uniform bounding boxes
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
                    cv2.putText(display_frame, label, (x+3, y-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # Draw face bounding box with name label
            if self.face_box:
                x, y, w, h = self.face_box
                # Green box for face
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                
                # Display name (from folder or "Unknown")
                if self.current_last_name and self.current_last_name != "Unknown":
                    name_text = f"{self.current_last_name}, {self.current_first_name}"
                    box_color = (0, 255, 0)  # Green for recognized
                else:
                    name_text = "Unknown"
                    box_color = (0, 0, 255)  # Red for unknown
                
                # Name label background
                text_size = cv2.getTextSize(name_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.rectangle(display_frame, (x, y-30), (x + text_size[0] + 10, y), box_color, -1)
                cv2.putText(display_frame, name_text, (x+5, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Update status labels
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

            # Save to database every 5 seconds - ALWAYS SAVE when ANY detection occurs
            if (current_time - self.last_save_time >= 5.0):
                # Save if uniform detected OR face detected
                if self.last_predictions or self.face_box:
                    self.last_save_time = current_time
                    
                    # Determine uniform status
                    if self.last_predictions:
                        uniform_status = "Complete" if is_complete else "Incomplete"
                    else:
                        uniform_status = "Not Detected"
                        missing = []
                    
                    # ALWAYS use current detected name (Unknown if not recognized)
                    save_last_name = self.current_last_name if self.current_last_name else "Unknown"
                    save_first_name = self.current_first_name if self.current_first_name else ""
                    
                    print(f"\n📝 SAVING TO DATABASE...")
                    print(f"   Name: {save_last_name}, {save_first_name}")
                    print(f"   Uniform: {uniform_status}")
                    print(f"   Missing: {missing if missing else 'None'}")
                    
                    success = self.save_to_database(
                        save_last_name, 
                        save_first_name, 
                        uniform_status, 
                        missing
                    )
                    
                    # Play sound and show notification
                    if success and not self.detection_complete:
                        self.detection_complete = True
                        if not self.sound_played:
                            threading.Thread(target=self.play_success_sound, daemon=True).start()
                            self.sound_played = True
                        # Reset after 3 seconds for next student
                        threading.Timer(3.0, self.reset_detection_state).start()

            # Display frame
            try:
                frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                frame_rgb = cv2.resize(frame_rgb, (640, 480))
                img = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
                self.camera_label.config(image=img, text="")
                self.camera_label.image = img
            except Exception as e:
                print(f"✗ Display error: {e}")
        except Exception as e:
            print(f"✗ Update camera error: {e}")
        finally:
            self.root.after(30, self.update_camera)

    def reset_detection_state(self):
        """Reset detection state for next student"""
        self.detection_complete = False
        self.sound_played = False
        self.current_last_name = "Unknown"
        self.current_first_name = ""
        self.face_box = None
        print("\n🔄 Ready for next student...\n")

    # --- Stop camera ---
    def stop_camera(self):
        self.camera_active = False
        self.detection_running = False
        self.face_recognition_running = False
        time.sleep(0.3)
        if self.cap:
            self.cap.release()
            self.cap = None
        self.show_main_menu()

    # --- Admin panel ---
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

        tk.Button(controls_frame, text="🔍 SEARCH",
                  font=("Segoe UI", 11, "bold"),
                  bg=self.accent_color, fg=self.text_light, width=10, height=1,
                  cursor="hand2", border=0, relief=tk.FLAT,
                  command=self.refresh_admin_table).pack(side=tk.LEFT, padx=5)

        tk.Button(controls_frame, text="🔄 REFRESH",
                  font=("Segoe UI", 11, "bold"),
                  bg=self.success, fg=self.bg_primary, width=12, height=1,
                  cursor="hand2", border=0, relief=tk.FLAT,
                  command=self.refresh_admin_table).pack(side=tk.LEFT, padx=5)

        tk.Button(controls_frame, text="← BACK TO MENU",
                  font=("Segoe UI", 11, "bold"),
                  bg=self.highlight, fg=self.text_light, width=15, height=1,
                  cursor="hand2", border=0, relief=tk.FLAT,
                  command=self.show_main_menu).pack(side=tk.RIGHT, padx=5)
        
        table_frame = tk.Frame(self.root, bg=self.bg_primary)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=30, pady=(0, 30))
        
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Treeview",
                        background=self.bg_secondary,
                        foreground=self.text_light,
                        fieldbackground=self.bg_secondary,
                        rowheight=30,
                        font=("Segoe UI", 10))
        style.configure("Treeview.Heading",
                        background=self.accent_color,
                        foreground=self.text_light,
                        font=("Segoe UI", 11, "bold"))
        style.map("Treeview", background=[("selected", self.highlight)])
        
        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        columns = ("ID", "Last Name", "First Name", "Uniform", "Missing Items", "Time", "Status", "Date")
        self.admin_table = ttk.Treeview(table_frame, columns=columns, show="headings",
                                        yscrollcommand=scrollbar.set, height=20)
        scrollbar.config(command=self.admin_table.yview)
        
        self.admin_table.heading("ID", text="ID")
        self.admin_table.heading("Last Name", text="Last Name")
        self.admin_table.heading("First Name", text="First Name")
        self.admin_table.heading("Uniform", text="Uniform Status")
        self.admin_table.heading("Missing Items", text="Missing Items")
        self.admin_table.heading("Time", text="Time")
        self.admin_table.heading("Status", text="Status")
        self.admin_table.heading("Date", text="Date")
        
        self.admin_table.column("ID", width=50, anchor=tk.CENTER)
        self.admin_table.column("Last Name", width=120, anchor=tk.W)
        self.admin_table.column("First Name", width=120, anchor=tk.W)
        self.admin_table.column("Uniform", width=120, anchor=tk.CENTER)
        self.admin_table.column("Missing Items", width=200, anchor=tk.W)
        self.admin_table.column("Time", width=100, anchor=tk.CENTER)
        self.admin_table.column("Status", width=100, anchor=tk.CENTER)
        self.admin_table.column("Date", width=120, anchor=tk.CENTER)
        
        self.admin_table.pack(fill=tk.BOTH, expand=True)
        
        # Load data
        self.refresh_admin_table()

    def refresh_admin_table(self):
        """Refresh admin table with attendance records"""
        # Clear existing data
        for item in self.admin_table.get_children():
            self.admin_table.delete(item)

        # Get search text
        search_text = self.search_var.get().strip().lower()

        # Fetch from database
        try:
            self.cursor.execute("SELECT * FROM attendance ORDER BY id DESC")
            records = self.cursor.fetchall()

            filtered_records = []
            for record in records:
                # record format: (id, last_name, first_name, uniform_status, missing_items, time, status, date)
                last_name = record[1].lower() if record[1] else ""
                first_name = record[2].lower() if record[2] else ""
                
                if not search_text or search_text in last_name or search_text in first_name:
                    filtered_records.append(record)

            for record in filtered_records:
                self.admin_table.insert("", tk.END, values=record)

            print(f"✓ Admin panel loaded: {len(filtered_records)} records (filtered from {len(records)} total)")
        except Exception as e:
            print(f"✗ Error loading records: {e}")
            messagebox.showerror("Error", f"Failed to load attendance records: {e}")

    # --- Exit cleanup ---
    def on_closing(self):
        """Clean up resources on exit"""
        self.camera_active = False
        self.detection_running = False
        self.face_recognition_running = False
        if self.cap:
            self.cap.release()
        self.conn.close()
        self.root.destroy()


# --- Run App ---
if __name__ == "__main__":
    root = tk.Tk()
    app = AttendanceSystem(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()