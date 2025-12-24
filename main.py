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
        self.PROCESS_EVERY_N_FRAMES = 10  # Increased from 5 to 10 for better performance
        
        # Detection variables
        self.face_embeddings = {}
        self.face_cascade = None
        self.uniform_model = None
        self.frame_count = 0
        self.last_face_detections = []
        self.face_threshold = self.FACE_THRESHOLD
        
        # Attendance tracking
        self.logged_today = set()
        self.last_logged_time = {}
        self.cooldown_seconds = 300
        
        # Performance optimization
        self.detection_thread = None
        self.detection_queue = []
        self.processing = False

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
        
        self.conn.commit()

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
        """Play success sound in background thread"""
        def play():
            try:
                winsound.Beep(1000, 200)
                time.sleep(0.05)
                winsound.Beep(1200, 200)
            except:
                pass
        
        threading.Thread(target=play, daemon=True).start()
    
    def log_attendance(self, name, uniform_parts):
        """Log attendance to database with cooldown"""
        try:
            current_time = datetime.now()
            current_date = current_time.strftime("%B %d, %Y")
            current_time_str = current_time.strftime("%I:%M %p")
            
            if name in self.last_logged_time:
                time_diff = (current_time - self.last_logged_time[name]).total_seconds()
                if time_diff < self.cooldown_seconds:
                    return False
            
            detected_parts = [p.lower() for p in uniform_parts.keys()]
            
            female_indicators = ["ribbon", "dress", "brown dress", "belt"]
            male_indicators = ["white polo", "polo", "brown pants", "pants"]
            
            is_female = any(indicator in detected_parts for indicator in female_indicators)
            is_male = any(indicator in detected_parts for indicator in male_indicators)
            
            if is_female:
                required_parts = {
                    "ribbon": ["ribbon"],
                    "black shoes": ["black_shoes", "shoes", "black shoes"],
                    "school id": ["school_id", "id", "school id"],
                    "dress": ["dress", "brown dress", "brown_dress"],
                    "socks": ["socks"],
                    "belt": ["belt"]
                }
            elif is_male:
                required_parts = {
                    "white polo": ["polo", "white polo", "white_polo"],
                    "brown pants": ["pants", "brown pants", "brown_pants"],
                    "socks": ["socks"],
                    "black shoes": ["black_shoes", "shoes", "black shoes"],
                    "school id": ["school_id", "id", "school id"]
                }
            else:
                return False
            
            missing_parts = []
            for item_name, variations in required_parts.items():
                found = False
                for variation in variations:
                    if any(variation in detected or detected in variation for detected in detected_parts):
                        found = True
                        break
                if not found:
                    missing_parts.append(item_name)
            
            if missing_parts:
                uniform_status = "Incomplete"
                missing_items = ", ".join(missing_parts)
            else:
                uniform_status = "Complete"
                missing_items = "None"
            
            hour = current_time.hour
            minute = current_time.minute
            
            if hour < 7 or (hour == 7 and minute <= 15):
                status = "On Time"
            elif hour == 7 and 16 <= minute <= 30:
                status = "Tardy"
            else:
                status = "Late"
            
            self.cursor.execute('''INSERT INTO attendance 
                (full_name, uniform_status, missing_items, time, status, date)
                VALUES (?, ?, ?, ?, ?, ?)''',
                (name, uniform_status, missing_items, current_time_str, status, current_date))
            self.conn.commit()
            
            self.last_logged_time[name] = current_time
            self.play_success_sound()
            
            return True
            
        except:
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
        tk.Label(main_frame, text="Powered by DeepFace & YOLO • v2.1", 
                 font=("Segoe UI", 9),
                 bg=self.bg_primary, fg="#666666").pack(side=tk.BOTTOM, pady=20)

    def start_camera(self):
        self.open_camera_window()
        
        def init_camera():
            try:
                # Use DSHOW backend for faster camera init on Windows
                self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                if not self.cap.isOpened():
                    self.cap = cv2.VideoCapture(0)  # Fallback
                
                # Optimize camera settings
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for less lag
                
                self.camera_active = True
                self.frame_count = 0
                self.last_face_detections = []
                self.root.after(10, self.update_camera)  # Faster update rate
            except Exception as e:
                self.root.after(100, lambda: messagebox.showerror("Error", f"Failed to open camera: {e}"))
        
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
                    name = self.find_face_match(embedding, self.face_threshold)
                    current_face_detections.append((x, y, w, h, name))
            except:
                current_face_detections.append((x, y, w, h, "Detecting..."))
        
        return current_face_detections

    def update_camera(self):
        try:
            if not self.camera_active or not self.cap or not self.cap.isOpened():
                return
            
            ret, frame = self.cap.read()
            if not ret:
                self.root.after(10, self.update_camera)
                return
            
            self.frame_count += 1
            
            # Face recognition (less frequent for performance)
            if self.frame_count % self.PROCESS_EVERY_N_FRAMES == 0 and self.face_cascade and not self.processing:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                
                if len(faces) > 0:
                    self.processing = True
                    # Process in background thread
                    def process():
                        detections = self.process_face_recognition(frame.copy(), faces)
                        self.last_face_detections = detections
                        self.processing = False
                    
                    threading.Thread(target=process, daemon=True).start()
            
            # Draw face boxes (fast operation)
            for detection in self.last_face_detections:
                x, y, w, h, name = detection
                color = (255, 0, 0) if name != "Unknown" and name != "Detecting..." else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                label_width = len(name) * 10 + 10
                cv2.rectangle(frame, (x, y-30), (x+label_width, y), color, -1)
                cv2.putText(frame, name, (x+5, y-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Uniform detection (every frame for smooth boxes)
            uniform_parts = {}
            if self.uniform_model:
                uniform_results = self.uniform_model(frame, conf=self.UNIFORM_CONFIDENCE, verbose=False)
                
                for box in uniform_results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    class_name = uniform_results[0].names[cls]
                    
                    if class_name not in uniform_parts:
                        uniform_parts[class_name] = 0
                    uniform_parts[class_name] += 1
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    label = f"{class_name} {conf:.2f}"
                    label_width = len(label) * 8 + 10
                    cv2.rectangle(frame, (x1, y1-25), (x1+label_width, y1), (0, 255, 0), -1)
                    cv2.putText(frame, label, (x1+5, y1-8),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Update status
            num_faces = len(self.last_face_detections)
            recognized_faces = [d[4] for d in self.last_face_detections if d[4] != "Unknown" and d[4] != "Detecting..."]
            
            # Auto-log (in background to avoid blocking)
            if recognized_faces and uniform_parts:
                for face_name in recognized_faces:
                    def log_async(name):
                        logged = self.log_attendance(name, uniform_parts)
                        if logged:
                            self.root.after(0, lambda n=name: self.status_label.config(
                                text=f"✓ Attendance Logged: {n}", 
                                fg=self.success
                            ))
                    threading.Thread(target=lambda: log_async(face_name), daemon=True).start()
            
            if num_faces > 0:
                if recognized_faces:
                    self.status_label.config(
                        text=f"✓ Recognized: {', '.join(recognized_faces)}", 
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
            
            if uniform_parts:
                parts_text = " | ".join([f"{k}: {v}" for k, v in uniform_parts.items()])
                self.missing_label.config(text=f"Uniform parts: {parts_text}", fg=self.success)
            else:
                self.missing_label.config(text="No uniform parts detected", fg=self.warning)

            # Display frame (optimized)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
            self.camera_label.config(image=img, text="")
            self.camera_label.image = img
                
        except Exception as e:
            pass
        finally:
            self.root.after(10, self.update_camera)

    def stop_camera(self):
        self.camera_active = False
        time.sleep(0.1)
        if self.cap:
            self.cap.release()
            self.cap = None
        self.show_main_menu()

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
            messagebox.showerror("Error", f"Failed to export: {e}\n\nMake sure xlsxwriter is installed:\npip install xlsxwriter")

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
        self.conn.close()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = AttendanceSystem(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()