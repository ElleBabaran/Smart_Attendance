# Attendance System Fixes TODO

## Pending Tasks

## Completed Tasks
- [x] Change face detector backend from 'opencv' to 'retinaface' in main.py (face_recognition_worker and recognize_student_with_embeddings)
- [x] Change face detector backend from 'opencv' to 'retinaface' in generate_embeddings.py
- [x] Lower uniform detection confidence from 70% to 50% in main.py (detection_worker)
- [x] Increase minimum size filter for uniform predictions from 30x30 to 50x50 pixels in main.py
- [x] Increase face recognition threshold from 0.7 to 1.0 in main.py
- [x] Regenerate embeddings using updated generate_embeddings.py
- [x] Test the changes for improved accuracy
