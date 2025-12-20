from deepface import DeepFace
import pickle
import os

students_dir = "students"
print("=" * 60)
print("GENERATING FACE EMBEDDINGS")
print("=" * 60)
print("Looking for students folder at: " + students_dir)

if not os.path.exists(students_dir):
    print("ERROR: '" + students_dir + "' folder not found!")
    exit()

print("Found folders: " + str(os.listdir(students_dir)))
print()

embeddings_data = {}

for student_folder in os.listdir(students_dir):
    folder_path = os.path.join(students_dir, student_folder)
    if not os.path.isdir(folder_path):
        continue
    
    if ',' in student_folder:
        parts = student_folder.split(',')
        last_name = parts[0].strip()
        first_name = parts[1].strip()
    else:
        last_name = student_folder.strip()
        first_name = ""
    
    student_name = last_name + ", " + first_name
    embeddings_data[student_name] = []
    
    print("Processing: " + student_name)
    
    image_count = 0
    for img_file in os.listdir(folder_path):
        if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        
        img_path = os.path.join(folder_path, img_file)
        try:
            embedding = DeepFace.represent(
                img_path,
                model_name='Facenet',
                detector_backend='retinaface',
                enforce_detection=False
            )
            embeddings_data[student_name].append(embedding)
            image_count += 1
            print("   Generated embedding for " + img_file)
        except Exception as e:
            print("   Could not process " + img_file + ": " + str(e))
    
    print("   Total embeddings for " + student_name + ": " + str(image_count))
    print()

with open("embeddings.pkl", "wb") as f:
    pickle.dump(embeddings_data, f)

print("=" * 60)
print("SUCCESS! Saved embeddings for " + str(len(embeddings_data)) + " students")
print("File: embeddings.pkl")
print("=" * 60)
print()
print("Now run: python main.py")