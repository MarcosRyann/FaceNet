import cv2
from mtcnn import MTCNN
import numpy as np
import torch
import csv
from facenet_pytorch import InceptionResnetV1
detector = MTCNN()
facenet = InceptionResnetV1(pretrained='vggface2').eval()
def load_known_faces(embedding_file='modelfacenet.npz'):
    try:
        data = np.load(embedding_file)
        return data['embeddings'], data['names']
    except FileNotFoundError:
        print("Arquivo de embeddings conhecido não encontrado.")
        return None, None
known_embeddings, known_names = load_known_faces()

def calculate_similarity(embedding1, embedding2):
    return np.linalg.norm(embedding1 - embedding2)
SIMILARITY_THRESHOLD = 0.8
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Erro - câmera.")
else:
    try:
        while True:
            ret, frame = cap.read()
            if ret:
                detections_mtcnn = detector.detect_faces(frame)
                detections = []
                if detections_mtcnn:
                    for face in detections_mtcnn:
                        x, y, width, height = face['box']
                        face_crop = frame[y:y + height, x:x + width]
                        face_crop = cv2.resize(face_crop, (160, 160))
                        face_crop = np.transpose(face_crop, (2, 0, 1)) / 255.0
                        face_crop = np.expand_dims(face_crop, axis=0)
                        embedding = facenet(torch.tensor(face_crop, dtype=torch.float32)).detach().numpy().flatten()
                        best_match_name = "Desconhecido"
                        best_match_distance = float('inf')
                        if known_embeddings is not None:
                            for known_embedding, name in zip(known_embeddings, known_names):
                                distance = calculate_similarity(embedding, known_embedding)
                                if distance < best_match_distance:
                                    best_match_distance = distance
                                    best_match_name = name
                        if best_match_distance < SIMILARITY_THRESHOLD:
                            print(f"Classe: {best_match_name}, Similaridade: {1 - best_match_distance:.2f}")
                            detections.append({"name": best_match_name, "confidence": 1 - best_match_distance})
                            cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 0), 2)
                            cv2.putText(frame, f"{best_match_name} ({1 - best_match_distance:.2f})",
                                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        else:
                            print("Desconhecido detectado!")
                            detections.append({"name": "Desconhecido", "confidence": 1 - best_match_distance})
                            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 255), 2)
                            cv2.putText(frame, "Desconhecido", (x, y - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                else:
                    print("Nenhum rosto detectado.")
                cv2.imshow('Camera', frame)
                if cv2.waitKey(1) & 0xFF == ord('1'):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()
print("O loop foi encerrado.")
