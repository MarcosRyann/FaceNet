import os
import cv2
import numpy as np
from mtcnn import MTCNN
from facenet_pytorch import InceptionResnetV1
import torch
detector = MTCNN()
facenet = InceptionResnetV1(pretrained='vggface2').eval()
image_folder = "C:\PROJETOSUPER\FaceNet\Teste\dataset2.0"
output_file = "modelfacenet.npz"
embeddings = []
names = []

def save_detections_to_csv(detections, filename='detections.csv'):
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        for detection in detections:
            writer.writerow([detection['name'], detection['confidence'], datetime.now()])
def save_detections_to_txt(detections, filename='detections.txt'):
    with open(filename, mode='a') as file:
        for detection in detections:
            file.write(f"Name: {detection['name']}, Confidence: {detection['confidence']}, Date: {datetime.now()}\n")

   if detections:
                        save_detections_to_csv(detections)
                        save_detections_to_txt(detections)


def process_images():
    for person_name in os.listdir(image_folder):
        person_folder = os.path.join(image_folder, person_name)
        if not os.path.isdir(person_folder):
            continue
        print(f"Processando imagens para: {person_name}")
        for image_name in os.listdir(person_folder):
            image_path = os.path.join(person_folder, image_name)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Erro ao carregar a imagem: {image_path}")   
                continue
            detections = detector.detect_faces(image)
            if not detections:
                print(f"Nenhum rosto detectado em: {image_path}")
                continue
            x, y, width, height = detections[0]['box']
            face_crop = image[y:y + height, x:x + width]
            face_crop = cv2.resize(face_crop, (160, 160))
            face_crop = np.transpose(face_crop, (2, 0, 1)) / 255.0
            face_crop = np.expand_dims(face_crop, axis=0)
            embedding = facenet(torch.tensor(face_crop, dtype=torch.float32)).detach().numpy().flatten()
            embeddings.append(embedding)
            names.append(person_name)
process_images()
print(f"Salvando {len(embeddings)} embeddings conhecidos no arquivo {output_file}")
np.savez(output_file, embeddings=np.array(embeddings), names=np.array(names))
print("Processo concluído!")
