import os
import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from ultralytics import YOLO
import torch
from torchvision import models, transforms

# Load YOLOv8 model
yolo_model = YOLO("yolov8n.pt")

# Load ResNet model for embeddings
resnet = models.resnet18(pretrained=True)
resnet.fc = torch.nn.Identity()  # remove final classification layer
resnet.eval()

# Transformation for ResNet
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Function to compute embedding
def compute_embedding(image_path):
    img = cv2.imread(image_path)
    results = yolo_model(img)

    if results and results[0].boxes:
        for box in results[0].boxes:
            cls = int(box.cls[0])
            label = yolo_model.names[cls]
            if label in ["car", "truck", "bus", "ambulance", "fire truck"]:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cropped = img[y1:y2, x1:x2]
                if cropped.size == 0:
                    continue
                try:
                    tensor_img = transform(cropped).unsqueeze(0)
                    with torch.no_grad():
                        embedding = resnet(tensor_img).squeeze().numpy()
                    return embedding
                except:
                    continue
    return np.zeros(512)

# Get all embeddings in a directory
def get_embeddings(path):
    embeddings = []
    for filename in sorted(os.listdir(path)):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            emb = compute_embedding(os.path.join(path, filename))
            embeddings.append((filename, emb))
    return embeddings

# Main prioritization logic
def prioritize_images():
    dataset_dir = "dataset"
    priority_dir = "priority_images"

    # Load dataset reference embeddings
    dataset_embeddings = [emb for _, emb in get_embeddings(dataset_dir)]
    if not dataset_embeddings:
        return None, "Dataset folder is empty or invalid."

    dataset_matrix = np.stack(dataset_embeddings)

    # Load priority images
    priority_embeddings = get_embeddings(priority_dir)
    if not priority_embeddings:
        return None, "No images found in priority_images folder."

    batch_size = 5
    results = []

    for i in range(0, len(priority_embeddings), batch_size):
        batch = priority_embeddings[i:i + batch_size]
        batch_scores = []

        for filename, emb in batch:
            if np.count_nonzero(emb) == 0:
                score = 0
            else:
                score = cosine_similarity([emb], dataset_matrix).mean()
            batch_scores.append((filename, score))

        # Find image with highest similarity in this batch
        best_image, best_score = max(batch_scores, key=lambda x: x[1])
        results.append((i // batch_size + 1, best_image, best_score))

    return results, "Batch-wise prioritization complete."

# Run the app
if __name__ == "__main__":
    batch_results, msg = prioritize_images()
    if batch_results:
        for batch_num, image_name, score in batch_results:
            print(f"Batch {batch_num}: Prioritized Image = {image_name}, Similarity Score = {score:.4f}")
    else:
        print("Error:", msg)
