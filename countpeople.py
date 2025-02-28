import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

# Load a pre-trained Faster R-CNN model with a ResNet backbone
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()  # Set model to evaluation mode

# Define the transformation to apply to the image before passing it to the model
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert image to tensor
])

# Load the image
image_path = "kittelfjall/hotell/2025_01_31/134156.jpg"
image = Image.open(image_path)

# Apply the transformations to the image
input_tensor = transform(image)
input_batch = input_tensor.unsqueeze(0)  # Add a batch dimension

# If you have a GPU available, move the model and input to the GPU
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
input_batch = input_batch.to(device)

# Perform the inference
with torch.no_grad():
    prediction = model(input_batch)

# Get the detected objects and their labels (class indices)
boxes = prediction[0]['boxes']  # Bounding boxes of detected objects
labels = prediction[0]['labels']  # Labels for the detected objects
scores = prediction[0]['scores']  # Confidence scores for the detections

# Threshold to filter out low-confidence detections (e.g., confidence > 0.5)
threshold = 0.5
person_class_id = 1  # Class ID for "person" in COCO dataset
person_count = 0

# Loop through the detections and count the people
for i, score in enumerate(scores):
    if score > threshold and labels[i] == person_class_id:
        person_count += 1

print(f"Number of people detected: {person_count}")

# Visualize the detections
fig, ax = plt.subplots(1, figsize=(12,9))
ax.imshow(image)

# Loop through the bounding boxes and plot them on the image
for i, box in enumerate(boxes):
    if scores[i] > threshold and labels[i] == person_class_id:
        xmin, ymin, xmax, ymax = box.tolist()
        rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                             linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

plt.show()