import cv2
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import time
# from torchvision.prototype.models import resnet50, ResNet50_Weight


# Load the pre-trained Faster R-CNN model with a ResNet backbone
# model = resnet50(weights=ResNet50_Weights.ImageNet1k_V1)
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()  # Set model to evaluation mode

# Define the transformation to apply to the image before passing it to the model
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert image to tensor
])

# MJPEG stream URL
stream_url = "https://storgrova.moln8.com/hotelliften.mjpg"

# Open the MJPEG stream
# cap = cv2.VideoCapture(stream_url)

# # Check if the stream was opened successfully
# if not cap.isOpened():
#     print("Error: Unable to open stream.")
#     exit()

# If you have a GPU available, move the model and input to the GPU
device = torch.device("gpu") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
while True:
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print("Error: Unable to open stream.")
        exit()
    # Capture each frame from the MJPEG stream
    ret, frame = cap.read()

    # If no frame is read, break the loop
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Convert the frame to a PIL image for transformation
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Apply the transformations to the image
    input_tensor = transform(pil_image)
    input_batch = input_tensor.unsqueeze(0)  # Add a batch dimension

    # Move the batch to the device (GPU or CPU)
    input_batch = input_batch.to(device)

    # Perform inference
    with torch.no_grad():
        prediction = model(input_batch)

    # Get the detected objects and their labels (class indices)
    boxes = prediction[0]['boxes']  # Bounding boxes of detected objects
    labels = prediction[0]['labels']  # Labels for the detected objects
    scores = prediction[0]['scores']  # Confidence scores for the detections
   
    # Filter out low-confidence detections (e.g., confidence > 0.5)
    threshold = 0.8
    person_class_id = 1  # Class ID for "person" in COCO dataset

    person_count = 0
    # Loop through the detections and count the people
    for i, score in enumerate(scores):
        if score > threshold and labels[i] == person_class_id:
            person_count += 1

    print(f"Number of people detected: {person_count}")
    
    # Loop through the detections and draw bounding boxes for people
    for i, score in enumerate(scores):
        if score > threshold and labels[i] == person_class_id:
            xmin, ymin, xmax, ymax = boxes[i].tolist()

            # Draw the bounding box on the frame
            cv2.rectangle(frame, (int(xmin)-3, int(ymin)-3), (int(xmax)+3, int(ymax)+3), (0, 255, 0), 3)
            # Cropping an image
            # cropped_image = frame[int(xmin):int(ymin), int(xmax):int(ymax)]
            cropped_image = frame[int(ymin):int(ymax), int(xmin):int(xmax)]

            # Display cropped image
            cv2.imwrite("Cropped_Image.jpg", cropped_image)


    # Display the frame with the detected people
    cv2.imshow("MJPEG Stream - People Detection", frame)

    # Exit the loop if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cap.release()
    time.sleep(0.1)
# Release the VideoCapture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()