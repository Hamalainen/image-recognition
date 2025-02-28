import os, shutil
import glob
import requests
import time
import datetime
import cv2
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

now = datetime.datetime.now()

def procedure():
    time.sleep(30)

def skapaMappar():
    path = "kittelfjall/hotell/" + now.strftime("%Y_%m_%d") + "/"
    if not os.path.exists(path):
        os.makedirs(path)

def sparaHotell():
    try:
        cap = cv2.VideoCapture('https://storgrova.moln8.com/hotelliften.mjpg')
        ret, image = cap.read()
        filename = "kittelfjall/hotell/" + now.strftime("%Y_%m_%d") + "/"+ now.strftime("%H%M%S") + ".jpg"
        cv2.imwrite(filename, image)
        return filename
    except:
        print("nagot gick fel.")
def rensaMappar():
    tidAttRensa = (now + datetime.timedelta(days=-2)).strftime("%Y_%m_%d")
    path = "kittelfjall/hotell/" + tidAttRensa + "/"
    if os.path.exists(path):
        shutil.rmtree(path)
        print("rensar mapp " + path)

def raknaPersoner(filename):
    # Load a pre-trained Faster R-CNN model with a ResNet backbone
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()  # Set model to evaluation mode

    # Define the transformation to apply to the image before passing it to the model
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert image to tensor
    ])

    # Load the image
    image_path = filename
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
    threshold = 0.85
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

while True:
    now = datetime.datetime.now()
    skapaMappar()
    print("downloading: " + now.strftime("%Y_%m_%d %H:%M:%S"))
    raknaPersoner(sparaHotell())
    if int(now.strftime("%H")) == 0:   
        rensaMappar()
    procedure()