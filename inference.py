import numpy as np
import cv2
import torch
import glob as glob
import os
import time
import argparse
import matplotlib.pyplot as plt

from model import create_model
from config import (
    NUM_CLASSES, DEVICE, CLASSES
)

np.random.seed(42)

# Construct the argument parser.
parser = argparse.ArgumentParser()
parser.add_argument(
    '-i', '--input', 
    help='path to input image directory',
)
parser.add_argument(
    '--imgsz', 
    default=None,
    type=int,
    help='image resize shape'
)
parser.add_argument(
    '--threshold',
    default=0.25,
    type=float,
    help='detection threshold'
)
parser.add_argument(
    '--visualize',
    action='store_true',
    help='Whether to display image using matplotlib'
)
args = vars(parser.parse_args())

os.makedirs('inference_outputs/images', exist_ok=True)

COLORS = [[0, 0, 0], [255, 0, 0]]

# Load the best model and trained weights.
model = create_model(num_classes=NUM_CLASSES, size=640)
checkpoint = torch.load('outputs/best_model.pth', map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE).eval()

# Directory where all the images are present.
DIR_TEST = args['input']
test_images = glob.glob(f"{DIR_TEST}/*.jpg")
print(f"Test instances: {len(test_images)}")

frame_count = 0  # To count total frames.
total_fps = 0    # To get the final frames per second.

for i in range(len(test_images)):
    image_name = test_images[i].split(os.path.sep)[-1].split('.')[0]
    image = cv2.imread(test_images[i])
    orig_image = image.copy()
    
    if args['imgsz'] is not None:
        image = cv2.resize(image, (args['imgsz'], args['imgsz']))

    print(image.shape)

    # BGR to RGB.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    image /= 255.0
    image_input = np.transpose(image, (2, 0, 1)).astype(np.float32)
    image_input = torch.tensor(image_input, dtype=torch.float).to(DEVICE)
    image_input = torch.unsqueeze(image_input, 0)

    start_time = time.time()
    with torch.no_grad():
        outputs = model(image_input)
    end_time = time.time()

    fps = 1 / (end_time - start_time)
    total_fps += fps
    frame_count += 1

    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]

    if len(outputs[0]['boxes']) != 0:
        boxes = outputs[0]['boxes'].data.numpy()
        scores = outputs[0]['scores'].data.numpy()
        boxes = boxes[scores >= args['threshold']].astype(np.int32)
        draw_boxes = boxes.copy()
        pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]
        
        for j, box in enumerate(draw_boxes):
            class_name = pred_classes[j]
            color = COLORS[CLASSES.index(class_name)]
            xmin = int((box[0] / image.shape[1]) * orig_image.shape[1])
            ymin = int((box[1] / image.shape[0]) * orig_image.shape[0])
            xmax = int((box[2] / image.shape[1]) * orig_image.shape[1])
            ymax = int((box[3] / image.shape[0]) * orig_image.shape[0])
            cv2.rectangle(orig_image, (xmin, ymin), (xmax, ymax), color[::-1], 3)
            cv2.putText(orig_image, class_name, (xmin, ymin - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color[::-1], 2,
                        lineType=cv2.LINE_AA)

        output_path = f"inference_outputs/images/{image_name}.jpg"
        cv2.imwrite(output_path, orig_image)

        if args['visualize']:
            display_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(10, 8))
            plt.imshow(display_image)
            plt.axis('off')
            plt.title(f"Prediction: {image_name}")
            plt.show()

    print(f"Image {i+1} done...")
    print('-' * 50)

print('TEST PREDICTIONS COMPLETE')

# Calculate and print the average FPS.
avg_fps = total_fps / frame_count
print(f"Average FPS: {avg_fps:.3f}")
