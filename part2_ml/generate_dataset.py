import os
import cv2
import numpy as np

BASE = "data/dataset"
classes = ["fragile", "heavy", "hazardous"]

os.makedirs(BASE, exist_ok=True)

for c in classes:
    os.makedirs(f"{BASE}/{c}", exist_ok=True)

def make_image(label, idx):
    img = np.ones((224,224,3), dtype=np.uint8) * 255

    # draw random rectangle (box simulation)
    x1 = np.random.randint(20,120)
    y1 = np.random.randint(20,120)
    x2 = x1 + np.random.randint(60,150)
    y2 = y1 + np.random.randint(60,150)

    color = {
        "fragile": (0,0,255),      # red
        "heavy": (255,0,0),        # blue
        "hazardous": (0,255,255)   # yellow
    }[label]

    cv2.rectangle(img, (x1,y1), (x2,y2), color, -1)

    cv2.putText(img, label.upper(),
                (20,200),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0,0,0), 2)

    return img

for label in classes:
    for i in range(40):
        img = make_image(label, i)
        cv2.imwrite(f"{BASE}/{label}/{label}_{i}.png", img)

print("Dataset generated in data/dataset/")
