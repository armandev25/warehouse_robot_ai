import cv2
import numpy as np
import tensorflow as tf

# ---- LOAD MODEL ----
model = tf.keras.models.load_model("models/warehouse_classifier.keras")
labels = ["fragile", "hazardous", "heavy"]

# ---- PRELOAD STATIC INSTRUCTIONS (FAST DEMO MODE) ----
instructions = {
    "fragile": """
FRAGILE ITEMS:
Use low acceleration and reduced grip force.
Do not stack fragile items more than two layers.
Use padded grippers when available.
""",
    "heavy": """
HEAVY ITEMS:
Verify torque limits before lifting.
Keep load centered.
Avoid steep surfaces while transporting.
""",
    "hazardous": """
HAZARDOUS MATERIALS:
Perform safety scan before pickup.
Use sealed containment mode.
Keep away from heat sources.
"""
}

print("\nRobot system ready.")
print("Press 'h' for handling instructions")
print("Press 'q' to quit\n")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame,(400,400))

    # ---- CLASSIFY ----
    img = cv2.resize(frame,(224,224))/255.0
    pred = model.predict(np.expand_dims(img,0), verbose=0)
    label = labels[np.argmax(pred)]

    cv2.putText(frame, f"Class: {label}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)

    cv2.imshow("Warehouse Robot", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        print("\nExiting...")
        break

    if key == ord('h'):
        print("\nHandling Instructions:")
        print(instructions[label])

cap.release()
cv2.destroyAllWindows()
