import cv2
import numpy as np

cap = cv2.VideoCapture(0)

# Store previous frame centers for tracking
prev_centers = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (800,600))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    current_centers = []

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Ignore small noise
        if area > 1500:
            x,y,w,h = cv2.boundingRect(cnt)

            cx = int(x + w/2)
            cy = int(y + h/2)

            current_centers.append((cx,cy))

            # Draw bounding box
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

            # Draw center point
            cv2.circle(frame, (cx,cy), 5, (0,0,255), -1)

    # -------- TRACKING LOGIC --------
    for curr in current_centers:
        for prev in prev_centers:
            dist = np.linalg.norm(np.array(curr) - np.array(prev))

            # If center is close to previous frame → same object
            if dist < 50:
                cv2.line(frame, curr, prev, (255,0,0), 2)

    prev_centers = current_centers

    cv2.imshow("Tracking Demo", frame)
    cv2.imshow("Edges", edges)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
