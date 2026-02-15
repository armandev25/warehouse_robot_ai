"""
Simplified Webcam Demo - Computer Vision Only
No ML dependencies - just object detection and tracking
"""

import cv2
import numpy as np
from collections import OrderedDict
from datetime import datetime
import sys


class SimpleCentroidTracker:
    """Simple centroid-based object tracker"""
    
    def __init__(self, max_disappeared=50, max_distance=75):
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        
    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1
        
    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]
        
    def update(self, rects):
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects
        
        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (x, y, w, h)) in enumerate(rects):
            cx = int(x + w / 2.0)
            cy = int(y + h / 2.0)
            input_centroids[i] = (cx, cy)
        
        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            
            D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - 
                             input_centroids, axis=2)
            
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            used_rows = set()
            used_cols = set()
            
            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                if D[row, col] > self.max_distance:
                    continue
                
                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0
                
                used_rows.add(row)
                used_cols.add(col)
            
            unused_rows = set(range(D.shape[0])) - used_rows
            unused_cols = set(range(D.shape[1])) - used_cols
            
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            
            for col in unused_cols:
                self.register(input_centroids[col])
        
        return self.objects


class SimpleVisionDetector:
    """Simplified computer vision detector"""
    
    def __init__(self, min_area=1000, max_area=100000):
        self.min_area = min_area
        self.max_area = max_area
        self.tracker = SimpleCentroidTracker()
        
    def preprocess_frame(self, frame):
        """Preprocess frame for detection"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11, 2
        )
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel, iterations=1)
        
        return morph
    
    def detect_objects(self, frame):
        """Detect objects and return bounding boxes with IDs"""
        processed = self.preprocess_frame(frame)
        
        # Find contours (compatible with OpenCV 3.x and 4.x)
        try:
            contours, _ = cv2.findContours(
                processed,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
        except:
            # For older OpenCV versions
            _, contours, _ = cv2.findContours(
                processed,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
        
        # Filter and extract objects
        objects = []
        bboxes = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area < self.min_area or area > self.max_area:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            
            # Aspect ratio filter
            aspect_ratio = float(w) / h if h > 0 else 0
            if aspect_ratio < 0.2 or aspect_ratio > 5.0:
                continue
            
            # Calculate centroid
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx = x + w // 2
                cy = y + h // 2
            
            bboxes.append((x, y, w, h))
            objects.append({
                'bbox': (x, y, w, h),
                'centroid': (cx, cy),
                'area': area,
                'aspect_ratio': aspect_ratio
            })
        
        # Update tracker
        tracked_objects = self.tracker.update(bboxes)
        
        # Match detections to tracked IDs
        results = []
        for object_id, centroid in tracked_objects.items():
            # Find matching detection
            min_dist = float('inf')
            matched_obj = None
            
            for obj in objects:
                obj_cx, obj_cy = obj['centroid']
                dist = np.sqrt((obj_cx - centroid[0])**2 + (obj_cy - centroid[1])**2)
                if dist < min_dist:
                    min_dist = dist
                    matched_obj = obj
            
            if matched_obj and min_dist < 50:
                matched_obj['id'] = object_id
                results.append(matched_obj)
        
        return results


def main():
    print("\n" + "="*80)
    print(" "*20 + "SIMPLIFIED WAREHOUSE VISION DEMO")
    print(" "*25 + "Computer Vision Only")
    print("="*80)
    print("\nThis version uses only OpenCV - no ML dependencies!")
    print("It will detect and track objects, but won't classify them.\n")
    
    # Try to open webcam
    print("Opening webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("\n✗ Error: Could not access webcam")
        print("\nTroubleshooting:")
        print("  1. Make sure your webcam is connected")
        print("  2. Check if another application is using the camera")
        print("  3. Try running: python test_webcam.py")
        sys.exit(1)
    
    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    
    print(f"\n✓ Webcam opened successfully!")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    
    # Initialize detector
    print("\nInitializing object detector...")
    detector = SimpleVisionDetector(min_area=1000, max_area=100000)
    print("✓ Detector ready!")
    
    # Instructions
    print("\n" + "="*80)
    print("INSTRUCTIONS")
    print("="*80)
    print("""
Place objects in front of the camera:
  • Boxes, books, or packages work well
  • Good lighting helps detection
  • Objects should be clearly visible

KEYBOARD CONTROLS:
  'q' - Quit
  'p' - Pause/Resume
  's' - Save screenshot
  '+' - Increase sensitivity (lower min area)
  '-' - Decrease sensitivity (higher min area)
  'h' - Show help
    """)
    
    print("="*80)
    print("Press SPACE to start detection...")
    print("(Click on the window that appears, then press SPACE)")
    print("="*80 + "\n")
    
    # Create window first
    window_name = 'Warehouse Vision - Simplified Demo'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Show first frame
    ret, first_frame = cap.read()
    if ret:
        instructions = first_frame.copy()
        h, w = instructions.shape[:2]
        
        # Add start message
        cv2.putText(instructions, "PRESS SPACE TO START", (w//4, h//2),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.putText(instructions, "Press 'q' to quit", (w//3, h//2 + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.imshow(window_name, instructions)
    
    # Wait for spacebar specifically
    while True:
        key = cv2.waitKey(100) & 0xFF
        if key == ord(' ') or key == 13:  # Space or Enter
            break
        elif key == ord('q'):
            print("\nCancelled by user")
            cap.release()
            cv2.destroyAllWindows()
            sys.exit(0)
    
    # Main loop
    paused = False
    frame_count = 0
    total_detections = 0
    
    print("🎥 LIVE DETECTION ACTIVE\n")
    
    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("\n✗ Error reading frame")
                    break
                
                frame_count += 1
                
                # Detect objects
                results = detector.detect_objects(frame)
                total_detections += len(results)
                
                # Draw detections
                output_frame = frame.copy()
                
                for obj in results:
                    x, y, w, h = obj['bbox']
                    obj_id = obj['id']
                    cx, cy = obj['centroid']
                    
                    # Draw bounding box
                    cv2.rectangle(output_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Draw object ID
                    label = f"Object ID: {obj_id}"
                    cv2.putText(output_frame, label, (x, y-25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Draw size
                    size_label = f"{w}x{h} px"
                    cv2.putText(output_frame, size_label, (x, y-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    
                    # Draw centroid
                    cv2.circle(output_frame, (cx, cy), 5, (0, 0, 255), -1)
                    
                    # Draw center coordinates
                    coord_label = f"({cx},{cy})"
                    cv2.putText(output_frame, coord_label, (cx-40, cy-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
                
                # Draw statistics
                stats_text = f"Objects: {len(results)} | Frame: {frame_count}"
                cv2.putText(output_frame, stats_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Draw help
                help_text = "Press 'h' for help | 'q' to quit"
                cv2.putText(output_frame, help_text, (10, height-20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                # Show frame
                cv2.imshow(window_name, output_frame)
            
            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n✓ Quitting...")
                break
                
            elif key == ord('p'):
                paused = not paused
                status = "PAUSED" if paused else "RESUMED"
                print(f"\n{status}")
                
            elif key == ord('s'):
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"detection_{timestamp}.jpg"
                cv2.imwrite(filename, output_frame)
                print(f"\n✓ Saved: {filename}")
                
            elif key == ord('+') or key == ord('='):
                detector.min_area = max(500, detector.min_area - 500)
                print(f"\n✓ Min area: {detector.min_area} (more sensitive)")
                
            elif key == ord('-') or key == ord('_'):
                detector.min_area = min(10000, detector.min_area + 500)
                print(f"\n✓ Min area: {detector.min_area} (less sensitive)")
                
            elif key == ord('h'):
                print("\n" + "="*60)
                print("KEYBOARD CONTROLS")
                print("="*60)
                print("  'q' - Quit")
                print("  'p' - Pause/Resume")
                print("  's' - Save screenshot")
                print("  '+' - More sensitive (detect smaller objects)")
                print("  '-' - Less sensitive (ignore small objects)")
                print("  'h' - Show this help")
                print("="*60 + "\n")
    
    except KeyboardInterrupt:
        print("\n\n✓ Interrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        # Print statistics
        print("\n" + "="*80)
        print("SESSION STATISTICS")
        print("="*80)
        print(f"  Frames processed: {frame_count}")
        print(f"  Total detections: {total_detections}")
        if frame_count > 0:
            print(f"  Average objects/frame: {total_detections/frame_count:.2f}")
        print("="*80)
        print("\n✓ Demo complete!\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
