import cv2
import numpy as np
from ultralytics import YOLO
from collections import Counter
import os


class RetailAnomalyDetector:
    def __init__(self, model_weight='yolov8s.pt'):
        """
        Initializes YOLO. Using 'yolov8s.pt' (small) for a balance of speed and accuracy.
        'yolov8n.pt' (nano) is faster but less accurate.
        """
        print(f"Loading Light RAG Object Detector ({model_weight})...")
        # The first run will download the weights automatically
        self.model = YOLO(model_weight)
        # Define classes usually found together to avoid false positives
        # e.g., if dominant is 'bottle', 'cup' might also be acceptable.
        # For this demo, we will stick to strict dominant category matching.

    def detect_and_highlight(self, image_path):
        # 1. Load Image
        if not os.path.exists(image_path):
            print(f"Error: Input image not found at {image_path}")
            return

        img = cv2.imread(image_path)
        if img is None:
            print("Error: Could not read image.")
            return

        output_img = img.copy()

        # 2. Run Inference
        print("Scanning image for products...")
        results = self.model(img, conf=0.4, verbose=False)  # conf=0.4 filters weak detections
        result = results[0]

        if len(result.boxes) == 0:
            print("No items detected on shelf.")
            return

        # 3. Extract Detected Categories
        detected_categories = []
        detected_objects = []  # Store details for later processing

        for box in result.boxes:
            class_id = int(box.cls)
            class_name = self.model.names[class_id]
            detected_categories.append(class_name)

            # Get coordinates for later drawing
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detected_objects.append({
                'class': class_name,
                'coords': (x1, y1, x2, y2)
            })

        # 4. Determine Dominant Category (The "Normal" for this shelf)
        # Use Counter to find the most frequent item class
        category_counts = Counter(detected_categories)
        most_common = category_counts.most_common(1)

        if not most_common:
            print("Could not determine dominant category.")
            return

        dominant_category = most_common[0][0]
        count = most_common[0][1]
        total_items = len(detected_categories)

        print(f"Analysis complete: Found {total_items} items.")
        print(f"Dominant Category defining shelf context: '{dominant_category.upper()}' (constitutes {count}/{total_items} items)")

        # 5. Identify and Highlight Anomalies based on context
        anomalies_found = 0
        for obj in detected_objects:
            x1, y1, x2, y2 = obj['coords']

            if obj['class'] != dominant_category:
                # --- THE HIGHLIGHTING LOGIC ---
                anomalies_found += 1
                print(f"!!! Misplaced item detected: {obj['class']}")

                # Draw thick RED bounding box for anomaly
                cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 0, 255), 5)

                # Add label tag above box
                label = f"MISPLACED: {obj['class']}"
                # Calculate text size for background rectangle
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                # Draw filled red rectangle behind text for readability
                cv2.rectangle(output_img, (x1, y1 - h - 10), (x1 + w, y1), (0, 0, 255), -1)
                # Draw white text
                cv2.putText(output_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            else:
                # Optional: Draw thin green box for correct items for context
                cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 1)

        # 6. Final Output Generation
        output_path=image_path.replace(".","_new.").replace("data_in","data_out")
        if anomalies_found > 0:
            print(f"Successfully processed. Found {anomalies_found} misplaced item(s).")
            cv2.imwrite(output_path, output_img)
            print(f"Saved highlighted image to: {output_path}")
            # Optional: display result immediately
            # cv2.imshow("Misplaced Product Detection", output_img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
        else:
            print("Shelf compliant. No misplaced items detected.")

# --- Main Execution ---
if __name__ == "__main__":
    # The image should contain a repeating pattern of items (e.g., many bottles)
    # and one odd item out (e.g., an apple or cup among bottles).

    # Retail store shelf item anomaly detector
    for root, dirs, files in os.walk(os.path.abspath("data_in")):
        for file in files:
            print(os.path.join(root, file))
            input_image_path = os.path.join(root, file)
            detector = RetailAnomalyDetector()
            detector.detect_and_highlight(input_image_path)

    print("Done")