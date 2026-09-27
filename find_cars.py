from ultralytics import YOLO
import cv2

print("Waking up the AI Brain...")
# 1. Load the AI Brain (YOLOv8 Nano - very fast!)
model = YOLO('yolov8n.pt') 

# 2. Load your giant stitched map picture
image_path = 'Map 1.png'
image = cv2.imread(image_path)

if image is None:
    print(f"Oops! Couldn't find {image_path}. Make sure it is in this folder.")
else:
    print("Looking at the map to find cars...")
    # 3. Ask the AI to find everything it recognizes!
    results = model(image)

    # 4. Draw the boxes and count cars!
    car_count = 0
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # In the AI's brain, Class #2 means "car" and Class #7 means "truck"
            class_id = int(box.cls[0])
            if class_id == 2 or class_id == 7:
                car_count += 1
                
                # Get the coordinates for the box and draw a blue rectangle
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 4)

    print(f"\nSUCCESS! I found {car_count} cars/trucks on the map!")

    # 5. Save the new picture with the cars highlighted
    cv2.imwrite('Cars_Found_Map.png', image)
    print("Saved the picture as 'Cars_Found_Map.png'!")
