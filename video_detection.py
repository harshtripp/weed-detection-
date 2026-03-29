import cv2
import numpy as np
import os

# =========================
# PATH SETUP
# =========================
base_dir = os.path.dirname(os.path.abspath(__file__))

weights_path = os.path.join(base_dir, "..", "data", "weights", "crop_weed_detection.weights")
config_path = os.path.join(base_dir, "..", "data", "cfg", "crop_weed.cfg")

video_path = os.path.join(base_dir, "..", "data", "output_video.avi")
# =========================
# LOAD MODEL
# =========================
net = cv2.dnn.readNet(weights_path, config_path)

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

classes = ["crop", "weed"]

# =========================
# LOAD VIDEO
# =========================
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Cannot open video")
    exit()

print("Processing video... Press Q to exit")

# =========================
# PROCESS VIDEO
# =========================
while True:
    ret, frame = cap.read()

    if not ret:
        break

    height, width, channels = frame.shape

    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (512, 512), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    # Detection
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    crop_count = 0
    weed_count = 0

    if len(indexes) > 0:
        for i in indexes.flatten():
            if class_ids[i] == 0:
                crop_count += 1
            else:
                weed_count += 1

    print("Crops:", crop_count, "Weeds:", weed_count)


    total = crop_count + weed_count

    if total > 0:
        weed_density = (weed_count / total) * 100
    else:
        weed_density = 0  


    if weed_density < 20:
        recommendation = "No action needed"
    elif weed_density < 50:
        recommendation = "Manual removal suggested"
    else:
        recommendation = "Use pesticide"    

    print(f"Weed Density: {weed_density:.2f}%")




    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = classes[class_ids[i]]
            confidence = confidences[i]

            color = (0, 255, 0) if label == "crop" else (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame,
                        f"{label}: {confidence:.2f}",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2)
            
            cv2.putText(frame,
            f"Weed Density: {weed_density:.2f}%",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2)

    cv2.putText(frame,
            f"Crops: {crop_count}  Weeds: {weed_count}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2)
    

    cv2.putText(frame,
            f"Recommendation: {recommendation}",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2)
    
    cv2.imshow("Video Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()