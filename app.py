import streamlit as st
import cv2
import numpy as np
import os
import pandas as pd
import time

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Smart Weed Detection", layout="wide")

st.title("🌱 Smart Crop & Weed Detection System")
st.markdown("AI-powered precision agriculture decision system")

# =========================
# SIDEBAR
# =========================
st.sidebar.title("⚙️ Settings")

mode = st.sidebar.radio("Mode", ["Image", "Video"])
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5)
show_boxes = st.sidebar.checkbox("Show Bounding Boxes", True)

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    weights_path = os.path.join(base_dir, "performing_detection", "data", "weights", "crop_weed_detection.weights")
    config_path = os.path.join(base_dir, "performing_detection", "data", "cfg", "crop_weed.cfg")

    net = cv2.dnn.readNet(weights_path, config_path)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    return net, output_layers

net, output_layers = load_model()
classes = ["crop", "weed"]

# =========================
# IMAGE MODE (MULTI IMAGE)
# =========================
if mode == "Image":

    uploaded_files = st.file_uploader(
        "📤 Upload Images",
        type=["jpg", "png", "jpeg"],
        accept_multiple_files=True
    )

    if uploaded_files:

        total_crop = 0
        total_weed = 0

        for uploaded_file in uploaded_files:

            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            output_image = image.copy()

            height, width, _ = image.shape

            blob = cv2.dnn.blobFromImage(image, 1/255.0, (512, 512), swapRB=True, crop=False)
            net.setInput(blob)
            outputs = net.forward(output_layers)

            boxes, confidences, class_ids = [], [], []

            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]

                    if confidence > confidence_threshold:
                        cx, cy = int(detection[0]*width), int(detection[1]*height)
                        w, h = int(detection[2]*width), int(detection[3]*height)
                        x, y = int(cx - w/2), int(cy - h/2)

                        boxes.append([x,y,w,h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, 0.4)

            crop_count = 0
            weed_count = 0

            if len(indexes) > 0:
                for i in indexes.flatten():
                    x, y, w, h = boxes[i]
                    label = classes[class_ids[i]]

                    if label == "crop":
                        crop_count += 1
                        color = (0,255,0)
                    else:
                        weed_count += 1
                        color = (0,0,255)

                    if show_boxes:
                        cv2.rectangle(output_image, (x,y),(x+w,y+h), color,2)

            total_crop += crop_count
            total_weed += weed_count

            st.image(output_image, caption="Processed Image", channels="BGR")

        total = total_crop + total_weed
        weed_density = (total_weed / total) * 100 if total > 0 else 0

        st.subheader("🌾 Field Summary")

        col1, col2, col3 = st.columns(3)
        col1.metric("🌾 Crops", total_crop)
        col2.metric("🌱 Weeds", total_weed)
        col3.metric("📊 Density", f"{weed_density:.2f}%")

        st.progress(int(weed_density))

        # Chart
        df = pd.DataFrame({"Type":["Crop","Weed"],"Count":[total_crop,total_weed]})
        st.bar_chart(df.set_index("Type"))

        # Recommendation
        if weed_density < 20:
            recommendation = "Healthy Field 🌱"
            st.success(recommendation)
        elif weed_density < 50:
            recommendation = "Moderate Weed Growth ⚠️"
            st.warning(recommendation)
        else:
            recommendation = "High Weed Infestation 🚨"
            st.error(recommendation)

        # Report
        report = f"""
Total Crops: {total_crop}
Total Weeds: {total_weed}
Weed Density: {weed_density:.2f}%
Recommendation: {recommendation}
"""
        st.download_button("📥 Download Report", report, file_name="image_report.txt")

# =========================
# VIDEO MODE
# =========================
if mode == "Video":

    uploaded_file = st.file_uploader("📤 Upload Video", type=["mp4","avi","mov"])

    if uploaded_file is not None:

        temp_path = "temp_video.mp4"
        with open(temp_path,"wb") as f:
            f.write(uploaded_file.read())

        cap = cv2.VideoCapture(temp_path)
        frame_window = st.image([])

        total_crop, total_weed = 0, 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            height, width, _ = frame.shape

            blob = cv2.dnn.blobFromImage(frame,1/255.0,(512,512),swapRB=True,crop=False)
            net.setInput(blob)
            outputs = net.forward(output_layers)

            boxes, confidences, class_ids = [], [], []

            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]

                    if confidence > confidence_threshold:
                        cx, cy = int(detection[0]*width), int(detection[1]*height)
                        w, h = int(detection[2]*width), int(detection[3]*height)
                        x, y = int(cx - w/2), int(cy - h/2)

                        boxes.append([x,y,w,h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, 0.4)

            if len(indexes) > 0:
                for i in indexes.flatten():
                    x,y,w,h = boxes[i]
                    label = classes[class_ids[i]]

                    if label == "crop":
                        total_crop += 1
                        color = (0,255,0)
                    else:
                        total_weed += 1
                        color = (0,0,255)

                    if show_boxes:
                        cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)

            frame_window.image(frame,channels="BGR")

        cap.release()

        total = total_crop + total_weed
        weed_density = (total_weed/total)*100 if total>0 else 0

        st.subheader("📊 Final Video Analysis")

        col1,col2,col3 = st.columns(3)
        col1.metric("🌾 Crops", total_crop)
        col2.metric("🌱 Weeds", total_weed)
        col3.metric("📊 Density", f"{weed_density:.2f}%")

        st.progress(int(weed_density))

        # Chart
        df = pd.DataFrame({"Type":["Crop","Weed"],"Count":[total_crop,total_weed]})
        st.bar_chart(df.set_index("Type"))

        # Recommendation
        if weed_density < 20:
            recommendation = "Healthy Field 🌱"
            st.success(recommendation)
        elif weed_density < 50:
            recommendation = "Moderate Weed Growth ⚠️"
            st.warning(recommendation)
        else:
            recommendation = "High Weed Infestation 🚨"
            st.error(recommendation)

        # Report
        report = f"""
Total Crops: {total_crop}
Total Weeds: {total_weed}
Weed Density: {weed_density:.2f}%
Recommendation: {recommendation}
"""
        st.download_button("📥 Download Report", report, file_name="video_report.txt")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown("EPICS PROJECT 🚀")