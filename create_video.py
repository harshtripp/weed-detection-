import cv2
import os

# Get correct base directory
base_dir = os.path.dirname(os.path.abspath(__file__))

# Path to images folder
image_folder = os.path.join(base_dir, "images")

# Output video name
video_name = os.path.join(base_dir, "output_video.avi")

# Get all images
images = [img for img in os.listdir(image_folder) if img.endswith((".jpg", ".png", ".jpeg"))]

# Sort images
images.sort()

# Read first image
first_image_path = os.path.join(image_folder, images[0])
frame = cv2.imread(first_image_path)

height, width, layers = frame.shape

# Create video writer
video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'XVID'), 2, (width, height))

print("Creating video...")

# Add images
for image in images:
    img_path = os.path.join(image_folder, image)
    frame = cv2.imread(img_path)

    if frame is None:
        continue

    video.write(frame)

video.release()

print("Video created successfully:", video_name)