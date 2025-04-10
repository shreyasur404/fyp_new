import time
import streamlit as st
from ultralytics import YOLO
import cv2
import glob
import os

# Load the YOLOv8 model (use a pre-trained model such as 'yolov8n', 'yolov8s', etc.)
model = YOLO('yolov8n.pt')  # Replace with a fine-tuned model if available for specific traffic datasets

# Define the object classes to count
target_classes = {"car": 2, "bus": 5}  # Class IDs based on the YOLOv8 COCO dataset

# Function to simulate traffic lights
def simulate_traffic_light(lane_name, is_green):
    return f"🟢 Green Light: Lane '{lane_name}'" if is_green else f"🔴 Red Light: Lane '{lane_name}'"

# Streamlit app UI
st.set_page_config(layout="wide")
st.title("Automated Traffic Light Simulation")

# Sidebar for directory input
st.sidebar.title("Image Upload Configuration")
directory_path = st.sidebar.text_input("Enter the path to the images directory:")
output_directory = st.sidebar.text_input("Enter the path to save annotated images:")

# Load images button
if st.sidebar.button("Load Images from Directory"):
    if directory_path and output_directory:
        image_extensions = ('*.jpg', '*.png', '*.jpeg')
        image_files = [file for ext in image_extensions for file in glob.glob(os.path.join(directory_path, ext))]
        
        if not image_files:
            st.sidebar.error("No images found in the specified directory!")
        else:
            st.sidebar.success(f"Loaded {len(image_files)} images from '{directory_path}'")
            os.makedirs(output_directory, exist_ok=True)
    else:
        st.sidebar.error("Please enter valid directory paths for input and output.")

if 'image_files' not in locals() or not image_files:
    st.write("Please load images using the sidebar.")
    st.stop()

# Process images
lane_counts = {}
annotated_images = []

for idx, image_path in enumerate(image_files, start=1):
    image = cv2.imread(image_path)
    if image is None:
        st.write(f"Could not load image: {image_path}")
        continue

    results = model.predict(source=image, save=False, conf=0.3)
    
    car_count = sum(int(box.cls) == target_classes["car"] for result in results for box in result.boxes)
    bus_count = sum(int(box.cls) == target_classes["bus"] for result in results for box in result.boxes)
    
    annotated_image = results[0].plot()
    annotated_image_path = os.path.join(output_directory, f"ann_image_{idx}.jpg")
    cv2.imwrite(annotated_image_path, annotated_image)
    annotated_images.append(annotated_image_path)
    
    lane_name = os.path.basename(image_path).split('.')[0]
    lane_counts[lane_name] = {"cars": car_count, "buses": bus_count, "total": car_count + bus_count}

# Sort lanes by priority
sorted_lanes = sorted(lane_counts.items(), key=lambda x: x[1]['total'], reverse=True)

# Traffic light scheduling
traffic_light_results = []
batch_size = 4
for i in range(0, len(sorted_lanes), batch_size):
    batch = sorted_lanes[i:i + batch_size]
    batch_results = [simulate_traffic_light(lane, priority == 0) + f" | Cars: {counts['cars']}, Buses: {counts['buses']}, Total: {counts['total']}"
                     for priority, (lane, counts) in enumerate(batch)]
    traffic_light_results.append(batch_results)

# Layout
col1, col2, col3 = st.columns([2, 2, 3])

with col1:
    st.markdown("<p class='stHeader'>Original Traffic Images</p>", unsafe_allow_html=True)
    for i in range(0, len(image_files), 4):
        cols = st.columns(4)
        for idx, image_path in enumerate(image_files[i:i + 4]):
            image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            cols[idx].image(image, use_container_width=True, caption=os.path.basename(image_path))

with col2:
    st.markdown("<p class='stHeader'>Annotated Traffic Images</p>", unsafe_allow_html=True)
    for i in range(0, len(annotated_images), 4):
        cols = st.columns(4)
        for idx, annotated_image_path in enumerate(annotated_images[i:i + 4]):
            image = cv2.cvtColor(cv2.imread(annotated_image_path), cv2.COLOR_BGR2RGB)
            cols[idx].image(image, use_container_width=True, caption=os.path.basename(annotated_image_path))

with col3:
    st.markdown("<p class='stHeader'>Traffic Light Results</p>", unsafe_allow_html=True)
    st.markdown("<p class='stText'>Traffic light simulation based on priority scheduling:</p>", unsafe_allow_html=True)
    
    if traffic_light_results:
        for batch_index, batch_results in enumerate(traffic_light_results):
            st.markdown(f"<p class='stHeader'>Batch {batch_index + 1}</p>", unsafe_allow_html=True)
            for result in batch_results:
                st.markdown(f"<p class='stText'>{result}</p>", unsafe_allow_html=True)
            if batch_index < len(traffic_light_results) - 1:
                st.markdown("<p class='stText'>Processing next batch in 5 seconds...</p>", unsafe_allow_html=True)
                time.sleep(5)
        
        for annotated_image_path in annotated_images:
            try:
                os.remove(annotated_image_path)
            except OSError as e:
                st.write(f"Error deleting file {annotated_image_path}: {e}")
        
        st.markdown("<p class='stText'>All annotated images have been deleted from the directory.</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p class='stText'>No results to display. Please process images and run the simulation.</p>", unsafe_allow_html=True)
