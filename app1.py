import time
import streamlit as st
from ultralytics import YOLO
import cv2
import glob
import os

# Load the YOLO model
model = YOLO('yolov8n.pt')

# Track classes of interest
target_classes = {
    "car": 2,
    "bus": 5,
    "motorcycle": 3,
    "person": 0
}

def simulate_traffic_light(lane_name, is_green):
    return f"🟢 Green Light: Lane '{lane_name}'" if is_green else f"🔴 Red Light: Lane '{lane_name}'"

# Streamlit setup
st.set_page_config(layout="wide")
st.title("🚦 Smart Traffic Light Control System")

# Sidebar configuration
st.sidebar.title("Configuration")
directory_path = st.sidebar.text_input("📁 Input Image Directory:")
output_directory = st.sidebar.text_input("💾 Output Annotated Directory:")

if st.sidebar.button("Load Images"):
    if directory_path and output_directory:
        image_files = [f for ext in ('*.jpg', '*.jpeg', '*.png') for f in glob.glob(os.path.join(directory_path, ext))]
        if not image_files:
            st.sidebar.error("No images found!")
        else:
            os.makedirs(output_directory, exist_ok=True)
            st.session_state.image_files = image_files
            st.sidebar.success(f"Loaded {len(image_files)} images.")
    else:
        st.sidebar.error("Specify both input and output directories!")

if 'image_files' not in st.session_state or not st.session_state.image_files:
    st.warning("Upload and load images using the sidebar.")
    st.stop()

# Process images
lane_data = {}
annotated_images = []

for idx, img_path in enumerate(st.session_state.image_files):
    image = cv2.imread(img_path)
    if image is None:
        continue

    results = model.predict(image, conf=0.3)

    counts = {"car": 0, "bus": 0, "motorcycle": 0, "person": 0}
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls)
            for cls, cls_id in target_classes.items():
                if class_id == cls_id:
                    counts[cls] += 1

    total_vehicles = counts["car"] + counts["bus"] + counts["motorcycle"]
    pedestrians = counts["person"]
    lane_name = os.path.basename(img_path).split('.')[0]

    priority_score = total_vehicles + 0.5 * pedestrians

    lane_data[lane_name] = {
        "vehicles": total_vehicles,
        "pedestrians": pedestrians,
        "counts": counts,
        "priority": priority_score,
        "jaywalking": pedestrians > total_vehicles
    }

    annotated = results[0].plot()
    out_path = os.path.join(output_directory, f"annotated_{idx}.jpg")
    cv2.imwrite(out_path, annotated)
    annotated_images.append(out_path)

# Batch processing (4 at a time)
batch_size = 4
sorted_lanes = sorted(lane_data.items(), key=lambda x: -x[1]['priority'])

traffic_batches = []
for i in range(0, len(sorted_lanes), batch_size):
    batch = sorted_lanes[i:i + batch_size]
    traffic_batches.append(batch)

# Display: Original and Annotated images
col1, col2 = st.columns([1, 1])

with col1:
    with st.expander("🖼️ Original Images", expanded=True):
        for i in range(0, len(st.session_state.image_files), 4):
            cols = st.columns(4)
            for j, path in enumerate(st.session_state.image_files[i:i+4]):
                img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
                cols[j].image(img, use_container_width=True, caption=os.path.basename(path))

with col2:
    with st.expander("🧠 Annotated Images", expanded=True):
        for i in range(0, len(annotated_images), 4):
            cols = st.columns(4)
            for j, path in enumerate(annotated_images[i:i+4]):
                img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
                cols[j].image(img, use_container_width=True, caption=os.path.basename(path))

st.divider()
st.subheader("🚥 Traffic Light Simulation (with Jaywalking Detection)")

for idx, batch in enumerate(traffic_batches):
    st.markdown(f"### ⏱️ Batch {idx + 1}")

    for i, (lane, stats) in enumerate(batch):
        is_green = (i == 0)
        message = simulate_traffic_light(lane, is_green)
        st.write(f"{message} | 🚗 Vehicles: {stats['vehicles']}, 🚶 Pedestrians: {stats['pedestrians']}")

        if stats["jaywalking"]:
            st.markdown("<span style='color:yellow'>⚠️ Jaywalking Detected: Pedestrians > Vehicles</span>", unsafe_allow_html=True)

    if idx < len(traffic_batches) - 1:
        st.info("⏳ Waiting 5 seconds before next batch...")
        time.sleep(5)

# Cleanup
for path in annotated_images:
    try:
        os.remove(path)
    except Exception as e:
        st.warning(f"Could not delete {path}: {e}")
