import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np

# Load YOLO model once (adjust path/model if needed)
model = YOLO('yolov8n.pt')

def box_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def distance(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def in_penalty_area(pos, frame_width, frame_height):
    """Simple penalty box approx: bottom center horizontal box (for one goal)"""
    # Adjust based on camera angle; example assumes goal at frame bottom center
    box_width = frame_width * 0.4  # 40% width penalty box approx
    box_height = frame_height * 0.2 # 20% height penalty box approx
    box_x_min = (frame_width - box_width) / 2
    box_x_max = box_x_min + box_width
    box_y_min = frame_height - box_height
    box_y_max = frame_height
    x, y = pos
    return (box_x_min <= x <= box_x_max) and (box_y_min <= y <= box_y_max)

def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "Error opening video file."

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    last_ball_holder_center = None
    pass_faults = []
    shots = []
    player_passes = 0
    frame_count = 0

    MAX_PASS_DISTANCE = 200  # pixels
    SHOOTING_SPEED_THRESHOLD = 40  # pixel/frame speed threshold for shot

    ball_positions = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        results = model(frame)

        players = []
        ball = None

        for result in results:
            cls_array = result.boxes.cls.cpu().numpy()
            boxes = result.boxes.xyxy.cpu().numpy()
            for i, cls in enumerate(cls_array):
                cls = int(cls)
                box = boxes[i]
                if cls == 0:  # Player
                    players.append({'bbox': box})
                elif cls == 32:  # Ball
                    ball = {'bbox': box}

        if ball is not None and players:
            ball_center = box_center(ball['bbox'])
            ball_positions.append(ball_center)

            distances = [(box_center(p['bbox']), distance(box_center(p['bbox']), ball_center)) for p in players]
            distances.sort(key=lambda x: x[1])
            current_holder_center = distances[0][0]

            # Pass detection
            if last_ball_holder_center is not None:
                dist_change = distance(last_ball_holder_center, current_holder_center)
                if dist_change > 10:
                    player_passes += 1
                    fault = None
                    suggestion = None

                    if dist_change > MAX_PASS_DISTANCE:
                        fault = f"Long risky pass"
                        suggestion = "Try a shorter, safer pass."

                    if fault:
                        pass_faults.append({
                            'frame': frame_count,
                            'fault': fault,
                            'suggestion': suggestion
                        })

            # Shot detection: check ball speed exceeding threshold
            if len(ball_positions) > 1:
                ball_speed = distance(ball_positions[-1], ball_positions[-2])
                if ball_speed > SHOOTING_SPEED_THRESHOLD:
                    # Determine if shot is inside penalty box
                    shot_pos = ball_positions[-1]
                    in_penalty = in_penalty_area(shot_pos, frame_width, frame_height)
                    shots.append({
                        'frame': frame_count,
                        'position': shot_pos,
                        'in_penalty_area': in_penalty
                    })

            last_ball_holder_center = current_holder_center

    cap.release()

    analysis_summary = {
        'total_frames': frame_count,
        'total_passes': player_passes,
        'pass_faults': pass_faults,
        'shots': shots
    }
    return analysis_summary, None

# Streamlit frontend setup
st.set_page_config(page_title="Football Video Analysis Chatbot", layout="wide")
st.title("⚽ Football Video Analysis Chatbot")
st.markdown("""
Upload a football match video, then ask questions or get detailed analysis about passes and shots including risks and tactical suggestions.
""")

uploaded_file = st.file_uploader("Upload a Football Video", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Save uploaded video temporarily
    video_path = "temp_uploaded_video.mp4"
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"Uploaded: {uploaded_file.name}")

    with st.spinner("Analyzing video... This may take a while depending on video length."):
        analysis_result, error = analyze_video(video_path)

    if error:
        st.error(error)
    else:
        st.markdown("### Video Analysis Summary")
        st.write(f"**Total Frames Processed:** {analysis_result['total_frames']}")
        st.write(f"**Total Passes Detected:** {analysis_result['total_passes']}")

        if analysis_result['pass_faults']:
            st.markdown("### Risky Passes Detected")
            for i, fault in enumerate(analysis_result['pass_faults'], 1):
                st.markdown(f"""
                <div style="
                    border: 2px solid #ff4b4b;
                    background-color: #ffe6e6;
                    border-radius: 8px;
                    padding: 15px;
                    margin-bottom: 10px;
                    box-shadow: 0 2px 6px rgba(255, 75, 75, 0.3);
                ">
                    <h4 style="color:#b30000; margin-bottom: 8px;">Fault #{i}</h4>
                    <p><strong style="color:green;">Frame:</strong> {fault['frame']}</p>
                    <p><strong style="color:green;">Issue Detected:</strong> <span style="color:#2e7d32;">{fault['fault']}</span></p>
                    <p><strong>Suggested Improvement:</strong> <span style="color:#222222; font-weight:bold;">{fault['suggestion']}</span></p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("No risky passes detected! Great passing accuracy. 👍")

        if analysis_result['shots']:
            st.markdown("### Shots Detected")
            for j, shot in enumerate(analysis_result['shots'], 1):
                position_str = f"X: {int(shot['position'][0])}, Y: {int(shot['position'][1])}"
                area_str = "inside penalty area" if shot['in_penalty_area'] else "outside penalty area"
                suggestion = ("Good shooting position" if shot['in_penalty_area'] 
                              else "Try to shoot from or get closer to the penalty area for better chances.")

                st.markdown(f"""
                <div style="
                    border: 2px solid #2e7d32;
                    background-color: #e6f4ea;
                    border-radius: 8px;
                    padding: 15px;
                    margin-bottom: 10px;
                    box-shadow: 0 2px 6px rgba(46, 125, 50, 0.3);
                ">
                    <h4 style="color:#1b5e20; margin-bottom: 8px;">Shot #{j}</h4>
                    <p><strong style="color:green;">Frame:</strong> {shot['frame']}</p>
                    <p><strong>Position:</strong> {position_str} ({area_str})</p>
                    <p><strong>Suggestion:</strong> <span style="color:#222222; font-weight:bold;">{suggestion}</span></p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No shots detected in this video.")

    st.markdown("---")
    user_input = st.text_input("Ask the chatbot (e.g., players, passes, shots):")
    if st.button("Send") and user_input.strip():
        msg = user_input.strip().lower()
        if "players" in msg:
            st.info("The model detects players in each frame and tracks the ball.")
        elif "pass" in msg:
            st.info(f"Total passes detected: {analysis_result['total_passes']} with risky passes flagged.")
        elif "shot" in msg:
            st.info(f"Shots detected: {len(analysis_result['shots'])}. Suggestions based on shot positions are provided.")
        else:
            st.warning("You can ask about 'players', 'passes', or 'shots'.")

else:
    st.info("Please upload a football video to start analysis.")
