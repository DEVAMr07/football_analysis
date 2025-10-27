import tkinter as tk
from tkinter import filedialog
import threading
import cv2
from ultralytics import YOLO
import numpy as np

# Load YOLO model once
model = YOLO('yolov8n.pt')

def box_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def distance(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

# Simple pass fault detection state
last_ball_holder_center = None  # Track ball holder by position (center coord)
pass_faults = []

def select_file():
    file_path = filedialog.askopenfilename(
        title='Select a video file',
        filetypes=[('Video files', '*.mp4 *.avi *.mov')]
    )
    if file_path:
        chat_log.config(state=tk.NORMAL)
        chat_log.insert(tk.END, f"You uploaded: {file_path}\n")
        chat_log.config(state=tk.DISABLED)
        chat_log.see(tk.END)
        threading.Thread(target=process_video, args=(file_path,)).start()

def process_video(path):
    global last_ball_holder_center, pass_faults
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        chat_log.config(state=tk.NORMAL)
        chat_log.insert(tk.END, "Bot: Error opening video file.\n")
        chat_log.config(state=tk.DISABLED)
        return

    frame_count = 0
    player_passes = 0
    pass_faults.clear()
    last_ball_holder_center = None

    MAX_PASS_DISTANCE = 200  # Threshold for risky long passes

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
                if cls == 0:
                    players.append({'bbox': box})
                elif cls == 32:
                    ball = {'bbox': box}

        if ball is not None and players:
            ball_center = box_center(ball['bbox'])

            # Find closest player bbox center to ball center = current ball holder
            distances = [(box_center(p['bbox']), distance(box_center(p['bbox']), ball_center)) for p in players]
            distances.sort(key=lambda x: x[1])
            current_holder_center = distances[0][0]

            # Detect pass if ball holder center changed significantly from last frame
            pass_occurred = False
            if last_ball_holder_center is not None:
                dist_change = distance(last_ball_holder_center, current_holder_center)
                if dist_change > 10:  # Small tolerance to avoid noise
                    pass_occurred = True
                    player_passes += 1

                    pass_distance = dist_change
                    fault = None
                    suggestion = None

                    if pass_distance > MAX_PASS_DISTANCE:
                        fault = f"Long risky pass ({int(pass_distance)}) pixels"
                        suggestion = "Try a shorter, safer pass"

                    if fault:
                        pass_faults.append((frame_count, fault, suggestion))

            last_ball_holder_center = current_holder_center

    cap.release()

    # Prepare summary message
    summary = f"Processed {frame_count} frames.\nTotal passes detected: {player_passes}\n"
    if pass_faults:
        summary += "Detected some risky passes:\n"
        for f in pass_faults:
            frame, fault, suggest = f
            summary += f"At frame {frame}: {fault}. Suggestion: {suggest}\n"
    else:
        summary += "No risky passes detected."

    chat_log.config(state=tk.NORMAL)
    chat_log.insert(tk.END, "Bot: Video analysis complete.\n" + summary + "\n\n")
    chat_log.config(state=tk.DISABLED)
    chat_log.see(tk.END)

def send_message():
    msg = entry.get().strip()
    if not msg:
        return

    chat_log.config(state=tk.NORMAL)
    chat_log.insert(tk.END, "You: " + msg + "\n")

    if "players" in msg.lower():
        bot_response = "The AI detects and counts players in the video."
    elif "pass" in msg.lower():
        bot_response = "The AI detects passes and flags risky long passes with suggestions."
    else:
        bot_response = "You can ask about players or passes in the video."

    chat_log.insert(tk.END, "Bot: " + bot_response + "\n\n")
    chat_log.config(state=tk.DISABLED)
    chat_log.see(tk.END)
    entry.delete(0, tk.END)

root = tk.Tk()
root.title("Football Video Chatbot")

chat_log = tk.Text(root, state=tk.DISABLED, width=80, height=20)
chat_log.pack(padx=10, pady=10)

btn_upload = tk.Button(root, text="Upload Video", command=select_file)
btn_upload.pack(pady=5)

entry = tk.Entry(root, width=60)
entry.pack(side=tk.LEFT, padx=(10,0), pady=10)

btn_send = tk.Button(root, text="Send", command=send_message)
btn_send.pack(side=tk.LEFT, padx=5, pady=10)

root.mainloop()