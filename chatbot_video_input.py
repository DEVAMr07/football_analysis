import tkinter as tk
from tkinter import filedialog
import threading
import time

def select_file():
    global video_path
    video_path = filedialog.askopenfilename(
        title='Select Football Video',
        filetypes=[("Video files", "*.mp4 *.avi *.mov")]
    )
    if video_path:
        update_chat_log(f"Bot: Video selected: {video_path}")

def update_chat_log(message):
    chat_log.config(state=tk.NORMAL)
    chat_log.insert(tk.END, message + "\n")
    chat_log.config(state=tk.DISABLED)
    chat_log.see(tk.END)

def process_video_and_respond(user_msg):
    update_chat_log("Bot: Processing video, please wait...")
    # Simulate long processing
    time.sleep(3)  
    # Example response: You can insert YOLO processing here
    update_chat_log(f"Bot: Analysis complete! (You asked: {user_msg})")

def send_message():
    msg = entry.get().strip()
    if not msg:
        return
    if not video_path:
        update_chat_log("Bot: Please upload a video first.")
        return

    update_chat_log(f"You: {msg}")
    entry.delete(0, tk.END)

    # Run processing in separate thread
    threading.Thread(target=process_video_and_respond, args=(msg,)).start()

root = tk.Tk()
root.title("Football Chatbot")

video_path = None

chat_log = tk.Text(root, state=tk.DISABLED, width=70, height=20)
chat_log.pack(padx=10, pady=10)

btn_upload = tk.Button(root, text="Upload Video", command=select_file)
btn_upload.pack(pady=5)

entry = tk.Entry(root, width=60)
entry.pack(side=tk.LEFT, padx=(10,0), pady=10)

btn_send = tk.Button(root, text="Send", command=send_message)
btn_send.pack(side=tk.LEFT, padx=5, pady=10)

root.mainloop()
