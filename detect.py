from ultralytics import YOLO

# Load a pre-trained YOLO model
model = YOLO('yolov8n.pt')  # lightweight model for beginners

# Run detection on your video
results = model.predict(source="9517709-uhd_4096_2160_24fps.mp4", show=True, save=True)
for result in results:
    # Bounding boxes (x1,y1,x2,y2)
    boxes = result.boxes.xyxy.cpu().numpy()

    # Confidence scores
    scores = result.boxes.conf.cpu().numpy()

    # Class IDs (e.g. 0 for person, 32 for ball)
    classes = result.boxes.cls.cpu().numpy()

    # You can print or save this info for event analysis
    print(f'Detected {len(boxes)} objects')
results = model.track(source="9517709-uhd_4096_2160_24fps.mp4", show=True, save=True, persist=True)
for result in results:
    boxes = result.boxes.xyxy.cpu().numpy()
    ids = result.boxes.id.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()
    print(f'Tracked {len(ids)} objects with IDs: {ids}')
    
