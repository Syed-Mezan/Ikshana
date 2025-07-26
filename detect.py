import cv2
from datetime import datetime
from fpdf import FPDF
from ultralytics import YOLO
import requests
import os
import qrcode 
import google.generativeai as genai # New import

# === GLOBAL STATE ===
detection_active = False
recording_active = False
video_writer = None
tree_count = 0
current_mode = "tree"
detection_log = []
tracked_objects = {}
CONF_THRESHOLD = 0.70
IOU_THRESHOLD = 0.50
ZOOM_ENABLED = False

# === LOAD MODELS ===
try:
    model_leaf = YOLO("models/best_leaf.pt")
    model_tree = YOLO("models/best_Tree.pt")
except Exception as e:
    print(f"Error loading models: {e}.")
    exit()

# === CONTROL FUNCTIONS (Unchanged from before) ===
def set_mode(mode):
    global current_mode, CONF_THRESHOLD
    if mode in ['leaf', 'tree']:
        current_mode = mode
        if mode == 'leaf': CONF_THRESHOLD = 0.30
        else: CONF_THRESHOLD = 0.70
        return CONF_THRESHOLD
    return None

def set_thresholds(conf, iou):
    global CONF_THRESHOLD, IOU_THRESHOLD
    if conf is not None: CONF_THRESHOLD = conf
    if iou is not None: IOU_THRESHOLD = iou

def set_zoom_state(enabled: bool):
    global ZOOM_ENABLED
    ZOOM_ENABLED = enabled

def start_detection():
    global detection_active, tree_count, detection_log, tracked_objects
    detection_active = True
    tree_count = 0
    detection_log = []
    tracked_objects = {}

def stop_detection():
    global detection_active
    detection_active = False

def start_recording():
    global recording_active
    recording_active = True

def stop_recording():
    global recording_active, video_writer
    recording_active = False
    if video_writer: video_writer.release(); video_writer = None

def get_model():
    return model_leaf if current_mode == 'leaf' else model_tree

def get_detection_summary():
    if not detection_log: return {}
    summary = {}
    unique_detections = set(detection_log)
    for entry in unique_detections:
        try:
            disease = entry.split(" (")[0]
            summary[disease] = summary.get(disease, 0) + 1
        except (ValueError, IndexError):
            continue
    return summary

# === NEW: GEMINI AI RECOMMENDATION FUNCTION ===
def generate_ai_recommendation():
    """
    Analyzes the detection log using the Gemini API and returns recommendations.
    """
    summary = get_detection_summary()
    if not summary:
        return "No detections were made, so no analysis can be generated. Please run a scan first."

    try:
        # Configure the API key from environment variables
        api_key = os.getenv("AIzaSyA7yDQNaUQPMv-t4BrEynX47XZQ6gjInU4")
        if not api_key:
            return "Error: GEMINI_API_KEY not found. Please set the environment variable."
        genai.configure(api_key=api_key)
        
        # Create the model
        model = genai.GenerativeModel('gemini-1.5-flash')

        # Format the detection data for the prompt
        detection_data = "\n".join([f"- {count} instances of {disease}" for disease, count in summary.items()])

        # Construct a detailed prompt
        prompt = f"""
        Act as an expert agronomist. Based on the following drone-based visual inspection summary of a crop field, provide a concise analysis and a list of actionable recommendations for the farmer.

        Detection Summary:
        {detection_data}
        Total Plants Counted: {tree_count}

        Provide your response in two parts:
        1.  **Analysis:** A brief, one or two-sentence summary of the overall crop health based on the data.
        2.  **Recommendations:** A bulleted list of clear, prioritized actions the farmer should take.
        """
        
        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        print(f"Error during Gemini API call: {e}")
        return "An error occurred while generating the AI recommendation. Please check the console log."

# === MAIN VIDEO FEED GENERATOR (Unchanged from before) ===
def gen_frames():
    # ... (This entire function is the same as the last version)
    global video_writer, tree_count, detection_log, tracked_objects
    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): print("Error: Could not open video stream."); return
    color_map = {'tree': (0, 255, 0), 'blight': (0, 0, 255), 'leaf spot': (0, 165, 255), 'rust': (30, 105, 210)}
    default_color = (255, 0, 255)
    while True:
        success, frame = cap.read()
        if not success or frame is None: continue
        frame_height, frame_width, _ = frame.shape
        processed_frame = frame.copy() 
        line_y = int(frame_height * 0.5)
        if detection_active:
            model = get_model()
            results = model.track(source=frame, persist=True, tracker="bytetrack.yaml", conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)[0]
            if current_mode == 'tree': cv2.line(processed_frame, (0, line_y), (frame_width, line_y), (0, 255, 0), 2)
            first_detection_box = None
            if results.boxes is not None:
                for box in results.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0]); conf = float(box.conf[0]); label = model.names[int(box.cls[0])]
                    if first_detection_box is None: first_detection_box = (x1, y1, x2, y2)
                    display_label = label
                    if box.id is not None: track_id = int(box.id[0]); display_label = f"{label} ID:{track_id}"
                    color = color_map.get(label.lower(), default_color)
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
                    text_size, _ = cv2.getTextSize(display_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2); text_w, text_h = text_size
                    cv2.rectangle(processed_frame, (x1, y1 - text_h - 15), (x1 + text_w + 10, y1), color, -1)
                    cv2.putText(processed_frame, display_label, (x1 + 5, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    detection_log.append(f"{label} ({conf*100:.1f}%)")
                    if label.lower() == "tree" and box.id is not None:
                        track_id = int(box.id[0]); center_y = int((y1 + y2) / 2)
                        if track_id in tracked_objects:
                            prev_y = tracked_objects[track_id]['y']
                            if prev_y < line_y and center_y >= line_y and not tracked_objects[track_id]['counted']:
                                tree_count += 1; tracked_objects[track_id]['counted'] = True; cv2.line(processed_frame, (0, line_y), (frame_width, line_y), (0, 0, 255), 4)
                            elif prev_y > line_y and center_y <= line_y and not tracked_objects[track_id]['counted']:
                                tree_count += 1; tracked_objects[track_id]['counted'] = True; cv2.line(processed_frame, (0, line_y), (frame_width, line_y), (0, 0, 255), 4)
                            tracked_objects[track_id]['y'] = center_y
                        else:
                            tracked_objects[track_id] = {'y': center_y, 'counted': False}
            if len(results.boxes) > 0: cv2.imwrite("last_frame.jpg", frame)
            final_frame = processed_frame
            if ZOOM_ENABLED and current_mode == 'leaf' and first_detection_box is not None:
                x1, y1, x2, y2 = first_detection_box
                center_x = (x1 + x2) // 2; center_y = (y1 + y2) // 2; zoom_factor = 2.0; zoomed_w = int(frame_width / zoom_factor); zoomed_h = int(frame_height / zoom_factor)
                crop_x1 = max(0, center_x - zoomed_w // 2); crop_y1 = max(0, center_y - zoomed_h // 2); crop_x2 = min(frame_width, crop_x1 + zoomed_w); crop_y2 = min(frame_height, crop_y1 + zoomed_h)
                zoomed_region = processed_frame[crop_y1:crop_y2, crop_x1:crop_x2]
                final_frame = cv2.resize(zoomed_region, (frame_width, frame_height))
        else:
             final_frame = frame 
        if recording_active:
            if video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*'MJPG'); timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                video_writer = cv2.VideoWriter(f'rec_{timestamp}.avi', fourcc, 20.0, (frame.shape[1], frame.shape[0]))
            if video_writer is not None: video_writer.write(final_frame)
        ret, buffer = cv2.imencode('.jpg', final_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    cap.release()

# === PDF Generation (Unchanged from before) ===
def clean_text(text):
    # ... (rest of PDF code is unchanged)
    return ''.join([c for c in text if ord(c) < 128])
class PDF(FPDF):
    def header(self):
        if not os.path.exists("logo_online.png"):
            try:
                logo_url = "https://multiplexdrone.com/wp-content/uploads/2025/01/M-DRONE_Final-Logo.png"
                r = requests.get(logo_url);
                if r.status_code == 200:
                    with open("logo_online.png", "wb") as f: f.write(r.content)
            except Exception as e: print(f"Could not download logo: {e}")
        if os.path.exists("logo_online.png"): self.image("logo_online.png", 10, 8, 25)
        self.set_font('Arial', 'B', 12); self.cell(0, 10, 'M-Drone: Disease & Crop Health Report', 0, 1, 'C'); self.ln(5)
    def footer(self):
        self.set_y(-15); self.set_font('Arial', 'I', 8); self.cell(0, 5, 'Generated by M-Drone | Ikshana', 0, 1, 'C'); self.cell(0, 5, f'Page {self.page_no()}', 0, 0, 'C')
    def chapter_title(self, title):
        self.set_font('Arial', 'B', 14); self.set_fill_color(76, 175, 80); self.set_text_color(255, 255, 255); self.cell(0, 10, title, 0, 1, 'L', fill=True); self.ln(4); self.set_text_color(0, 0, 0)
    def info_table(self, data):
        self.set_font('Arial', 'B', 10); col_width = self.w / 2.2
        for row in data:
            self.cell(col_width, 8, row[0], border=1); self.set_font('Arial', '', 10); self.cell(col_width, 8, row[1], border=1); self.ln()
        self.ln(5); self.set_font('Arial', 'B', 10)
def generate_pdf_report(location, crop_type, farm_area, operator_name):
    if not detection_log or not os.path.exists("last_frame.jpg"): return None
    now = datetime.now(); date_str = now.strftime("%m%d"); location_str = location.split(',')[0].replace(' ', '').upper()
    report_id = f"MDR-AGRI-{date_str}-{location_str}"; pdf_filename = f"{report_id}.pdf"
    pdf = PDF(); pdf.add_page()
    pdf.chapter_title('1. Scan & Location Details')
    pdf.info_table([['Location:', clean_text(location)],['Date & Time of Scan:', clean_text(now.strftime("%Y-%m-%d %H:%M:%S"))],['Operator Name:', clean_text(operator_name)]])
    pdf.chapter_title('2. Field Overview')
    try: density = int(tree_count) / float(farm_area) if float(farm_area) > 0 else 0
    except (ValueError, ZeroDivisionError): density = 0
    pdf.info_table([['Total Farm Area:', f"{farm_area} acres"],['Crop Type:', clean_text(crop_type)],['Detected Plant Count (Line-Cross):', str(tree_count)],['Avg. Plant Density:', f"{density:.2f} plants/acre"]])
    pdf.add_page(); pdf.chapter_title('3. Detection Snapshot'); pdf.image("last_frame.jpg", x=10, w=pdf.w - 20); pdf.ln(4); pdf.set_font('Arial', '', 9); pdf.multi_cell(0, 5, 'The image above shows a snapshot from the scan with detected objects highlighted by bounding boxes.')
    pdf.chapter_title('4. Disease Detection Summary')
    summary = get_detection_summary(); total_detections = len(detection_log)
    pdf.set_font('Arial', 'B', 10); pdf.set_fill_color(76, 175, 80); pdf.set_text_color(255, 255, 255)
    pdf.cell(80, 8, 'Disease Type', 1, 0, 'C', fill=True); pdf.cell(40, 8, 'Count', 1, 0, 'C', fill=True); pdf.cell(40, 8, '% Affected', 1, 1, 'C', fill=True)
    pdf.set_font('Arial', '', 10); pdf.set_text_color(0, 0, 0)
    for disease, count in summary.items():
        percentage = (count / total_detections) * 100 if total_detections > 0 else 0
        pdf.cell(80, 8, clean_text(disease), 1, 0); pdf.cell(40, 8, str(count), 1, 0, 'C'); pdf.cell(40, 8, f'{percentage:.2f}%', 1, 1, 'C')
    pdf.add_page(); pdf.chapter_title('5. Verification ID & QR Code'); pdf.set_font('Arial', '', 12); pdf.cell(0, 10, f'Report ID: {report_id}', 0, 1)
    qr_img_path = 'report_qr.png'; qr = qrcode.make(report_id); qr.save(qr_img_path); pdf.image(qr_img_path, x=pdf.w - 50, y=pdf.get_y()-10, w=40)
    try:
        pdf.output(pdf_filename); return pdf_filename
    except Exception as e:
        return None