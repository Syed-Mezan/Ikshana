import os
from flask import Flask, render_template, Response, jsonify, send_from_directory, request
from detect import gen_frames, set_mode, start_detection, stop_detection, \
                  start_recording, stop_recording, generate_pdf_report, tree_count, get_detection_summary, \
                  set_thresholds, set_zoom_state, generate_ai_recommendation

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_mode/<mode>')
def set_mode_route(mode):
    default_conf = set_mode(mode)
    return jsonify({'status': f'Mode set to {mode}', 'default_confidence': default_conf})

@app.route('/set_thresholds', methods=['POST'])
def set_thresholds_route():
    data = request.json
    conf = data.get('confidence')
    iou = data.get('iou')
    set_thresholds(conf, iou)
    return jsonify({'status': 'Thresholds updated'})

@app.route('/toggle_zoom')
def toggle_zoom_route():
    state = request.args.get('state', 'false')
    enabled = state.lower() == 'true'
    set_zoom_state(enabled)
    return jsonify({'status': f'Zoom set to {enabled}'})

# === NEW ROUTE FOR AI RECOMMENDATIONS ===
@app.route('/get_recommendation')
def get_recommendation_route():
    recommendation_text = generate_ai_recommendation()
    return jsonify({'recommendation': recommendation_text})

@app.route('/start')
def start():
    start_detection()
    return "✅ Detection started"

@app.route('/stop')
def stop():
    stop_detection()
    return "⏹️ Detection stopped"

@app.route('/start_recording')
def start_rec():
    start_recording()
    return "🔴 Recording started"

@app.route('/stop_recording')
def stop_rec():
    stop_recording()
    return "💾 Recording stopped and saved"

@app.route('/tree_count')
def get_tree_count():
    return str(tree_count)
    
@app.route('/detection_summary')
def detection_summary():
    summary = get_detection_summary()
    return jsonify(summary)

@app.route('/download_pdf')
def download_pdf():
    location = request.args.get('location', 'Unknown Location')
    crop_type = request.args.get('crop', 'Unknown Crop')
    farm_area = request.args.get('area', '0')
    operator_name = request.args.get('operator', 'N/A')
    report_filename = generate_pdf_report(location, crop_type, farm_area, operator_name)
    
    if report_filename:
        try:
            root_dir = os.path.dirname(os.path.abspath(__file__))
            return send_from_directory(root_dir, report_filename, as_attachment=True)
        except FileNotFoundError:
            return "Error: Report file not found on server.", 404
    else:
        return "Could not generate PDF report. No detections have been logged yet.", 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)