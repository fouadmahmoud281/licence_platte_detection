import streamlit as st
import cv2
import numpy as np
from PIL import Image
import arabic_reshaper
from bidi.algorithm import get_display
from inference_sdk import InferenceHTTPClient
import sqlite3
import datetime
import pandas as pd
import plotly.express as px
import os

def load_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap');
        
        :root {
            --police-blue: #1a3263;
            --police-gold: #c4a777;
            --alert-red: #dc3545;
            --success-green: #28a745;
        }
        
        .stApp {
            background-color: #f0f2f6;
            font-family: 'Roboto', sans-serif;
        }
        
        .main-header {
            background: linear-gradient(90deg, var(--police-blue), #2a4273);
            padding: 2rem;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        .card {
            background-color: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            margin-bottom: 1rem;
        }
        
        .stButton>button {
            background-color: var(--police-blue);
            color: white;
            font-weight: 500;
            padding: 0.5rem 1rem;
            border-radius: 5px;
            border: none;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            transition: all 0.3s ease;
        }
        
        .stButton>button:hover {
            background-color: #2a4273;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        }
        
        .stTextInput>div>div>input {
            border-radius: 5px;
            border: 2px solid #e1e4e8;
            padding: 0.5rem;
        }
        
        .stTextInput>div>div>input:focus {
            border-color: var(--police-blue);
            box-shadow: 0 0 0 2px rgba(26, 50, 99, 0.1);
        }
        
        .alert {
            padding: 1rem;
            border-radius: 5px;
            margin: 1rem 0;
            display: flex;
            align-items: center;
            font-weight: 500;
        }
        
        .alert-danger {
            background-color: #ffeaea;
            border-left: 4px solid var(--alert-red);
            color: var(--alert-red);
        }
        
        .alert-success {
            background-color: #eafaf1;
            border-left: 4px solid var(--success-green);
            color: var(--success-green);
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
            background-color: #f8f9fa;
            padding: 0.5rem;
            border-radius: 10px;
        }
        
        .stTabs [data-baseweb="tab"] {
            background-color: var(--police-blue);
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 5px;
        }
        
        .metric-card {
            background: white;
            padding: 1rem;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        
        .metric-value {
            font-size: 2rem;
            font-weight: bold;
            color: var(--police-blue);
        }
        
        .status-badge {
            padding: 0.25rem 0.75rem;
            border-radius: 15px;
            font-size: 0.875rem;
            font-weight: 500;
        }
        
        .status-alert {
            background-color: var(--alert-red);
            color: white;
        }
        
        .status-clear {
            background-color: var(--success-green);
            color: white;
        }
        
        .data-table {
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
        }
        
        .data-table th, .data-table td {
            padding: 0.75rem;
            border-bottom: 1px solid #e1e4e8;
            text-align: left;
        }
        
        .data-table th {
            background-color: #f8f9fa;
            font-weight: 500;
        }
        
        .search-box {
            background-color: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            margin-bottom: 2rem;
        }
        
        .chart-container {
            background-color: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            margin-bottom: 1rem;
        }
        .stolen-vehicle-card {
            background-color: #fff3f3;
            border-left: 5px solid #dc3545;
            margin: 20px 0;
            padding: 15px;
        }
        
        .detection-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        
        .detection-details {
            margin-left: 20px;
        }
        
        .status-badge {
            padding: 5px 10px;
            border-radius: 15px;
            color: white;
            font-weight: bold;
        }
        
        .alert-danger {
            background-color: #dc3545;
            color: white;
            padding: 15px;
            border-radius: 5px;
            text-align: center;
            margin: 20px 0;
        }
        .detection-card {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        
        .report-card {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .alert {
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
        }
        
        .alert-danger {
            background-color: #dc3545;
            color: white;
        }
        
        .alert-success {
            background-color: #28a745;
            color: white;
        }
        
        .alert h4 {
            margin-top: 0;
            margin-bottom: 10px;
        }
        
        .alert-content {
            background-color: rgba(255,255,255,0.1);
            padding: 10px;
            border-radius: 5px;
            margin-top: 10px;
        }
        
        .emergency-contact {
            background-color: #ff9800;
            color: white;
            padding: 15px;
            border-radius: 5px;
            margin-top: 20px;
        }
        
        .emergency-contact h4 {
            margin-top: 0;
            margin-bottom: 10px;
        }
        </style>
    """, unsafe_allow_html=True)
def get_api_key():
    # First check if API key exists in session state
    if 'roboflow_api_key' not in st.session_state:
        st.session_state['roboflow_api_key'] = None
    
    # If no API key, show input in sidebar
    if not st.session_state['roboflow_api_key']:
        with st.sidebar:
            st.markdown("### 🔑 API Configuration")
            api_key = st.text_input(
                "Enter your Roboflow API Key",
                type="password",
                help="Enter your Roboflow API key to use the detection features"
            )
            if api_key:
                st.session_state['roboflow_api_key'] = api_key


def reset_database():
    import os
    try:
        if os.path.exists('license_plates.db'):
            os.remove('license_plates.db')
            print("Database reset successfully")
    except Exception as e:
        print(f"Error resetting database: {e}")

def init_db():
    conn = sqlite3.connect('license_plates.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS detected_plates
                 (timestamp TEXT,
                  plate_number TEXT,
                  status TEXT,
                  location TEXT,
                  officer_id TEXT,
                  incident_type TEXT,
                  notes TEXT)''')
    conn.commit()
    return conn

def init_stolen_vehicles_db():
    conn = sqlite3.connect('license_plates.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS stolen_vehicles
                 (plate_number TEXT PRIMARY KEY,
                  vehicle_make TEXT,
                  vehicle_model TEXT,
                  vehicle_color TEXT,
                  date_reported TEXT,
                  additional_details TEXT)''')
    conn.commit()
    return conn

def populate_sample_stolen_vehicles():
    conn = init_stolen_vehicles_db()
    c = conn.cursor()
    
    sample_vehicles = [
        ('ب ع د 1234', 'Toyota', 'Corolla', 'White', '2024-02-15', 'Stolen from Cairo downtown'),
        ('س ي ر 5678', 'Honda', 'Civic', 'Black', '2024-02-16', 'Armed carjacking in Alexandria'),
        ('م ن ل 9012', 'Hyundai', 'Elantra', 'Silver', '2024-02-17', 'Stolen from parking lot'),
        ('ك ت ب 3456', 'Nissan', 'Sunny', 'Red', '2024-02-18', 'Home invasion theft'),
        ('ط ر ق 7890', 'Kia', 'Cerato', 'Blue', '2024-02-19', 'Keys stolen during break-in')
    ]
    
    try:
        c.executemany('''INSERT OR REPLACE INTO stolen_vehicles 
                        (plate_number, vehicle_make, vehicle_model, vehicle_color, 
                         date_reported, additional_details)
                        VALUES (?, ?, ?, ?, ?, ?)''', sample_vehicles)
        conn.commit()
    except Exception as e:
        print(f"Error populating stolen vehicles: {e}")
    finally:
        conn.close()
def process_arabic_characters(plate_img):
    # Check if API key exists
    if not st.session_state.get('roboflow_api_key'):
        st.error("⚠️ Please enter your Roboflow API key in the sidebar")
        st.stop()
        
    client = InferenceHTTPClient(
        api_url="https://classify.roboflow.com",
        api_key=st.session_state.roboflow_api_key
    )

    try:
        temp_path = 'temp_plate.jpg'
        cv2.imwrite(temp_path, plate_img)
        result = client.infer(temp_path, model_id="arabic-character-detection/1")

        if os.path.exists(temp_path):
            os.remove(temp_path)

        if 'predictions' in result and 'predicted_classes' in result:
            chars_with_conf = []
            for char in result['predicted_classes']:
                confidence = result['predictions'][char]['confidence']
                chars_with_conf.append((char, confidence))

            numbers = []
            letters = []

            for char, conf in chars_with_conf:
                if char.isdigit() or char in '٠١٢٣٤٥٦٧٨٩':
                    numbers.append(char)
                else:
                    letters.append(char)

            formatted_text = ''.join(letters) + ' ' + ''.join(numbers)
            return formatted_text.strip()

    except Exception as e:
        st.error(f"Error in character detection: {str(e)}")
        return ''

def process_plate_detection(image, conf_threshold=0.3):
    # Check if API key exists
    if not st.session_state.get('roboflow_api_key'):
        st.error("⚠️ Please enter your Roboflow API key in the sidebar")
        st.stop()
        
    client = InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key=st.session_state.roboflow_api_key
    )


    
    file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    result = client.infer(img, model_id="egyptian-cars-database/1")
    detected_plates = []
    annotated_img = img.copy()
    
    if 'predictions' in result:
        for prediction in result['predictions']:
            if prediction['confidence'] > conf_threshold:
                x1 = int(prediction['x'] - prediction['width'] / 2)
                y1 = int(prediction['y'] - prediction['height'] / 2)
                x2 = int(prediction['x'] + prediction['width'] / 2)
                y2 = int(prediction['y'] + prediction['height'] / 2)
                
                padding = 10
                plate_img = img[max(0, y1-padding):min(img.shape[0], y2+padding),
                              max(0, x1-padding):min(img.shape[1], x2+padding)]
                
                if plate_img.size > 0:
                    plate_img = cv2.resize(plate_img, None, fx=2, fy=2, 
                                         interpolation=cv2.INTER_CUBIC)
                    
                    lab = cv2.cvtColor(plate_img, cv2.COLOR_BGR2LAB)
                    l, a, b = cv2.split(lab)
                    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                    cl = clahe.apply(l)
                    enhanced = cv2.merge((cl,a,b))
                    plate_img = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
                    
                    plate_text = process_arabic_characters(plate_img)
                    
                    if plate_text:
                        detected_plates.append(plate_text)
                        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), 
                                    (0, 255, 0), 2)
                        cv2.putText(annotated_img, plate_text, (x1, y1-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return detected_plates, annotated_img

def main():
    st.set_page_config(
        page_title="Law Enforcement License Plate Detection",
        page_icon="🚓",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    load_css()
    init_db()
    init_stolen_vehicles_db()
    populate_sample_stolen_vehicles()
    get_api_key()
    # Check if API key is set
    if not st.session_state.get('roboflow_api_key'):
        st.warning("⚠️ Please enter your Roboflow API key in the sidebar to use detection features")

    st.markdown("""
        <div class="main-header">
            <h1>🚔 Law Enforcement License Plate Detection System</h1>
            <p>Advanced Vehicle Identification and Tracking System</p>
        </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3,tab4 = st.tabs([
        "🎯 Plate Detection",
        "🔍 Database Search",
        "📊 Analytics",
        "🚨 Stolen Vehicles"

    ])
    
    with tab1:
        detection_page()
    with tab2:
        search_page()
    with tab3:
        analytics_page()
    with tab4:
        stolen_vehicles_page()

def detection_page():
    st.markdown("### 🚓 Vehicle Detection System")
    
    # Create two columns with adjusted ratios
    col1, col2 = st.columns([2,1])
    
    with col1:
        st.markdown('<div class="detection-card">', unsafe_allow_html=True)
        st.markdown("#### 📸 Upload Vehicle Image")
        uploaded_file = st.file_uploader(
            "Drag and drop image here",
            type=['jpg', 'jpeg', 'png']
        )
        
        if uploaded_file:
            with st.spinner('Processing image...'):
                detected_plates, annotated_img = process_plate_detection(uploaded_file)
                
                # Display processed image
                st.image(annotated_img, caption="Processed Image", use_column_width=True)
                
                # Process each detected plate
                for plate in detected_plates:
                    stolen_info = check_stolen_vehicle(plate)
                    
                    if stolen_info:
                        st.markdown(f"""
                            <div class="alert alert-danger">
                                <h4>🚨 STOLEN VEHICLE DETECTED 🚨</h4>
                                <div class="alert-content">
                                    <p><strong>Plate Number:</strong> {stolen_info[0]}</p>
                                    <p><strong>Vehicle:</strong> {stolen_info[1]} {stolen_info[2]} ({stolen_info[3]})</p>
                                    <p><strong>Reported:</strong> {stolen_info[4]}</p>
                                    <p><strong>Details:</strong> {stolen_info[5]}</p>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                            <div class="alert alert-success">
                                <h4>✅ VEHICLE CLEAR</h4>
                                <p>Plate Number: {plate}</p>
                            </div>
                        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="report-card">', unsafe_allow_html=True)
        st.markdown("#### 📝 Incident Report")
        
        # Create the incident report form
        with st.form("incident_report_form"):
            location = st.text_input("📍 Location", 
                placeholder="Enter detection location")
            
            officer_id = st.text_input("👮 Officer ID",
                placeholder="Enter your ID")
            
            incident_type = st.selectbox("🏷️ Incident Type",
                ["Routine Check", "Suspicious Vehicle", "Traffic Stop", 
                 "Parking Violation", "Other"])
            
            if incident_type == "Other":
                incident_type_other = st.text_input("Specify Incident Type")
            
            notes = st.text_area("📋 Notes",
                placeholder="Enter any additional observations or notes")
            
            submit_report = st.form_submit_button("🚨 Submit Report")
        
        if submit_report:
            if 'detected_plates' in locals():
                if location and officer_id:
                    with st.spinner('Saving report...'):
                        # Prepare incident type
                        final_incident_type = incident_type_other if incident_type == "Other" else incident_type
                        
                        # Save report
                        save_incident_report(
                            plates=detected_plates,
                            location=location,
                            officer_id=officer_id,
                            incident_type=final_incident_type,
                            notes=notes
                        )
                        
                        st.success("✅ Report submitted successfully!")
                        
                        # If any plates were stolen, show emergency contact info
                        if any(check_stolen_vehicle(plate) for plate in detected_plates):
                            st.markdown("""
                                <div class="emergency-contact">
                                    <h4>🚔 Emergency Contacts</h4>
                                    <p>Police Command Center: <strong>911</strong></p>
                                    <p>Vehicle Theft Unit: <strong>555-0123</strong></p>
                                </div>
                            """, unsafe_allow_html=True)
                else:
                    st.error("Please fill in all required fields")
            else:
                st.warning("⚠️ Please detect a license plate first")
        
        st.markdown('</div>', unsafe_allow_html=True)

def save_incident_report(plates, location, officer_id, incident_type, notes):
    """Save incident report to database with enhanced information"""
    conn = init_db()
    c = conn.cursor()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    try:
        for plate in plates:
            status = "ALERT" if check_stolen_vehicle(plate) else "CLEAR"
            c.execute("""
                INSERT INTO detected_plates 
                (timestamp, plate_number, status, location, officer_id, 
                 incident_type, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (timestamp, plate, status, location, officer_id, 
                  incident_type, notes))
        
        conn.commit()
    except Exception as e:
        st.error(f"Error saving report: {str(e)}")
    finally:
        conn.close()


def search_page():
    st.markdown('<div class="search-box">', unsafe_allow_html=True)
    st.markdown("### 🔍 Search Vehicle Records")
    
    # Create two columns for search options
    col1, col2 = st.columns(2)
    with col1:
        search_query = st.text_input("Enter License Plate Number")
    with col2:
        search_type = st.selectbox(
            "Search Type",
            ["All Records", "Stolen Vehicles Only", "Detection History Only"]
        )
    
    if search_query:
        with st.spinner('Searching database...'):
            # Initialize detection_results as empty list
            detection_results = []
            
            # Check stolen vehicles database first
            stolen_info = check_stolen_vehicle(search_query)
            
            if stolen_info and (search_type in ["All Records", "Stolen Vehicles Only"]):
                st.markdown("""
                    <div class="alert alert-danger">
                        🚨 <strong>STOLEN VEHICLE ALERT</strong> 🚨
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown('<div class="card stolen-vehicle-card">', unsafe_allow_html=True)
                st.markdown(f"""
                    <div class="data-row">
                        <h4>Stolen Vehicle Details</h4>
                        <p><strong>Plate Number:</strong> {stolen_info[0]}</p>
                        <p><strong>Vehicle:</strong> {stolen_info[1]} {stolen_info[2]}</p>
                        <p><strong>Color:</strong> {stolen_info[3]}</p>
                        <p><strong>Date Reported:</strong> {stolen_info[4]}</p>
                        <p><strong>Additional Details:</strong> {stolen_info[5]}</p>
                    </div>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Search detection history if needed
            if search_type in ["All Records", "Detection History Only"]:
                detection_results = search_detection_history(search_query)
                if detection_results:
                    st.markdown("#### Recent Detection History")
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    for result in detection_results:
                        timestamp = format_timestamp(result[0])
                        status_color = "red" if result[2] == "ALERT" else "green"
                        st.markdown(f"""
                            <div class="data-row">
                                <div class="detection-header">
                                    <h4>📅 {timestamp}</h4>
                                    <span class="status-badge" style="background-color: {status_color}">
                                        {result[2]}
                                    </span>
                                </div>
                                <div class="detection-details">
                                    <p><strong>Location:</strong> {result[3]}</p>
                                    <p><strong>Officer ID:</strong> {result[4]}</p>
                                    <p><strong>Notes:</strong> {result[5]}</p>
                                </div>
                            </div>
                            <hr>
                        """, unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
            
            if not stolen_info and not detection_results:
                st.info("No records found for this plate number")
    
    st.markdown('</div>', unsafe_allow_html=True)


def check_stolen_vehicle(plate):
    """Check if a vehicle is in the stolen vehicles database"""
    conn = sqlite3.connect('license_plates.db')
    c = conn.cursor()
    try:
        c.execute("""
            SELECT plate_number, vehicle_make, vehicle_model, 
                   vehicle_color, date_reported, additional_details 
            FROM stolen_vehicles 
            WHERE plate_number = ?
        """, (plate,))
        return c.fetchone()
    finally:
        conn.close()

def search_detection_history(plate):
    """Search for plate detection history"""
    conn = sqlite3.connect('license_plates.db')
    c = conn.cursor()
    try:
        c.execute("""
            SELECT timestamp, plate_number, status, location, officer_id, notes
            FROM detected_plates 
            WHERE plate_number LIKE ? 
            ORDER BY timestamp DESC
            LIMIT 10
        """, (f"%{plate}%",))
        return c.fetchall()
    finally:
        conn.close()


def format_timestamp(timestamp):
    """Format timestamp for display"""
    try:
        dt = datetime.datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
        return dt.strftime("%B %d, %Y at %I:%M %p")
    except:
        return timestamp


def analytics_page():
    st.markdown("### 📊 Analytics Dashboard")
    
    conn = init_db()
    df = pd.read_sql_query("SELECT * FROM detected_plates", conn)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        total_scans = len(df)
        st.metric("Total Scans", f"{total_scans:,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        alerts = len(df[df['status'] == 'ALERT'])
        st.metric("Total Alerts", f"{alerts:,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        locations = df['location'].nunique()
        st.metric("Unique Locations", f"{locations:,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        df['date'] = pd.to_datetime(df['timestamp']).dt.date
        daily_counts = df.groupby('date').size().reset_index(name='counts')
        fig1 = px.line(daily_counts, x='date', y='counts', 
                      title="Daily Detection Trend",
                      labels={'date': 'Date', 'counts': 'Number of Detections'})
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        status_counts = df['status'].value_counts()
        fig2 = px.pie(values=status_counts.values, names=status_counts.index,
                      title="Status Distribution")
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
def check_plate_status(plate):
    conn = sqlite3.connect('license_plates.db')
    c = conn.cursor()
    try:
        c.execute("SELECT * FROM stolen_vehicles WHERE plate_number = ?", (plate,))
        result = c.fetchone()
        if result:
            return "ALERT"
        return "CLEAR"
    finally:
        conn.close()


def save_to_database(plates, location, officer_id, notes):
    conn = init_db()
    c = conn.cursor()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    try:
        for plate in plates:
            status = check_plate_status(plate)
            c.execute("""
                INSERT INTO detected_plates 
                (timestamp, plate_number, status, location, officer_id, notes, image_path)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (timestamp, plate, status, location, officer_id, notes, ""))
        
        conn.commit()
        st.success("Data saved successfully!")
    except Exception as e:
        st.error(f"Error saving to database: {str(e)}")
    finally:
        conn.close()

def search_database(query):
    conn = init_db()
    c = conn.cursor()
    try:
        c.execute("""
            SELECT * FROM detected_plates 
            WHERE plate_number LIKE ? 
            ORDER BY timestamp DESC
        """, (f"%{query}%",))
        results = c.fetchall()
        return results
    except Exception as e:
        st.error(f"Error searching database: {str(e)}")
        return []
    finally:
        conn.close()

def get_analytics_data():
    conn = init_db()
    try:
        df = pd.read_sql_query("""
            SELECT 
                date(timestamp) as date,
                COUNT(*) as total_scans,
                SUM(CASE WHEN status = 'ALERT' THEN 1 ELSE 0 END) as alerts,
                COUNT(DISTINCT location) as unique_locations
            FROM detected_plates
            GROUP BY date(timestamp)
            ORDER BY date(timestamp)
        """, conn)
        return df
    except Exception as e:
        st.error(f"Error getting analytics data: {str(e)}")
        return pd.DataFrame()
    finally:
        conn.close()
def stolen_vehicles_page():
    st.markdown("### 🚨 Stolen Vehicles Management")
    
    tab1, tab2 = st.tabs(["View Stolen Vehicles", "Add New Record"])
    
    with tab1:
        display_stolen_vehicles()
    
    with tab2:
        add_stolen_vehicle()

def display_stolen_vehicles():
    conn = sqlite3.connect('license_plates.db')
    df = pd.read_sql_query("SELECT * FROM stolen_vehicles", conn)
    conn.close()
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    if not df.empty:
        st.dataframe(df, use_container_width=True)
        
        # Add export functionality
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download Stolen Vehicles List",
            data=csv,
            file_name="stolen_vehicles.csv",
            mime="text/csv"
        )
    else:
        st.info("No stolen vehicles in the database")
    st.markdown('</div>', unsafe_allow_html=True)

def add_stolen_vehicle():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    with st.form("stolen_vehicle_form"):
        plate_number = st.text_input("License Plate Number")
        col1, col2 = st.columns(2)
        with col1:
            vehicle_make = st.text_input("Vehicle Make")
            vehicle_color = st.text_input("Vehicle Color")
        with col2:
            vehicle_model = st.text_input("Vehicle Model")
            date_reported = st.date_input("Date Reported")
        
        additional_details = st.text_area("Additional Details")
        
        submitted = st.form_submit_button("Add to Database")
        
        if submitted:
            if plate_number and vehicle_make and vehicle_model:
                conn = sqlite3.connect('license_plates.db')
                c = conn.cursor()
                try:
                    c.execute('''INSERT INTO stolen_vehicles 
                                (plate_number, vehicle_make, vehicle_model, 
                                 vehicle_color, date_reported, additional_details)
                                VALUES (?, ?, ?, ?, ?, ?)''',
                             (plate_number, vehicle_make, vehicle_model,
                              vehicle_color, date_reported.strftime('%Y-%m-%d'),
                              additional_details))
                    conn.commit()
                    st.success("Vehicle added to stolen vehicles database")
                except sqlite3.IntegrityError:
                    st.error("This license plate is already in the database")
                except Exception as e:
                    st.error(f"Error adding vehicle: {e}")
                finally:
                    conn.close()
            else:
                st.warning("Please fill in all required fields")
    st.markdown('</div>', unsafe_allow_html=True)

def format_timestamp(timestamp):
    dt = datetime.datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
    return dt.strftime("%B %d, %Y %I:%M %p")

def create_downloadable_report(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="plate_detection_report.csv">Download Report</a>'
    return href

def display_error_message(message):
    st.markdown(f"""
        <div class="alert alert-danger">
            ⚠️ {message}
        </div>
    """, unsafe_allow_html=True)

def display_success_message(message):
    st.markdown(f"""
        <div class="alert alert-success">
            ✅ {message}
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")