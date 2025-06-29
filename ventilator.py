import streamlit as st
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks, savgol_filter
from datetime import datetime
import requests
import math

st.set_page_config(page_title="IoT Ventilator UI", layout="wide")

# Hospital Ventilator CSS Styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;700&display=swap');
    
    .main-container {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        color: #00ff41;
        font-family: 'Roboto Mono', monospace;
    }
    
    .ventilator-header {
        background: linear-gradient(90deg, #0f3460 0%, #16537e 100%);
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
        border: 2px solid #00ff41;
        box-shadow: 0 0 20px rgba(0, 255, 65, 0.3);
    }
    
    .ventilator-title {
        font-size: 32px;
        font-weight: bold;
        color: #00ff41;
        text-align: center;
        text-shadow: 0 0 10px rgba(0, 255, 65, 0.5);
        margin: 0;
    }
    
    .status-bar {
        display: flex;
        justify-content: space-between;
        background: #000;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #00ff41;
        margin-bottom: 15px;
    }
    
    .status-item {
        color: #00ff41;
        font-size: 14px;
        font-weight: bold;
    }
    
    .vital-display {
        background: linear-gradient(145deg, #0a0a0a, #1a1a1a);
        border: 2px solid #00ff41;
        border-radius: 15px;
        padding: 20px;
        margin: 10px;
        box-shadow: 0 0 15px rgba(0, 255, 65, 0.2);
        text-align: center;
    }
    
    .vital-value {
        font-size: 48px;
        font-weight: bold;
        color: #00ff41;
        text-shadow: 0 0 10px rgba(0, 255, 65, 0.8);
        margin: 10px 0;
    }
    
    .vital-label {
        font-size: 16px;
        color: #ffffff;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    .vital-unit {
        font-size: 18px;
        color: #cccccc;
    }
    
    .alarm-panel {
        background: #2d1b1b;
        border: 2px solid #ff4444;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    .alarm-text {
        color: #ff4444;
        font-weight: bold;
        font-size: 14px;
    }
    
    .waveform-container {
        background: #000;
        border: 2px solid #00ff41;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        height: 200px;
    }
    
    .control-button {
        background: linear-gradient(145deg, #0f4c75, #3282b8);
        color: white;
        border: 2px solid #00ff41;
        border-radius: 8px;
        padding: 12px 24px;
        font-size: 16px;
        font-weight: bold;
        cursor: pointer;
        box-shadow: 0 0 10px rgba(0, 255, 65, 0.3);
    }
    
    .control-button:hover {
        background: linear-gradient(145deg, #3282b8, #0f4c75);
        box-shadow: 0 0 20px rgba(0, 255, 65, 0.5);
    }
    
    .patient-info {
        background: linear-gradient(145deg, #1a1a2e, #16213e);
        border: 1px solid #00ff41;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    
    .blink {
        animation: blink 1s infinite

;
    }
    
    @keyframes blink {
        0%, 50% { opacity: 1; }
        51%, 100% { opacity: 0.3; }
    }
    
    .collecting-indicator {
        color: #ffaa00;
        font-weight: bold;
        animation: pulse 1.5s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.7; transform: scale(1.05); }
        100% { opacity: 1; transform: scale(1); }
    }
    </style>
""", unsafe_allow_html=True)

# === CONFIG ===
READ_DURATION = 30
TOUCH_THRESHOLD = 100000
API_URL = "https://health-monitor-7lno.onrender.com/latest"

# Generate fake real-time waveforms for display during collection
def generate_fake_ecg(t):
    """Generate realistic ECG waveform"""
    hr = 75  # beats per minute
    ecg = np.zeros_like(t)
    beat_interval = 60 / hr
    
    for beat_time in np.arange(0, t[-1], beat_interval):
        beat_idx = np.argmin(np.abs(t - beat_time))
        if beat_idx < len(t) - 50:
            # P wave
            p_wave = 0.1 * np.exp(-((t[beat_idx:beat_idx+20] - beat_time) / 0.05)**2)
            ecg[beat_idx:beat_idx+20] += p_wave
            
            # QRS complex
            if beat_idx + 30 < len(t):
                qrs = 0.8 * np.exp(-((t[beat_idx+20:beat_idx+30] - (beat_time + 0.12)) / 0.02)**2)
                ecg[beat_idx+20:beat_idx+30] += qrs
            
            # T wave
            if beat_idx + 50 < len(t):
                t_wave = 0.2 * np.exp(-((t[beat_idx+30:beat_idx+50] - (beat_time + 0.25)) / 0.08)**2)
                ecg[beat_idx+30:beat_idx+50] += t_wave
    
    # Add some noise
    ecg += 0.02 * np.random.normal(0, 1, len(t))
    return ecg

def generate_fake_resp(t):
    """Generate realistic respiratory waveform"""
    rr = 18  # breaths per minute
    resp = 0.5 * np.sin(2 * np.pi * rr * t / 60) + 0.1 * np.random.normal(0, 1, len(t))
    return resp

def generate_fake_spo2(t):
    """Generate realistic SpO2 waveform"""
    hr = 75
    spo2 = 0.3 * np.sin(2 * np.pi * hr * t / 60) + 0.05 * np.random.normal(0, 1, len(t))
    return spo2

# ThingSpeak function
def send_to_thingspeak(spo2, rr, hr, name, age, gender):
    try:
        api_key = st.secrets["THINGSPEAK_API_KEY"]
    except KeyError:
        st.error("❌ ThingSpeak API key not found. Please configure it in Streamlit secrets.")
        return
    url = "https://api.thingspeak.com/update"
    payload = {
        "api_key": api_key,
        "field1": rr,
        "field2": spo2,
        "field3": hr,
        "field4": age,
        "field5": gender,
        "field6": name,
    }
    try:
        response = requests.get(url, params=payload, timeout=10)
        if response.status_code == 200:
            st.success("📡 Data pushed to ThingSpeak!")
        else:
            st.error(f"❌ ThingSpeak push failed. Status: {response.status_code}")
    except Exception as e:
        st.error(f"❌ Error sending to ThingSpeak: {e}")

# === FILTERING FUNCTIONS ===
def bandpass_filter(signal, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def preprocess_signal(signal, fs):
    filtered = bandpass_filter(signal, 0.1, 0.8, fs)
    smoothed = savgol_filter(filtered, 101, 3)
    return smoothed

def detect_breath_peaks(filtered_ir, fs):
    peaks, _ = find_peaks(filtered_ir, distance=int(2.5 * fs), prominence=0.2 * np.std(filtered_ir))
    return peaks

def calculate_heart_rate(ir_values, fs):
    ir_filtered = bandpass_filter(ir_values, 0.8, 2.5, fs)
    peaks, _ = find_peaks(ir_filtered, distance=int(0.5 * fs), prominence=0.4 * np.std(ir_filtered))
    duration_sec = len(ir_filtered) / fs
    heart_rate = (len(peaks) / duration_sec) * 60
    heart_rate = round(heart_rate)
    return heart_rate, ir_filtered, peaks

def calculate_spo2(ir_raw, red_raw, fs):
    ir_filtered = bandpass_filter(ir_raw, 0.5, 3.0, fs)
    red_filtered = bandpass_filter(red_raw, 0.5, 3.0, fs)
    peaks, _ = find_peaks(ir_filtered, distance=int(0.6 * fs), prominence=0.02 * max(ir_filtered))
    R_values = []
    for i in range(len(peaks) - 1):
        start = peaks[i]
        end = peaks[i + 1]
        if end - start < 3:
            continue
        ir_seg = ir_raw[start:end]
        red_seg = red_raw[start:end]
        ir_filt_seg = ir_filtered[start:end]
        red_filt_seg = red_filtered[start:end]
        AC_ir = np.max(ir_filt_seg) - np.min(ir_filt_seg)
        DC_ir = np.mean(ir_seg)
        AC_red = np.max(red_filt_seg) - np.min(red_filt_seg)
        DC_red = np.mean(red_seg)
        if DC_ir == 0 or DC_red == 0:
            continue
        R = (AC_red / DC_red) / (AC_ir / DC_ir)
        R_values.append(R)
    if not R_values:
        return 0.0, []
    R_avg = np.mean(R_values)
    spo2 = 104 - 17 * R_avg
    spo2 = max(0, min(100, spo2))
    return spo2, R_values

def read_ir_data():
    st.write(f"🔌 Fetching from: {API_URL}")
    ir_values = []
    red_values = []
    touched = False
    SAMPLE_INTERVAL = 1  # seconds
    retries = 0
    start_time = time.time()
    progress_bar = st.progress(0)
    status_placeholder = st.empty()

    while time.time() - start_time < READ_DURATION:
        try:
            res = requests.get(API_URL, timeout=5)
            if res.status_code != 200:
                status_placeholder.warning(f"Server error. Retry {retries + 1}/3...")
                retries += 1
                if retries > 3:
                    status_placeholder.error("❌ Maximum retries reached. Aborting.")
                    return [], []
                time.sleep(SAMPLE_INTERVAL)
                continue

            data = res.json().get("data", [])
            for sample in data:
                try:
                    ir, red = sample
                except (ValueError, TypeError):
                    continue

                # Wait for finger touch detection
                if not touched and ir > TOUCH_THRESHOLD:
                    status_placeholder.write("✋ Finger detected. Starting 30-second recording...")
                    touched = True
                    start_time = time.time()

                if touched:
                    ir_values.append(ir)
                    red_values.append(red)

            # Update progress
            progress = min((time.time() - start_time) / READ_DURATION, 1.0)
            progress_bar.progress(progress)
            status_placeholder.markdown(f"""
            <div class='collecting-indicator' style='text-align: center; font-size: 18px;'>
                📊 COLLECTING DATA... {progress*100:.0f}% COMPLETE
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            status_placeholder.error(f"❌ Error fetching: {e}")
            time.sleep(SAMPLE_INTERVAL)
        
        time.sleep(SAMPLE_INTERVAL)

    if ir_values and red_values:
        st.success(f"✅ Collected {len(ir_values)} samples in {READ_DURATION} seconds.")
    else:
        st.error("❌ No valid data collected.")
    return ir_values, red_values

def display_real_time_waveforms(time_offset=0):
    """Display animated waveforms during data collection with real-time fluctuation"""
    current_time = datetime.now().strftime("%H:%M:%S")
    window_duration = 8  # seconds
    t = np.linspace(time_offset, time_offset + window_duration, 800)
    
    ecg = generate_fake_ecg(t)
    resp = generate_fake_resp(t)
    spo2_wave = generate_fake_spo2(t)
    
    variation_factor = 0.1 + 0.05 * np.sin(time_offset * 0.5)
    ecg *= (1 + variation_factor)
    resp *= (1 + 0.15 * np.sin(time_offset * 0.3))
    spo2_wave *= (1 + 0.08 * np.cos(time_offset * 0.7))
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8))
    fig.patch.set_facecolor('black')
    
    ax1.plot(t, ecg, color='#00ff41', linewidth=2.5)
    ax1.set_ylabel('ECG (mV)', color='white', fontsize=12, fontweight='bold')
    ax1.set_facecolor('black')
    ax1.grid(True, alpha=0.3, color='#00ff41')
    ax1.tick_params(colors='white')
    ax1.set_title(f'ELECTROCARDIOGRAM - {current_time}', color='#00ff41', fontweight='bold')
    ax1.set_xlim(time_offset, time_offset + window_duration)
    ax1.set_ylim(-0.6, 1.2)
    
    ax2.plot(t, resp, color='#ffaa00', linewidth=2.5)
    ax2.set_ylabel('RESP (L/min)', color='white', fontsize=12, fontweight='bold')
    ax2.set_facecolor('black')
    ax2.grid(True, alpha=0.3, color='#ffaa00')
    ax2.tick_params(colors='white')
    ax2.set_title('RESPIRATORY WAVEFORM', color='#ffaa00', fontweight='bold')
    ax2.set_xlim(time_offset, time_offset + window_duration)
    ax2.set_ylim(-0.8, 0.8)
    
    ax3.plot(t, spo2_wave, color='#ff4444', linewidth=2.5)
    ax3.set_ylabel('SpO₂ (%)', color='white', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Time (seconds)', color='white', fontsize=12)
    ax3.set_facecolor('black')
    ax3.grid(True, alpha=0.3, color='#ff4444')
    ax3.tick_params(colors='white')
    ax3.set_title('PULSE OXIMETRY', color='#ff4444', fontweight='bold')
    ax3.set_xlim(time_offset, time_offset + window_duration)
    ax3.set_ylim(-0.5, 0.5)
    
    plt.tight_layout()
    return fig

def main():
    # Initialize session state
    if 'show_results' not in st.session_state:
        st.session_state.show_results = False
    if 'results_data' not in st.session_state:
        st.session_state.results_data = {}
    if 'patient_data' not in st.session_state:
        st.session_state.patient_data = {'name': '', 'age': '', 'gender': 'Male'}

    # Header
    st.markdown("""
    <div class='ventilator-header'>
        <h1 class='ventilator-title'>RVCE HEALTHSTATION</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Status bar
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.markdown(f"""
    <div class='status-bar'>
        <div class='status-item'>⏰ {current_time}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar - Patient Information Panel
    with st.sidebar:
        st.markdown("""
        <div class='patient-info'>
            <h2 style='color: #00ff41; text-align: center;'>👤 PATIENT DATA</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.session_state.patient_data['name'] = st.text_input("Patient Name", value=st.session_state.patient_data['name'], placeholder="Enter full name")
        st.session_state.patient_data['age'] = st.text_input("Age", value=st.session_state.patient_data['age'], placeholder="Enter age")
        st.session_state.patient_data['gender'] = st.selectbox("Gender", ["Male", "Female", "Other"], index=["Male", "Female", "Other"].index(st.session_state.patient_data['gender']))
        
        st.markdown("---")
        st.markdown("### 🔧 SYSTEM CONTROLS")
        
        if st.button("🔄 NEW SESSION", help="Clear buffer and start fresh"):
            try:
                requests.post("https://health-monitor-7lno.onrender.com/reset", timeout=5)
                st.session_state.show_results = False
                st.session_state.results_data = {}
                st.session_state.patient_data = {'name': '', 'age': '', 'gender': 'Male'}
                st.success("🧹 Buffer cleared! Ready for new patient.")
            except Exception as e:
                st.warning(f"⚠ Could not reach the Flask reset endpoint: {e}")

    # Main control button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("▶️ START MONITORING", key="start_btn", help="Begin data collection and analysis"):
            st.session_state.show_results = False
            st.markdown("""
            <div class='collecting-indicator' style='text-align: center; font-size: 24px; margin: 20px 0;'>
                🔄 COLLECTING PATIENT DATA... PLEASE WAIT
            </div>
            """, unsafe_allow_html=True)
            
            # Display real-time waveforms during collection
            st.markdown("### REAL-TIME MONITORING")
            waveform_placeholder = st.empty()
            
            # Collect data
            ir_values, red_values = read_ir_data()
            
            if ir_values and red_values:
                # Perform calculations
                fs = len(ir_values) / READ_DURATION
                filtered_ir = preprocess_signal(ir_values, fs)
                peaks = detect_breath_peaks(filtered_ir, fs)
                rr = math.ceil(len(peaks) * (60 / READ_DURATION))
                hr, _, _ = calculate_heart_rate(ir_values, fs)
                spo2, _ = calculate_spo2(np.array(ir_values), np.array(red_values), fs)
                
                # Store results in session state
                st.session_state.results_data = {
                    'rr': rr, 'hr': hr, 'spo2': spo2,
                    'ir_values': ir_values, 'red_values': red_values,
                    'filtered_ir': filtered_ir, 'peaks': peaks
                }
                st.session_state.show_results = True
                st.success("✅ Data collection completed successfully!")
                
                # Display final waveforms
                with waveform_placeholder.container():
                    fig = display_real_time_waveforms(time.time() - time.time())
                    st.pyplot(fig)
                    plt.close(fig)
            else:
                st.error("❌ Data collection failed. Please try again.")
                return
    
    # Display results if available
    if st.session_state.show_results and st.session_state.results_data:
        data = st.session_state.results_data
        
        # Vital Signs Display
        st.markdown("### 💓 VITAL SIGNS MONITOR")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class='vital-display'>
                <div class='vital-label'>RESPIRATORY RATE</div>
                <div class='vital-value'>{data['rr']:.0f}</div>
                <div class='vital-unit'>breaths/min</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class='vital-display'>
                <div class='vital-label'>HEART RATE</div>
                <div class='vital-value'>{data['hr']}</div>
                <div class='vital-unit'>bpm</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            color = "#00ff41" if data['spo2'] >= 75 else "#ff4444"
            st.markdown(f"""
            <div class='vital-display'>
                <div class='vital-label'>SpO₂ SATURATION</div>
                <div class='vital-value' style='color: {color}'>{data['spo2']:.1f}</div>
                <div class='vital-unit'>%</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Connection Status Panel
        st.markdown("""
        <div style='background: #0a3d0a; border: 2px solid #00ff41; border-radius: 10px; padding: 15px; margin: 10px 0; text-align: center;'>
            <div style='color: #00ff41; font-weight: bold; font-size: 16px;'>✅ SENSOR CONNECTION: ACTIVE</div>
            <div style='color: #ffffff; font-size: 14px; margin-top: 5px;'>👆 Finger detected and data processed successfully</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Waveform Analysis
        st.markdown("### 📊 SIGNAL ANALYSIS")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(10, 6))
            fig.patch.set_facecolor('black')
            ax.set_facecolor('black')
            ax.plot(data['filtered_ir'], color='#00ff41', linewidth=2, label='Respiratory Signal')
            ax.plot(data['peaks'], np.array(data['filtered_ir'])[data['peaks']], 'ro', 
                   markersize=8, label='Detected Breaths')
            ax.set_xlabel('Sample', color='white')
            ax.set_ylabel('Amplitude', color='white')
            ax.set_title('RESPIRATORY WAVEFORM ANALYSIS', color='#00ff41', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3, color='#00ff41')
            ax.tick_params(colors='white')
            st.pyplot(fig)
            plt.close(fig)
        
        with col2:
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            fig2.patch.set_facecolor('black')
            ax2.set_facecolor('black')
            ax2.plot(data['ir_values'], color='#ff4444', linewidth=2, alpha=0.8, label='IR Signal')
            ax2.plot(data['red_values'], color='#ffaa00', linewidth=2, alpha=0.8, label='RED Signal')
            ax2.set_xlabel('Sample', color='white')
            ax2.set_ylabel('Amplitude', color='white')
            ax2.set_title('RAW SENSOR DATA', color='#ff4444', fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3, color='#ff4444')
            ax2.tick_params(colors='white')
            st.pyplot(fig2)
            plt.close(fig2)
        
        # Send to ThingSpeak
        if st.button("📡 TRANSMIT TO CLOUD"):
            send_to_thingspeak(
                data['spo2'], 
                data['rr'], 
                data['hr'], 
                st.session_state.patient_data['name'], 
                st.session_state.patient_data['age'], 
                st.session_state.patient_data['gender']
            )

if __name__ == "__main__":
    main()
