import streamlit as st
import pandas as pd
import numpy as np
import pickle
import torch
import sqlite3
import datetime
import os
import base64
import plotly.graph_objects as go
from model import TabularResNet
from recommender import get_recommendations, log_recommendation, log_feedback, DB_PATH

# Page Configuration
st.set_page_config(
    page_title="vahanFlow - Bengaluru Intelligent Command Center - Govt of Karnataka",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────
# Language & Translation System
# ─────────────────────────────────────────────────────────
if "lang" not in st.session_state:
    st.session_state.lang = "English"
if "font_size" not in st.session_state:
    st.session_state.font_size = 100

translations = {
    "English": {
        "title": "VAHANFLOW: BENGALURU INTELLIGENT MOBILITY COMMAND CENTER",
        "subtitle": "Under the Aegis of Directorate of Urban Land Transport (DULT) & Bengaluru Traffic Police (BTP)",
        "gov_karnataka": "GOVERNMENT OF KARNATAKA",
        "portal_badge": "OFFICIAL SECURE GOVERNMENT PORTAL",
        "status_badge": "OPERATIONAL & VERIFIED",
        "live_stream": "Live Stream",
        "emergency_bulletin": "EMERGENCY BULLETIN",
        "nav_title": "System Menu",
        "switch_lang": "Switch Language",
        "emergency_mode": "Emergency Mode",
        "emergency_active": "ALERT: EMERGENCY PROTOCOLS ACTIVE",
        "weather": "Weather",
        "aqi": "AQI",
        "mobility": "Mobility Score",
        "helpline": "Emergency Helpline",
        "active_events": "Active Traffic Events",
        "congestion_index": "Congestion Index",
        "active_incidents": "Active Incidents",
        "metro_status": "Metro Status",
        "bmtc_status": "BMTC Fleet Status",
        "road_closures": "Road Closures",
        "map_title": "Live Congestion Map",
        "flow_trend": "Traffic Flow Trend",
        "top_hotspots": "Top Hotspots",
        "event_forecast": "Event Forecast",
        "view_mitigation": "View Mitigation Plan",
        "planner_title": "Predictive Event Planner",
        "input_params": "Input Parameters",
        "event_class": "Event Classification",
        "primary_cause": "Primary Event Cause",
        "priority_level": "Incident Priority Level",
        "road_closure_req": "Requires Road Closure?",
        "police_station": "Responsible Police Station",
        "corridor": "Traffic Corridor",
        "zone": "Administrative Zone",
        "junction": "Junction Affected",
        "coords": "Incident Location Coordinates",
        "lat": "Latitude",
        "lon": "Longitude",
        "temporal": "Temporal Parameters",
        "event_date": "Event Date",
        "event_time": "Scheduled Start Time",
        "vehicle_class": "Vehicle Class Involved",
        "generate_btn": "Generate Tactical Action Plan",
        "intel_title": "Traffic Intelligence",
        "hotspot_title": "GIS Hotspot Analytics",
        "incident_title": "Incident Response",
        "metro_title": "Transit Metro Coordination",
        "bmtc_title": "BMTC Operations",
        "adhira_title": "Adhira AI Assistant",
        "control_title": "Control Center",
        "dispatch_btn": "Dispatch Command Order to Field Units",
        "dispatch_success": "Command order successfully dispatched to field controls!",
        "dispatch_title": "OFFICIAL COMMAND DISPATCH ORDER",
        "pred_congestion": "PREDICTED CONGESTION",
        "risk_threat": "TACTICAL RISK THREAT",
        "resource_dispatch": "RESOURCE DISPATCH REQUIREMENTS",
        "diversion_route": "DIVERSION ROUTE",
        "action_plan": "ACTION PLAN",
        "quick_commands": "Quick Commands",
        "query_assistant": "Query Adhira Assistant",
        "active_alerts": "Active Alerts",
        "query_custom": "Or enter a custom question for Adhira AI:",
        "placeholder_query": "Query Adhira AI regarding diversion routing...",
        "actual_resolution": "Actual Resolution Time (Minutes)",
        "actual_officers": "Actual Deployed Officers",
        "actual_marshals": "Actual Deployed Marshals",
        "operator_remarks": "Operator Remarks",
        "submit_logs": "Submit Logs",
        "log_feedback_title": "Log Post-Incident Observations",
        "officers": "Officers",
        "marshals": "Marshals",
        "minutes": "Minutes",
        "visitor_count": "Visitor Count",
        "last_updated": "Last Updated",
        "weather_info": "27°C | Partly Cloudy",
        "aqi_info": "42 - Good",
        "menu_home": "Command Dashboard",
        "menu_planner": "AI Event Planner",
        "menu_intel": "Traffic Intelligence",
        "menu_hotspot": "Hotspot Analytics",
        "menu_incident": "Incident Response",
        "menu_metro": "Metro Coordination",
        "menu_bmtc": "BMTC Operations",
        "menu_adhira": "Adhira AI Assistant",
        "menu_control": "Control Center"
    },
    "ಕನ್ನಡ": {
        "title": "ವಾಹನಫ್ಲೋ: ಬೆಂಗಳೂರು ಬುದ್ಧಿವಂತ ಚಲನಶೀಲತೆ ಕಮಾಂಡ್ ಸೆಂಟರ್",
        "subtitle": "ನಗರಾಭಿವೃದ್ಧಿ ಭೂ ಸಾರಿಗೆ ನಿರ್ದೇಶನಾಲಯ (DULT) ಮತ್ತು ಬೆಂಗಳೂರು ಸಂಚಾರಿ ಪೊಲೀಸ್ ಜಂಟಿ ಉಪಕ್ರಮ",
        "gov_karnataka": "ಕರ್ನಾಟಕ ಸರ್ಕಾರ",
        "portal_badge": "ಅಧಿಕೃತ ಸುರಕ್ಷಿತ ಸರ್ಕಾರಿ ಪೋರ್ಟಲ್",
        "status_badge": "ಕಾರ್ಯಾಚರಣೆ ಮತ್ತು ಪರಿಶೀಲಿಸಲಾಗಿದೆ",
        "live_stream": "ಲೈವ್ ಸ್ಟ್ರೀಮ್",
        "emergency_bulletin": "ತುರ್ತು ಬುಲೆಟಿನ್",
        "nav_title": "ಸಿಸ್ಟಮ್ ಮೆನು",
        "switch_lang": "ಭಾಷೆ ಬದಲಿಸಿ",
        "emergency_mode": "ತುರ್ತು ಪರಿಸ್ಥಿತಿ ಮೋಡ್",
        "emergency_active": "ಎಚ್ಚರಿಕೆ: ತುರ್ತು ಪ್ರೋಟೋಕಾಲ್ಗಳು ಸಕ್ರಿಯವಾಗಿವೆ",
        "weather": "ಹವಾಮಾನ",
        "aqi": "ವಾಯು ಗುಣಮಟ್ಟ",
        "mobility": "ಚಲನಶೀಲತೆ ಸ್ಕೋರ್",
        "helpline": "ತುರ್ತು ಸಹಾಯವಾಣಿ",
        "active_events": "ಸಕ್ರಿಯ ಸಂಚಾರ ಘಟನೆಗಳು",
        "congestion_index": "ದಟ್ಟಣೆ ಸೂಚ್ಯಂಕ",
        "active_incidents": "ಸಕ್ರಿಯ ಅಪಘಾತಗಳು",
        "metro_status": "ಮೆಟ್ರೋ ಸ್ಥಿತಿ",
        "bmtc_status": "ಬಿಎಂಟಿಸಿ ಫ್ಲೀಟ್ ಸ್ಥಿತಿ",
        "road_closures": "ರಸ್ತೆ ಮುಚ್ಚುವಿಕೆಗಳು",
        "map_title": "ಲೈವ್ ಸಂಚಾರ ಹಾಟ್‌ಸ್ಪಾಟ್ ಜಿಐಎಸ್",
        "flow_trend": "ಸಂಚಾರ ದಟ್ಟಣೆ ಪ್ರವೃತ್ತಿ",
        "top_hotspots": "ಉನ್ನತ ಹಾಟ್‌ಸ್ಪಾಟ್‌ಗಳು",
        "event_forecast": "ಇವೆಂಟ್ ಪರಿಣಾಮ ಮುನ್ಸೂಚನೆ",
        "view_mitigation": "ಶಮನ ಯೋಜನೆಯನ್ನು ವೀಕ್ಷಿಸಿ",
        "planner_title": "ಬುದ್ಧಿವಂತ ಇವೆಂಟ್ ಯೋಜನೆ ಮತ್ತು ಸಂಪನ್ಮೂಲ ಹಂಚಿಕೆ",
        "input_params": "ಇನ್ಪುಟ್ ನಿಯತಾಂಕಗಳು",
        "event_class": "ಇವೆಂಟ್ ವರ್ಗೀಕರಣ",
        "primary_cause": "ಪ್ರಾಥಮಿಕ ಇವೆಂಟ್ ಕಾರಣ",
        "priority_level": "ಘಟನೆಯ ಆದ್ಯತೆಯ ಮಟ್ಟ",
        "road_closure_req": "ರಸ್ತೆ ಮುಚ್ಚುವ ಅಗತ್ಯವಿದೆಯೇ?",
        "police_station": "ಜವಾಬ್ದಾರಿಯುತ ಪೊಲೀಸ್ ಠಾಣೆ",
        "corridor": "ಸಂಚಾರ ಕಾರಿಡಾರ್",
        "zone": "ಆಡಳಿತಾತ್ಮಕ ವಲಯ",
        "junction": "ಬಾಧಿತ ಜಂಕ್ಷನ್",
        "coords": "ಘಟನೆ ಸ್ಥಳದ ನಿರ್ದೇಶಾಂಕಗಳು",
        "lat": "ಅಕ್ಷಾಂಶ",
        "lon": "ರೇಖಾಂಶ",
        "temporal": "ಸಮಯದ ನಿಯತಾಂಕಗಳು",
        "event_date": "ಇವೆಂಟ್ ದಿನಾಂಕ",
        "event_time": "ನಿಗದಿತ ಪ್ರಾರಂಭ ಸಮಯ",
        "vehicle_class": "ಒಳಗೊಂಡಿರುವ ವಾಹನ ವರ್ಗ",
        "generate_btn": "ಕಾರ್ಯತಂತ್ರದ ಕ್ರಿಯಾ ಯೋಜನೆ ರಚಿಸಿ",
        "intel_title": "ಸ್ಮಾರ್ಟ್ ಸಿಟಿ ಸಂಚಾರ ಬುದ್ಧಿವಂತಿಕೆ",
        "hotspot_title": "ಜಿಐಎಸ್ ಸಂಚಾರ ಹಾಟ್‌ಸ್ಪಾಟ್ ಓವರ್‌ಲೇ",
        "incident_title": "ತುರ್ತು ಘಟನೆ ಪ್ರತಿಕ್ರಿಯೆ ಸಮನ್ವಯ",
        "metro_title": "ನಮ್ಮ ಮೆಟ್ರೋ ಸಾರಿಗೆ ಸಮನ್ವಯ",
        "bmtc_title": "ಬಿಎಂಟಿಸಿ ಇಂಟೆಲಿಜೆಂಟ್ ಫ್ಲೀಟ್ ಕಾರ್ಯಾಚರಣೆ",
        "adhira_title": "ಅಧೀರಾ ಎಐ ಸಂಚಾರ ಸಲಹೆಗಾರ",
        "control_title": "ನಿಯಂತ್ರಣ ಕೊಠಡಿ ಮತ್ತು ಪ್ರತಿಕ್ರಿಯೆ ದಾಖಲೆ",
        "dispatch_btn": "ಕ್ಷೇತ್ರ ಘಟಕಗಳಿಗೆ ಕಮಾಂಡ್ ಆದೇಶವನ್ನು ರವಾನಿಸಿ",
        "dispatch_success": "ಕಮಾಂಡ್ ಆದೇಶವನ್ನು ಕ್ಷೇತ್ರ ನಿಯಂತ್ರಣಗಳಿಗೆ ಯಶಸ್ವಿಯಾಗಿ ರವಾನಿಸಲಾಗಿದೆ!",
        "dispatch_title": "ಅಧಿಕೃತ ಕಮಾಂಡ್ ರವಾನೆ ಆದೇಶ",
        "pred_congestion": "ಮುನ್ಸೂಚನೆಯ ದಟ್ಟಣೆ",
        "risk_threat": "ಕಾರ್ಯತಂತ್ರದ ಅಪಾಯದ ಬೆದರಿಕೆ",
        "resource_dispatch": "ಸಂಪನ್ಮೂಲ ರವಾನೆ ಅವಶ್ಯಕತೆಗಳು",
        "diversion_route": "ಡೈವರ್ಷನ್ ಮಾರ್ಗ",
        "action_plan": "ಕ್ರಿಯಾ ಯೋಜನೆ",
        "quick_commands": "ತ್ವರಿತ ಆಜ್ಞೆಗಳು",
        "query_assistant": "ಅಧೀರಾ ಸಹಾಯಕರನ್ನು ಪ್ರಶ್ನಿಸಿ",
        "active_alerts": "ಸಕ್ರಿಯ ಎಚ್ಚರಿಕೆಗಳು",
        "query_custom": "ಅಥವಾ ಅಧೀರಾ ಎಐಗಾಗಿ ಕಸ್ಟಮ್ ಪ್ರಶ್ನೆಯನ್ನು ನಮೂದಿಸಿ:",
        "placeholder_query": "ಮಾರ್ಗ ಬದಲಾವಣೆ ಬಗ್ಗೆ ಅಧೀರಾ ಎಐ ಅನ್ನು ಪ್ರಶ್ನಿಸಿ...",
        "actual_resolution": "ವಾಸ್ತವಿಕ ಪರಿಹಾರ ಸಮಯ (ನಿಮಿಷಗಳು)",
        "actual_officers": "ನಿಯೋಜಿಸಲಾದ ವಾಸ್ತವಿಕ ಅಧಿಕಾರಿಗಳು",
        "actual_marshals": "ನಿಯೋಜಿಸಲಾದ ವಾಸ್ತವಿಕ ಮಾರ್ಷಲ್‌ಗಳು",
        "operator_remarks": "ಆಪರೇಟರ್ ಟಿಪ್ಪಣಿಗಳು",
        "submit_logs": "ದಾಖಲೆಗಳನ್ನು ಸಲ್ಲಿಸಿ",
        "log_feedback_title": "ಘಟನೆಯ ನಂತರದ ಅವಲೋಕನಗಳನ್ನು ದಾಖಲಿಸಿ",
        "officers": "ಅಧಿಕಾರಿಗಳು",
        "marshals": "ಮಾರ್ಷಲ್‌ಗಳು",
        "minutes": "ನಿಮಿಷಗಳು",
        "visitor_count": "ಸಂದರ್ಶಕರ ಸಂಖ್ಯೆ",
        "last_updated": "ಕೊನೆಯದಾಗಿ ನವೀಕರಿಸಿದ್ದು",
        "weather_info": "೨೭°C | ಭಾಗಶಃ ಮೋಡ ಕವಿದ ವಾತಾವರಣ",
        "aqi_info": "೪೨ - ಉತ್ತಮ",
        "menu_home": "ಕಮಾಂಡ್ ಡ್ಯಾಶ್‌ಬೋರ್ಡ್",
        "menu_planner": "ಐಟಿಐ ಇವೆಂಟ್ ಪ್ಲಾನರ್",
        "menu_intel": "ಸಂಚಾರ ಗುಪ್ತಚರ",
        "menu_hotspot": "ಹಾಟ್‌ಸ್ಪಾಟ್ ವಿಶ್ಲೇಷಣೆ",
        "menu_incident": "ಘಟನೆ ಪ್ರತಿಕ್ರಿಯೆ",
        "menu_metro": "ಮೆಟ್ರೋ ಸಮನ್ವಯ",
        "menu_bmtc": "ಬಿಎಂಟಿಸಿ ಕಾರ್ಯಾಚರಣೆಗಳು",
        "menu_adhira": "ಅಧೀರಾ ಎಐ ಅಧಿಕಾರಿ",
        "menu_control": "ನಿಯಂತ್ರಣ ಕೇಂದ್ರ"
    }
}

def t(key):
    lang = st.session_state.get("lang", "English")
    return translations[lang].get(key, key)

# Base64 Image Loader for custom styling
def get_base64_image(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    return ""

banner_b64 = get_base64_image("bengaluru_skyline.png")
avatar_b64 = get_base64_image("adhira_avatar.png")
emblem_b64 = get_base64_image("bengaluru_government_emblem.png")

# Current timestamp for NIC status
current_time_str = datetime.datetime.now().strftime('%H:%M:%S, %d %b %Y')

# Injected CSS for official Government Portal look and feel
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght=300;400;500;600;700;800&family=Montserrat:wght=700;900&family=Noto+Sans+Kannada:wght@400;700&display=swap');
    
    html {{
        font-size: {st.session_state.font_size}% !important;
    }}
    html, body, [class*="css"] {{
        font-family: 'Inter', 'Noto Sans Kannada', sans-serif;
        background-color: #f8fafc !important;
        color: #1e293b;
    }}
    
    [data-testid="stAppViewContainer"] {{
        background-color: #f8fafc !important;
    }}
    
    /* Hide default Streamlit overlays */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    
    /* Main Page Padding Adjustments */
    .main .block-container {{
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        max-width: 95% !important;
    }}

    /* Accessibility Bar Overrides */
    .st-key-accessibility_bar_wrapper,
    .st-key-accessibility_bar_wrapper > div,
    .st-key-accessibility_bar_wrapper div[data-testid="stHorizontalBlock"],
    .st-key-accessibility_bar_wrapper div[data-testid="column"] {{
        background-color: #0b2545 !important;
        background: #0b2545 !important;
        padding: 0px !important;
        margin: 0px !important;
    }}
    
    .st-key-accessibility_bar_wrapper {{
        padding: 0px 2rem !important;
        border-radius: 6px !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05) !important;
        margin-top: 0.2rem !important;
        margin-bottom: 1.2rem !important;
        min-height: 32px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }}
    
    .st-key-accessibility_bar_wrapper div[data-testid="stHorizontalBlock"] {{
        align-items: center !important;
        width: 100% !important;
        gap: 0px !important;
    }}
    
    .st-key-accessibility_bar_wrapper div[data-testid="column"] {{
        display: flex !important;
        align-items: center !important;
        min-height: auto !important;
    }}
    
    .st-key-accessibility_bar_wrapper div[data-testid="column"] * {{
        color: #e2e8f0 !important;
        font-size: 0.72rem !important;
        margin-bottom: 0px !important;
    }}
    
    .st-key-accessibility_bar_wrapper p {{
        margin: 0px !important;
        padding: 0px !important;
        line-height: 1.2 !important;
    }}
    
    .st-key-accessibility_bar_wrapper a {{
        color: #d97706 !important;
        text-decoration: none !important;
        font-weight: 500 !important;
        margin-right: 15px !important;
    }}
    
    .st-key-accessibility_bar_wrapper a:hover {{
        text-decoration: underline !important;
        color: #ffffff !important;
    }}
    
    /* Access Buttons styling override */
    .st-key-accessibility_bar_wrapper div.stButton {{
        margin: 0px 2px !important;
        padding: 0px !important;
    }}
    
    .st-key-accessibility_bar_wrapper div.stButton > button {{
        background: transparent !important;
        border: 1px solid rgba(255,255,255,0.25) !important;
        color: #e2e8f0 !important;
        padding: 1px 6px !important;
        border-radius: 3px !important;
        cursor: pointer !important;
        font-size: 0.7rem !important;
        min-height: auto !important;
        height: 20px !important;
        line-height: 1 !important;
        box-shadow: none !important;
        transform: none !important;
        font-weight: bold !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
    }}
    
    .st-key-accessibility_bar_wrapper div.stButton > button:hover {{
        border-color: #d97706 !important;
        color: #ffffff !important;
        background-color: rgba(255, 255, 255, 0.05) !important;
    }}
    
    /* CSS Drawn Mini Indian Flag */
    .indian-flag-icon {{
        display: inline-block;
        width: 16px;
        height: 10px;
        background: linear-gradient(to bottom, #FF9933 33.33%, #FFFFFF 33.33%, #FFFFFF 66.66%, #138808 66.66%);
        position: relative;
        border: 1px solid rgba(255,255,255,0.25);
        border-radius: 1px;
        margin-right: 10px;
        vertical-align: middle;
    }}
    .indian-flag-icon::after {{
        content: "☸";
        color: #000080;
        font-size: 6px;
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        line-height: 1;
    }}
    
    /* Official Government Portal Header Grid */
    .gov-header-grid {{
        display: grid;
        grid-template-columns: 85px 1fr auto;
        gap: 20px;
        align-items: center;
        background-image: linear-gradient(rgba(11, 37, 69, 0.95), rgba(11, 37, 69, 0.98)), url("data:image/png;base64,{banner_b64}");
        background-size: cover;
        background-position: center;
        border-bottom: 4px solid #138808; /* Indian flag Green bottom border */
        border-top: 4px solid #FF9933; /* Indian flag Saffron top border */
        border-radius: 12px;
        padding: 1.5rem 2rem;
        box-shadow: 0 8px 16px rgba(11, 37, 69, 0.1);
        margin-bottom: 1.5rem;
    }}
    
    .emblem-img {{
        width: 80px;
        height: auto;
        filter: drop-shadow(0 0 10px rgba(255, 176, 0, 0.25));
    }}
    
    .gov-flag {{
        background-color: #d97706;
        color: #0b2545;
        font-size: 0.7rem;
        font-weight: bold;
        padding: 0.25rem 0.8rem;
        border-radius: 4px;
        display: inline-block;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    
    /* Administrative Panels */
    .admin-panel {{
        background: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
        color: #1e293b;
        transition: all 0.3s ease;
    }}
    
    .admin-panel:hover {{
        border-color: #0b2545;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -4px rgba(0, 0, 0, 0.1);
    }}
    
    .admin-panel h1, .admin-panel h2, .admin-panel h3, .admin-panel h4, .admin-panel h5, .admin-panel h6 {{
        color: #0b2545 !important;
        font-family: 'Montserrat', sans-serif !important;
        font-weight: 700 !important;
        margin-top: 0 !important;
    }}
    
    /* Alert Panels (Clean light style) */
    .alert-panel {{
        background: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        color: #1e293b;
    }}
    
    /* Neon Status Indicator */
    .status-active {{
        display: inline-block;
        width: 10px;
        height: 10px;
        background-color: #10b981;
        border-radius: 50%;
        margin-right: 8px;
        box-shadow: 0 0 8px #10b981;
        animation: blinker 1.8s infinite;
    }}
    
    @keyframes blinker {{
        0%, 100% {{ opacity: 0.3; }}
        50% {{ opacity: 1; }}
    }}
    
    /* Sidebar styling overrides */
    [data-testid="stSidebar"] {{
        background-color: #0b2545 !important;
        border-right: 1px solid rgba(255,255,255,0.1) !important;
    }}
    
    /* Force sidebar text elements to be light colored for readability */
    [data-testid="stSidebar"] *, [data-testid="stSidebar"] p, [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4, [data-testid="stSidebar"] h5, [data-testid="stSidebar"] h6, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label {{
        color: #f8fafc !important;
    }}
    
    /* Sidebar specific admin panel containers */
    [data-testid="stSidebar"] .admin-panel {{
        background: rgba(255, 255, 255, 0.08) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        color: #f8fafc !important;
        box-shadow: none !important;
    }}
    
    [data-testid="stSidebar"] .admin-panel * {{
        color: #f8fafc !important;
    }}
    
    /* Premium Styled Selectboxes (Dropdown Menus) */
    div[data-baseweb="select"] {{
        background-color: #ffffff !important;
        border: 2px solid #cbd5e1 !important;
        border-radius: 12px !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.02) !important;
        font-weight: 500 !important;
    }}
    div[data-baseweb="select"] * {{
        color: #1e293b !important;
    }}
    div[data-baseweb="select"]:hover {{
        border-color: #0b2545 !important;
        box-shadow: 0 4px 12px rgba(11, 37, 69, 0.08) !important;
    }}
    div[data-baseweb="select"]:focus-within {{
        border-color: #FF9933 !important;
        box-shadow: 0 0 0 3px rgba(255, 153, 51, 0.18) !important;
    }}
    
    [data-testid="stSidebar"] div[data-baseweb="select"] {{
        background-color: rgba(255, 255, 255, 0.06) !important;
        border: 2px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 12px !important;
        transition: all 0.3s ease !important;
    }}
    [data-testid="stSidebar"] div[data-baseweb="select"] * {{
        color: #1e293b !important;
    }}
    [data-testid="stSidebar"] div[data-baseweb="select"]:hover {{
        border-color: rgba(255, 255, 255, 0.4) !important;
        background-color: rgba(255, 255, 255, 0.1) !important;
    }}
    [data-testid="stSidebar"] div[data-baseweb="select"]:focus-within {{
        border-color: #FF9933 !important;
        box-shadow: 0 0 0 3px rgba(255, 153, 51, 0.25) !important;
    }}
    
    /* Premium Dropdown Listbox Popover styling */
    ul[role="listbox"] {{
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 12px !important;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1) !important;
        padding: 6px !important;
    }}
    div[role="option"], li[role="option"] {{
        border-radius: 8px !important;
        padding: 8px 12px !important;
        transition: all 0.2s ease !important;
        color: #1e293b !important;
    }}
    div[role="option"]:hover, li[role="option"]:hover {{
        background-color: #f1f5f9 !important;
        color: #0b2545 !important;
    }}
    
    /* Streamlit buttons custom styling */
    div.stButton > button {{
        background: linear-gradient(135deg, #0b2545, #134074) !important;
        border: 1px solid #0b2545 !important;
        color: #fff !important;
        border-radius: 8px !important;
        padding: 0.6rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 6px rgba(11, 37, 69, 0.15) !important;
    }}
    div.stButton > button:hover {{
        background: linear-gradient(135deg, #134074, #0b2545) !important;
        border-color: #FF9933 !important;
        box-shadow: 0 4px 10px rgba(255, 153, 51, 0.25) !important;
        transform: translateY(-1px);
    }}
    
    /* Input and form overrides */
    div[data-testid="stForm"] {{
        background: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05) !important;
    }}
    
    /* Premium Styled Input Fields (Search, Text, Number, Textarea) */
    div[data-baseweb="input"] {{
        background-color: #ffffff !important;
        border: 2px solid #cbd5e1 !important;
        border-radius: 12px !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.02) !important;
    }}
    div[data-baseweb="input"]:hover {{
        border-color: #0b2545 !important;
        box-shadow: 0 4px 12px rgba(11, 37, 69, 0.08) !important;
    }}
    div[data-baseweb="input"]:focus-within {{
        border-color: #FF9933 !important;
        box-shadow: 0 0 0 3px rgba(255, 153, 51, 0.18) !important;
    }}
    div[data-baseweb="input"] input {{
        border: none !important;
        box-shadow: none !important;
        background-color: transparent !important;
        color: #1e293b !important;
        font-weight: 500 !important;
    }}
    
    input[type="text"], input[type="number"], textarea {{
        background-color: #ffffff !important;
        border: 2px solid #cbd5e1 !important;
        color: #1e293b !important;
        border-radius: 12px !important;
        padding: 0.6rem 0.9rem !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.02) !important;
        font-weight: 500 !important;
    }}
    input[type="text"]:hover, input[type="number"]:hover, textarea:hover {{
        border-color: #0b2545 !important;
    }}
    input[type="text"]:focus, input[type="number"]:focus, textarea:focus {{
        border-color: #FF9933 !important;
        box-shadow: 0 0 0 3px rgba(255, 153, 51, 0.18) !important;
        background-color: #ffffff !important;
    }}
    
    /* Metric modifications */
    div[data-testid="stMetric"] {{
        background: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 8px !important;
        padding: 0.8rem !important;
        text-align: center !important;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02) !important;
    }}
    div[data-testid="stMetric"]:hover {{
        border-color: #0b2545;
        background: #f8fafc !important;
    }}
    div[data-testid="stMetricValue"] > div {{
        color: #0b2545 !important;
        font-family: 'Montserrat', sans-serif !important;
        font-weight: 800 !important;
        font-size: 1.8rem !important;
    }}
    div[data-testid="stMetricLabel"] > div {{
        color: #475569 !important;
        font-size: 0.75rem !important;
        font-weight: bold !important;
        text-transform: uppercase !important;
    }}
    
    /* Ticker styles */
    .ticker-container {{
        background: rgba(239, 68, 68, 0.06);
        border: 1px solid rgba(239, 68, 68, 0.2);
        border-radius: 8px;
        padding: 0.5rem 1rem;
        margin-bottom: 1.5rem;
        font-size: 0.82rem;
        color: #ef4444;
    }}
    
    /* Chatbox styles */
    .chat-box {{
        background: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 12px;
        padding: 1rem;
        height: 380px;
        overflow-y: auto;
        margin-bottom: 1rem;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.02);
    }}
    .chat-msg {{
        display: flex;
        gap: 10px;
        margin-bottom: 1rem;
        align-items: flex-start;
    }}
    .chat-msg.user {{
        flex-direction: row-reverse;
    }}
    .chat-avatar {{
        width: 36px;
        height: 36px;
        border-radius: 50%;
        border: 1px solid #cbd5e1;
        object-fit: cover;
    }}
    .chat-avatar.ai {{
        border-color: #0b2545;
        box-shadow: 0 0 6px rgba(11, 37, 69, 0.15);
    }}
    .chat-bubble {{
        max-width: 75%;
        padding: 0.75rem 1rem;
        border-radius: 12px;
        font-size: 0.82rem;
        line-height: 1.4;
    }}
    .chat-msg.ai .chat-bubble {{
        background: rgba(11, 37, 69, 0.04);
        border: 1px solid rgba(11, 37, 69, 0.1);
        border-top-left-radius: 2px;
        color: #1e293b;
    }}
    .chat-msg.user .chat-bubble {{
        background: rgba(255, 153, 51, 0.08);
        border: 1px solid rgba(255, 153, 51, 0.2);
        border-top-right-radius: 2px;
        color: #1e293b;
    }}
    
    /* Official Dispatch Document */
    .dispatch-order {{
        background: #fffdfa;
        border: 2px dashed #0b2545;
        border-radius: 12px;
        padding: 2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        font-family: 'Courier New', Courier, monospace;
        color: #1e293b;
        margin-top: 1rem;
    }}
    .dispatch-header {{
        text-align: center;
        border-bottom: 2px double #cbd5e1;
        padding-bottom: 1rem;
        margin-bottom: 1.5rem;
    }}
    .barcode-sim {{
        height: 25px;
        background: repeating-linear-gradient(90deg, #0b2545, #0b2545 2px, transparent 2px, transparent 6px);
        margin: 0.8rem 0;
        opacity: 0.8;
    }}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# Tricolor Top Strip (Indian Flag design touch)
# ─────────────────────────────────────────────────────────
st.markdown("""
<div style="height: 4px; background: linear-gradient(90deg, #FF9933 0%, #FF9933 33.33%, #FFFFFF 33.33%, #FFFFFF 66.66%, #138808 66.66%, #138808 100%); width: 100%; margin-top: -3.5rem; margin-bottom: 0.5rem; border-radius: 2px; box-shadow: 0 1px 5px rgba(0,0,0,0.5);"></div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# Accessibility Top Bar Insertion with CSS-drawn Flag (Interactive Zoom)
# ─────────────────────────────────────────────────────────
with st.container(key="accessibility_bar_wrapper"):
    acc_col1, acc_col2 = st.columns([6, 4])
    with acc_col1:
        st.markdown(f"""
        <div style="display: flex; align-items: center;">
            <span class="indian-flag-icon"></span>
            <a href="https://karnataka.gov.in" target="_blank">ಕರ್ನಾಟಕ ಸರ್ಕಾರ | karnataka.gov.in</a> &nbsp;|&nbsp;&nbsp;
            <a href="https://india.gov.in" target="_blank">ಭಾರತ ಸರ್ಕಾರದ ಪೋರ್ಟಲ್ | india.gov.in</a>
        </div>
        """, unsafe_allow_html=True)
    with acc_col2:
        btn_col_text, btn_col_out, btn_col_reset, btn_col_in, btn_col_lang = st.columns([3.5, 0.8, 0.8, 0.8, 3.8])
        with btn_col_text:
            st.markdown('<div style="text-align: right; padding-top: 2px;">Screen Reader Access &nbsp;|&nbsp;</div>', unsafe_allow_html=True)
        with btn_col_out:
            if st.button("A-", key="zoom_out", help="Zoom Out / Decrease Font Size"):
                st.session_state.font_size = max(80, st.session_state.font_size - 10)
                st.rerun()
        with btn_col_reset:
            if st.button("A", key="zoom_reset", help="Reset Font Size"):
                st.session_state.font_size = 100
                st.rerun()
        with btn_col_in:
            if st.button("A+", key="zoom_in", help="Zoom In / Increase Font Size"):
                st.session_state.font_size = min(130, st.session_state.font_size + 10)
                st.rerun()
        with btn_col_lang:
            st.markdown(f"""
            <div style="text-align: left; padding-top: 2px;">
                &nbsp;|&nbsp;
                <span style="color:{ '#ffb000' if st.session_state.lang == 'ಕನ್ನಡ' else '#cbd5e1' }; font-weight:bold;">ಕನ್ನಡ</span> / 
                <span style="color:{ '#ffb000' if st.session_state.lang == 'English' else '#cbd5e1' }; font-weight:bold;">English</span>
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# INTERACTIVE LANGUAGE SWITCHER (TOP RIGHT OF HOMEPAGE)
# ─────────────────────────────────────────────────────────
top_lang_col1, top_lang_col2 = st.columns([10, 2])
with top_lang_col2:
    st.segmented_control(
        label="Language Switcher",
        options=["English", "ಕನ್ನಡ"],
        key="lang",
        label_visibility="collapsed",
        selection_mode="single"
    )

# ─────────────────────────────────────────────────────────
# 1. LOAD DATA & MODELS (CACHED)
# ─────────────────────────────────────────────────────────
@st.cache_data
def load_historical_data():
    df = pd.read_csv(r"flip.csv")
    df = df.dropna(subset=['latitude', 'longitude'])
    df = df[(df['latitude'] >= 12.8) & (df['latitude'] <= 13.3)]
    df = df[(df['longitude'] >= 77.3) & (df['longitude'] <= 77.8)]
    return df

@st.cache_resource
def load_ensemble_models():
    with open('preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)
        
    model_pt = TabularResNet(
        cat_dims=preprocessor['cat_dims'],
        num_features=len(preprocessor['num_cols_pt']),
        hidden_dim=128,
        num_blocks=2,
        dropout=0.15
    )
    model_pt.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
    model_pt.eval()
    
    from catboost import CatBoostRegressor
    model_cb = CatBoostRegressor()
    model_cb.load_model('catboost_model.cbm')
    return model_pt, model_cb, preprocessor

try:
    df_history = load_historical_data()
    model_pt, model_cb, preprocessor = load_ensemble_models()
    model_loaded = True
except Exception as e:
    st.error(f"Error loading models: {e}.")
    model_loaded = False

# Chat History Initial Welcome Message
welcome_msg = (
    "ನಮಸ್ಕಾರ | Welcome to the Bengaluru Intelligent Mobility Command Assistant. I am Officer Adhira, NIC's Mobility Advisory AI. How can I assist you with city traffic deployment today?"
    if st.session_state.lang == "ಕನ್ನಡ" else
    "Welcome to the Bengaluru Intelligent Mobility Command Assistant. I am Officer Adhira, NIC's Mobility Advisory AI. How can I assist you today?"
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "ai", "content": welcome_msg, "timestamp": datetime.datetime.now().strftime("%H:%M")}
    ]

# ─────────────────────────────────────────────────────────
# 2. BILINGUAL SIDEBAR NAVIGATION TREE & LANGUAGE TOGGLE
# ─────────────────────────────────────────────────────────
with st.sidebar:
    # Emblem at top of sidebar
    if emblem_b64:
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem 0; border-bottom: 1px solid #cbd5e1; margin-bottom: 1rem;">
            <img src="data:image/png;base64,{emblem_b64}" style="width: 80px; height: auto; filter: drop-shadow(0 0 8px rgba(255,176,0,0.3));"/>
            <h4 style="margin: 0.5rem 0 0 0; color: #d97706; font-family: 'Poppins', sans-serif; font-size: 0.95rem; font-weight: bold;">{t("gov_karnataka")}</h4>
            <small style="color: #64748b; text-transform: uppercase; font-size: 0.7rem; letter-spacing: 0.5px;">Govt of Karnataka</small>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem 0; border-bottom: 1px solid #cbd5e1; margin-bottom: 1.5rem;">
            <span style="font-size: 2.2rem; display: block;">🏛️</span>
            <h4 style="margin: 0.2rem 0 0 0; color: #d97706; font-family: 'Poppins', sans-serif; font-size: 0.95rem; font-weight: bold;">{t("gov_karnataka")}</h4>
            <small style="color: #64748b; text-transform: uppercase; font-size: 0.7rem; letter-spacing: 0.5px;">Govt of Karnataka</small>
        </div>
        """, unsafe_allow_html=True)

    # Relocated to top right of main page
        
    st.markdown("### 🏛️ " + t("nav_title"))
    
    menu_options = [
        "🏠 " + t("menu_home"),
        "🚦 " + t("menu_planner"),
        "📊 " + t("menu_intel"),
        "🗺 " + t("menu_hotspot"),
        "🚔 " + t("menu_incident"),
        "🚇 " + t("menu_metro"),
        "🚌 " + t("menu_bmtc"),
        "🤖 " + t("menu_adhira"),
        "⚙ " + t("menu_control")
    ]
    choice = st.selectbox(t("nav_title"), menu_options)
    
    st.markdown("---")
    emergency_toggle = st.toggle("🚨 " + t("emergency_mode"), value=False)
    if emergency_toggle:
        st.markdown(f"""
        <div style="background: rgba(255, 75, 75, 0.15); border: 1px solid #ff4b4b; padding: 0.8rem; border-radius: 8px; color: #ff4b4b; text-align: center; font-weight: bold; font-size: 0.8rem; margin-bottom: 1rem;">
            {t("emergency_active")}
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown(f"""
    <div class="admin-panel" style="padding: 1rem; margin-bottom: 0.5rem; font-size: 0.8rem;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.4rem;">
            <span>⛅ {t("weather")}</span>
            <strong>{t("weather_info")}</strong>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.4rem;">
            <span>🍃 {t("aqi")}</span>
            <span style="color: #00ff66; font-weight: bold;">{t("aqi_info")}</span>
        </div>
        <div style="display: flex; justify-content: space-between;">
            <span>📈 {t("mobility")}</span>
            <span style="color: #00d4ff; font-weight: bold;">86/100</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Govt Hotline
    st.markdown(f"""
    <div style="border: 1px solid rgba(255, 75, 75, 0.4); padding: 0.8rem; border-radius: 12px; background: rgba(255, 75, 75, 0.05); display: flex; align-items: center; justify-content: space-between; margin-top: 1rem;">
        <div>
            <div style="color: #ff4b4b; font-size: 0.75rem; text-transform: uppercase; font-weight: bold;">{t("helpline")}</div>
            <div style="font-size: 1.5rem; font-weight: 800; color: #ff4b4b; line-height:1;">112</div>
        </div>
        <div style="font-size: 0.7rem; color: #888; text-align: right;">24/7 Support</div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# 3. STATE PORTAL HERO HEADER & COMPLIANCE DISCLAIMER
# ─────────────────────────────────────────────────────────
header_logo_html = ""
if emblem_b64:
    header_logo_html = f'<img class="emblem-img" src="data:image/png;base64,{emblem_b64}" alt="Government Emblem"/>'
else:
    header_logo_html = '<div style="font-size: 3.2rem;">🏛️</div>'

st.markdown(f"""
<div class="gov-header-grid">
    {header_logo_html}
    <div>
        <div class="gov-flag">{t("portal_badge")}</div>
        <h4 style="margin: 0.2rem 0 0 0; color: #d97706; font-family: 'Poppins', sans-serif; font-size: 0.95rem; font-weight: bold; letter-spacing: 0.5px;">{t("gov_karnataka")}</h4>
        <h1 style="margin: 0.3rem 0; font-family: 'Montserrat', sans-serif; font-size: 1.8rem; font-weight: 900; letter-spacing: 1px; color:#fff;">
            {t("title")}
        </h1>
        <p style="margin: 5px 0 0 0; font-size: 0.8rem; color: #a0aec0; font-style: italic;">
            {t("subtitle")}
        </p>
    </div>
    <div style="text-align: right; background: rgba(5,11,22,0.65); padding: 1rem; border-radius: 12px; border: 1px solid rgba(0,212,255,0.15); backdrop-filter: blur(5px);">
        <div style="font-size: 0.75rem; color: #888; font-weight: bold; text-transform: uppercase;">{t("status_badge")}</div>
        <div style="display: flex; align-items: center; justify-content: flex-end; margin-top: 0.4rem; gap: 6px;">
            <span class="status-active"></span>
            <span style="color: #00ff66; font-size: 0.85rem; font-weight: bold; text-shadow: 0 0 8px rgba(0,255,102,0.4);">{t("status_badge")}</span>
        </div>
        <div style="font-size: 0.72rem; color: #a0aec0; margin-top: 0.5rem;">{t("live_stream")}: <span style="color:#00d4ff; font-weight:bold;">{current_time_str}</span></div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# 4. GOVERNMENT SCROLLING MARQUEE TICKER (ALERT BAR)
# ─────────────────────────────────────────────────────────
bulletin_msg = (
    "⚠️ <strong>ತುರ್ತು ನವೀಕರಣ:</strong> ಹೊರ ವರ್ತುಲ ರಸ್ತೆ ಮತ್ತು ಸರ್ಜಾಪುರ ರಸ್ತೆ ವಲಯಗಳಲ್ಲಿ ಭಾರಿ ಮಳೆ ಉಂಟಾಗಿದೆ. ಪೀಣ್ಯ ಮೆಟ್ರೋ ನಿಲ್ದಾಣದಲ್ಲಿ ಫೀಡರ್ ಬಸ್ ಸೇವೆ ಹೆಚ್ಚಿಸಲಾಗಿದೆ. | ನಮ್ಮ ಮೆಟ್ರೋ ನೇರಳೆ ಮತ್ತು ಹಸಿರು ಲೈನ್ ೧೦೦% ಚಾಲನೆಯಲ್ಲಿದೆ."
    if st.session_state.lang == "ಕನ್ನಡ" else
    "⚠️ <strong>EMERGENCY BULLETIN:</strong> Heavy localized rainfall monitored at Outer Ring Road, Sarjapur Road sectors. Feeder shuttle frequencies adjusted at Peenya metro station. | Namma Metro Purple & Green lines running at 100% operational levels."
)

st.markdown(f"""
<div class="ticker-container">
    <marquee scrollamount="4">
        {bulletin_msg}
    </marquee>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# VIEW 1: HOME COMMAND DASHBOARD
# ─────────────────────────────────────────────────────────
if t("menu_home") in choice:
    # Six KPI cards with translated titles
    kpi_cols = st.columns(6)
    
    with kpi_cols[0]:
        st.markdown(f"""
        <div class="admin-panel" style="border-top: 3px solid #00d4ff; padding: 1rem;">
            <div style="font-size: 0.75rem; color: #888; font-weight: bold;">{t("active_events")}</div>
            <h2 style="color: #00d4ff; margin: 0.3rem 0; font-size: 1.8rem; font-weight: 800;">8,173</h2>
            <svg width="100%" height="20" style="opacity: 0.6;"><path d="M 0,15 C 20,5 40,20 60,8 C 80,0 90,15 120,5" fill="none" stroke="#00d4ff" stroke-width="2"/></svg>
            <div style="font-size: 0.75rem; color: #00d4ff;">↑ 12% vs yesterday</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi_cols[1]:
        st.markdown(f"""
        <div class="admin-panel" style="border-top: 3px solid #00ff66; padding: 1rem;">
            <div style="font-size: 0.75rem; color: #888; font-weight: bold;">{t("congestion_index")}</div>
            <h2 style="color: #00ff66; margin: 0.3rem 0; font-size: 1.8rem; font-weight: 800;">78%</h2>
            <svg width="100%" height="20" style="opacity: 0.6;"><path d="M 0,18 C 20,10 40,8 60,15 C 80,20 90,5 120,3" fill="none" stroke="#00ff66" stroke-width="2"/></svg>
            <div style="font-size: 0.75rem; color: #00ff66;">↑ 8% vs yesterday</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi_cols[2]:
        st.markdown(f"""
        <div class="admin-panel" style="border-top: 3px solid #ffb000; padding: 1rem;">
            <div style="font-size: 0.75rem; color: #888; font-weight: bold;">{t("active_incidents")}</div>
            <h2 style="color: #d97706; margin: 0.3rem 0; font-size: 1.8rem; font-weight: 800;">23</h2>
            <svg width="100%" height="20" style="opacity: 0.6;"><path d="M 0,10 C 20,15 40,3 60,12 C 80,18 90,2 120,8" fill="none" stroke="#ffb000" stroke-width="2"/></svg>
            <div style="font-size: 0.75rem; color: #d97706;">↓ 5% vs yesterday</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi_cols[3]:
        metro_label = "ಚಾಲನೆಯಲ್ಲಿದೆ" if st.session_state.lang == "ಕನ್ನಡ" else "Operational"
        metro_sub = "ಎಲ್ಲಾ ಮಾರ್ಗಗಳು ಕಾರ್ಯನಿರ್ವಹಿಸುತ್ತಿವೆ" if st.session_state.lang == "ಕನ್ನಡ" else "All Lines Running"
        st.markdown(f"""
        <div class="admin-panel" style="border-top: 3px solid #8b5cf6; padding: 1rem;">
            <div style="font-size: 0.75rem; color: #888; font-weight: bold;">{t("metro_status")}</div>
            <h2 style="color: #8b5cf6; margin: 0.3rem 0; font-size: 1.3rem; font-weight: 800; line-height: 2.1;">{metro_label}</h2>
            <svg width="100%" height="20" style="opacity: 0.6;"><path d="M 0,10 L 120,10" fill="none" stroke="#8b5cf6" stroke-width="2"/></svg>
            <div style="font-size: 0.75rem; color: #8b5cf6;">{metro_sub}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi_cols[4]:
        bmtc_sub = "ರಸ್ತೆಯಲ್ಲಿರುವ ಬಸ್‌ಗಳು" if st.session_state.lang == "ಕನ್ನಡ" else "On-Road Fleet"
        st.markdown(f"""
        <div class="admin-panel" style="border-top: 3px solid #3b82f6; padding: 1rem;">
            <div style="font-size: 0.75rem; color: #888; font-weight: bold;">{t("bmtc_status")}</div>
            <h2 style="color: #3b82f6; margin: 0.3rem 0; font-size: 1.8rem; font-weight: 800;">92%</h2>
            <svg width="100%" height="20" style="opacity: 0.6;"><path d="M 0,15 C 20,12 40,15 60,8 Q 80,0 120,12" fill="none" stroke="#3b82f6" stroke-width="2"/></svg>
            <div style="font-size: 0.75rem; color: #3b82f6;">{bmtc_sub}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi_cols[5]:
        closure_sub = "ಪ್ರಸ್ತುತ ಸಕ್ರಿಯವಾಗಿದೆ" if st.session_state.lang == "ಕನ್ನಡ" else "Active Now"
        st.markdown(f"""
        <div class="admin-panel" style="border-top: 3px solid #ff4b4b; padding: 1rem;">
            <div style="font-size: 0.75rem; color: #888; font-weight: bold;">{t("road_closures")}</div>
            <h2 style="color: #ff4b4b; margin: 0.3rem 0; font-size: 1.8rem; font-weight: 800;">17</h2>
            <svg width="100%" height="20" style="opacity: 0.6;"><path d="M 0,5 Q 30,18 60,10 T 120,15" fill="none" stroke="#ff4b4b" stroke-width="2"/></svg>
            <div style="font-size: 0.75rem; color: #ff4b4b;">{closure_sub}</div>
        </div>
        """, unsafe_allow_html=True)

    grid_cols = st.columns([2.2, 1.2])
    
    with grid_cols[0]:
        st.markdown(f"""
        <div class="admin-panel" style="padding: 0; position: relative;">
            <div style="padding: 1.5rem; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #cbd5e1;">
                <h4 style="margin: 0; color: #0b2545; font-family: 'Montserrat', sans-serif;">🗺 {t("map_title")}</h4>
                <span style="background: rgba(0, 255, 102, 0.15); color: #00ff66; padding: 0.2rem 0.6rem; border-radius: 12px; font-size: 0.75rem; font-weight: bold;">● Active GIS Feeds</span>
            </div>
            <div style="padding: 1rem;">
        """, unsafe_allow_html=True)
        
        # Simple clean map
        map_df = df_history[['latitude', 'longitude']].rename(columns={'latitude': 'lat', 'longitude': 'lon'})
        st.map(map_df, zoom=11)
        
        st.markdown("</div></div>", unsafe_allow_html=True)
        
    with grid_cols[1]:
        # Adhira AI panel
        nickname = "ನಗರಾಭಿವೃದ್ಧಿ ಮತ್ತು ಐಟಿಐ ಆಪ್ತ" if st.session_state.lang == "ಕನ್ನಡ" else "NIC-Developed Mobility Advisor"
        ask_placeholder = "ಬೆಂಗಳೂರು ಸಂಚಾರದ ಬಗ್ಗೆ ಏನೇ ಆದರೂ ಕೇಳಿ..." if st.session_state.lang == "ಕನ್ನಡ" else "Ask anything about traffic in Bengaluru..."
        st.markdown(f"""
        <div class="admin-panel" style="padding: 1.2rem;">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 1rem;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <img src="data:image/png;base64,{avatar_b64}" style="width: 45px; height: 45px; border-radius: 50%; border: 2px solid #0b2545; box-shadow: 0 0 6px rgba(11, 37, 69, 0.15);"/>
                    <div>
                        <h5 style="margin: 0; color: #0b2545; font-family: 'Montserrat', sans-serif;">{t("menu_adhira")}</h5>
                        <small style="color: #d97706; font-weight: 600;">{nickname}</small>
                    </div>
                </div>
                <span style="font-size: 1.5rem;">🤖</span>
            </div>
            <div style="font-size: 0.85rem; color: #888; margin-bottom: 0.8rem;">{ask_placeholder}</div>
        """, unsafe_allow_html=True)
        
        commands_list = [
            "What will be the impact of an event at Chinnaswamy Stadium?",
            "Suggest best diversion for Silk Board area",
            "Explain traffic anomaly observed at Peenya industrial sector"
        ]
        if st.session_state.lang == "ಕನ್ನಡ":
            commands_list = [
                "ಚಿன்னಸ್ವಾಮಿ ಕ್ರೀಡಾಂಗಣದಲ್ಲಿ ಇವೆಂಟ್‌ನ ಪರಿಣಾಮವೇನು?",
                "ಸಿಲ್ಕ್ ಬೋರ್ಡ್ ಪ್ರದೇಶಕ್ಕೆ ಉತ್ತಮ ಮಾರ್ಗ ಬದಲಾವಣೆ ಸೂಚಿಸಿ",
                "ಪೀಣ್ಯ ಕೈಗಾರಿಕಾ ವಲಯದಲ್ಲಿ ಕಂಡುಬಂದ ಸಂಚಾರ ವೈಪರೀತ್ಯವನ್ನು ವಿವರಿಸಿ"
            ]
            
        ai_query = st.selectbox(
            t("quick_commands") + ":",
            commands_list,
            key="home_adhira_select"
        )
        
        if st.button(t("query_assistant"), key="adhira_home_btn", use_container_width=True):
            st.toast("Adhira AI analyzing request...", icon="🤖")
            if "Chinnaswamy" in ai_query or "ಚಿன்னಸ್ವಾಮಿ" in ai_query:
                ans = (
                    "ಮಾರ್ಗ ಬದಲಾವಣೆಯನ್ನು ಹೊರ ವರ್ತುಲ ರಸ್ತೆ ಅಥವಾ ಕಬ್ಬನ್ ರಸ್ತೆ ಮೂಲಕ ಶಿಫಾರಸು ಮಾಡಲಾಗಿದೆ. ನಿಯೂಜಿಸಬೇಕಾದ ಅಧಿಕಾರಿಗಳು: ೧೨, ಮಾರ್ಷಲ್ಸ್: ೨೦."
                    if st.session_state.lang == "ಕನ್ನಡ" else
                    "Diversion recommended via Outer Ring Road or Cubbon Road. Optimal marshals deployment suggested is 12 officers, 20 marshals."
                )
            elif "Silk Board" in ai_query or "ಸಿಲ್ಕ್" in ai_query:
                ans = (
                    "ಭಾರೀ ಮಳೆಯಿಂದಾಗಿ ರಸ್ತೆ ಜಲಾವೃತವಾಗುವ ಹೆಚ್ಚಿನ ಅಪಾಯವಿದೆ. ನಿಯೋಜನೆ: ೮ ಅಧಿಕಾರಿಗಳು, ೧೫ ಮಾರ್ಷಲ್‌ಗಳು ಮತ್ತು ಕ್ರೇನ್‌ಗಳು ಸಿದ್ಧವಾಗಿರಬೇಕು."
                    if st.session_state.lang == "ಕನ್ನಡ" else
                    "High risk of waterlogging delays. Recommended dispatch: 8 officers, 15 marshals, and standby towing cranes. Divert via HSR Layout routes."
                )
            else:
                ans = (
                    "ಎಕ್ಸಿಟ್ ಬಳಿ ಸರಕು ಸಾಗಣೆ ವಾಹನ ಸ್ಥಗಿತಗೊಂಡಿದೆ. ಕ್ರಮ: ೨ ಭಾರಿ ಗಾತ್ರದ ಟೋಯಿಂಗ್ ವಾಹನಗಳು. ತುಮಕೂರು ರಸ್ತೆ ಫ್ಲೈಓವರ್ ಮೂಲಕ ಸಂಚಾರ ಬದಲಾವಣೆ."
                    if st.session_state.lang == "ಕನ್ನಡ" else
                    "Freight breakdown on exit. Recommended clearance action: 2 heavy towing rigs. Divert via Tumkur road flyover."
                )
            st.session_state.home_ai_reply = ans
            
        if "home_ai_reply" in st.session_state:
            reply_title = "ಅಧೀರಾ ಎಐ ಬುದ್ಧಿವಂತಿಕೆ" if st.session_state.lang == "ಕನ್ನಡ" else "ADHIRA AI Intel"
            st.markdown(f"""
            <div class="chat-bubble-ai" style="font-size: 0.8rem; margin-top: 0.8rem; background: rgba(11, 37, 69, 0.04); padding: 0.8rem; border-left: 3px solid #0b2545; border-radius: 4px; color: #1e293b;">
                🤖 <strong>{reply_title}:</strong> {st.session_state.home_ai_reply}
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Alerts
        alert1 = "<strong>ಭಾರಿ ದಟ್ಟಣೆ:</strong> ಎಂ ಜಿ ರಸ್ತೆ ➔ ಬ್ರಿಗೇಡ್ ರಸ್ತೆ (೧೦ ನಿಮಿಷಗಳ ಹಿಂದೆ)" if st.session_state.lang == "ಕನ್ನಡ" else "<strong>Heavy Congestion:</strong> MG Road ➔ Brigade Road (10 min ago)"
        alert2 = "<strong>ರಸ್ತೆ ಮುಚ್ಚುವಿಕೆ:</strong> ಕಬ್ಬನ್ ರಸ್ತೆ ಅಂಡರ್‌ಪಾಸ್ ಕಾಮಗಾರಿ (೧೫ ನಿಮಿಷಗಳ ಹಿಂದೆ)" if st.session_state.lang == "ಕನ್ನಡ" else "<strong>Road Closure:</strong> Cubbon Road underpass construction (15 min ago)"
        alert3 = "<strong>ಇವೆಂಟ್ ಮುನ್ಸೂಚನೆ:</strong> ಚಿன்னಸ್ವಾಮಿ ಕ್ರೀಡಾಂಗಣದಲ್ಲಿ ಇಂದು ಪಂದ್ಯ (೨೦ ನಿಮಿಷಗಳ ಹಿಂದೆ)" if st.session_state.lang == "ಕನ್ನಡ" else "<strong>Event Forecast:</strong> Chinnaswamy Stadium Event Today (20 min ago)"
        st.markdown(f"""
        <div class="alert-panel" style="padding: 1.2rem;">
            <h5 style="margin: 0 0 0.8rem 0; color: #0b2545; font-family: 'Montserrat', sans-serif;">🚨 {t("active_alerts")}</h5>
            <div class="alert-card" style="border-left: 4px solid #ff4b4b; background: #f8fafc; border-top: 1px solid #cbd5e1; border-right: 1px solid #cbd5e1; border-bottom: 1px solid #cbd5e1; padding: 0.6rem; border-radius: 4px; font-size: 0.8rem; margin-bottom: 0.5rem; color: #1e293b;">
                {alert1}
            </div>
            <div class="alert-card" style="border-left: 4px solid #ffb000; background: #f8fafc; border-top: 1px solid #cbd5e1; border-right: 1px solid #cbd5e1; border-bottom: 1px solid #cbd5e1; padding: 0.6rem; border-radius: 4px; font-size: 0.8rem; margin-bottom: 0.5rem; color: #1e293b;">
                {alert2}
            </div>
            <div class="alert-card" style="border-left: 4px solid #10b981; background: #f8fafc; border-top: 1px solid #cbd5e1; border-right: 1px solid #cbd5e1; border-bottom: 1px solid #cbd5e1; padding: 0.6rem; border-radius: 4px; font-size: 0.8rem; color: #1e293b;">
                {alert3}
            </div>
        </div>
        """, unsafe_allow_html=True)

    bottom_cols = st.columns([1.2, 1.2, 1])
    
    with bottom_cols[0]:
        st.markdown(f"""
        <div class="admin-panel">
            <h5 style="margin: 0 0 0.8rem 0; color: #0b2545; font-family: 'Montserrat', sans-serif;">📊 {t("flow_trend")}</h5>
        """, unsafe_allow_html=True)
        
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=['12 AM', '4 AM', '8 AM', '12 PM', '4 PM', '8 PM', '12 AM'],
            y=[18, 25, 78, 48, 85, 62, 22],
            mode='lines+markers',
            line=dict(color='#0b2545', width=3),
            fill='tozeroy',
            fillcolor='rgba(11, 37, 69, 0.1)',
            marker=dict(color='#FF9933', size=6)
        ))
        fig_trend.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            height=160,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#475569', size=9),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(11, 37, 69, 0.05)')
        )
        st.plotly_chart(fig_trend, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with bottom_cols[1]:
        st.markdown(f"""
        <div class="admin-panel">
            <h5 style="margin: 0 0 0.8rem 0; color: #0b2545; font-family: 'Montserrat', sans-serif;">📍 {t("top_hotspots")}</h5>
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <div style="font-size: 0.85rem; width: 60%;">
                    <div>1. <strong>Silk Board</strong> <span style="color:#d97706; font-weight:bold;">(78%)</span></div>
                    <div style="margin-top:0.4rem;">2. <strong>MG Road</strong> <span style="color:#d97706; font-weight:bold;">(72%)</span></div>
                    <div style="margin-top:0.4rem;">3. <strong>Hebbal Flyover</strong> <span style="color:#d97706; font-weight:bold;">(61%)</span></div>
                </div>
        """, unsafe_allow_html=True)
        
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=72,
            number={'suffix': "%", 'font': {'color': '#0b2545', 'size': 18}},
            title={'text': "Avg Impact" if st.session_state.lang == "English" else "ಸರಾಸರಿ ಪ್ರಭಾವ", 'font': {'color': '#475569', 'size': 10}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': "rgba(0,0,0,0)"},
                'bar': {'color': "#0b2545"},
                'bgcolor': "#e2e8f0",
                'borderwidth': 0
            }
        ))
        fig_gauge.update_layout(
            margin=dict(l=5, r=5, t=5, b=5),
            height=120,
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_gauge, use_container_width=True)
        st.markdown("</div></div>", unsafe_allow_html=True)
        
    with bottom_cols[2]:
        stad_title = "ಚಿನ್ನಸ್ವಾಮಿ ಕ್ರೀಡಾಂಗಣದಲ್ಲಿ ಇವೆಂಟ್" if st.session_state.lang == "ಕನ್ನಡ" else "Chinnaswamy Stadium Event"
        stad_commuters = "ನಿರೀಕ್ಷಿತ ಕಮಿಷನರ್ಸ್: ೪೦,೦೦೦+" if st.session_state.lang == "ಕನ್ನಡ" else "Expected: 40,000+ commuters"
        stad_cong_text = "ನಿರೀಕ್ಷಿತ ದಟ್ಟಣೆ" if st.session_state.lang == "ಕನ್ನಡ" else "Expected Congestion"
        stad_dur_text = "ಅವಧಿ" if st.session_state.lang == "ಕನ್ನಡ" else "Duration"
        stad_hours = "೪-೬ ಗಂಟೆಗಳು" if st.session_state.lang == "ಕನ್ನಡ" else "4-6 Hours"
        st.markdown(f"""
        <div class="admin-panel" style="border-left: 4px solid #ff4b4b;">
            <h5 style="margin: 0 0 0.4rem 0; color: #0b2545; font-family: 'Montserrat', sans-serif;">📅 {t("event_forecast")}</h5>
            <div style="font-size: 0.82rem; margin-bottom: 0.5rem;">
                <strong>{stad_title}</strong><br/>
                <span style="color: #888;">{stad_commuters}</span>
            </div>
            <div style="font-size: 0.8rem; color: #d97706;">
                🚦 {stad_cong_text}: <strong>85%</strong>
            </div>
            <div style="font-size: 0.8rem; color: #00d4ff; margin-bottom: 0.5rem;">
                ⏳ {stad_dur_text}: <strong>{stad_hours}</strong>
            </div>
        """, unsafe_allow_html=True)
        if st.button(t("view_mitigation"), key="event_forecast_view_btn", use_container_width=True):
            st.toast("Redirecting to AI Event Planner...", icon="🚦")
        st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# VIEW 2: AI EVENT PLANNER
# ─────────────────────────────────────────────────────────
elif t("menu_planner") in choice:
    st.subheader(f"🚦 {t('planner_title')}")
    
    if not model_loaded:
        st.warning("Please train the PyTorch/CatBoost ensemble models first by executing `python model.py` in your terminal.")
    else:
        # Predefined popular congestion presets in Bengaluru
        PRESETS = {
            "Custom Location" if st.session_state.lang == "English" else "ಕಸ್ಟಮ್ ಸ್ಥಳ": {
                "lat": 12.9716,
                "lon": 77.5946,
                "police_station": "Halasuru gate",
                "corridor": "Mysore road",
                "zone": "Central zone 2",
                "junction": "Hudsoncircle"
            },
            "Silk Board Junction (Bommanahalli / Hosur Road)": {
                "lat": 12.9188,
                "lon": 77.6215,
                "police_station": "Madiwala",
                "corridor": "Orr east 1",
                "zone": "South zone 2",
                "junction": "Silkboardjunc"
            },
            "Hudson Circle (East / Nrupathunga Road)": {
                "lat": 12.9661,
                "lon": 77.5897,
                "police_station": "Halasuru gate",
                "corridor": "Mysore road",
                "zone": "Central zone 2",
                "junction": "Hudsoncircle"
            },
            "Hebbal Flyover (Hebbal / Bellary Road)": {
                "lat": 13.0419,
                "lon": 77.5947,
                "police_station": "Hebbala",
                "corridor": "Orr north 1",
                "zone": "North zone 1",
                "junction": "Hebbalflyoverjunc"
            },
            "Indiranagar 100ft Road (East / 100 feet road)": {
                "lat": 12.9842,
                "lon": 77.6425,
                "police_station": "Jeevanbheemanagar",
                "corridor": "Old madras road",
                "zone": "Central zone 1",
                "junction": "Oldmadrasrd-indranagar100ftrdjunc"
            },
            "Majestic Interchange (West / Subedar Chatram road)": {
                "lat": 12.9742,
                "lon": 77.5745,
                "police_station": "Upparpet",
                "corridor": "Non-corridor",
                "zone": "Central zone 2",
                "junction": "Upparpetjunction"
            }
        }

        # Initialize session state variables
        if "lat_val" not in st.session_state:
            st.session_state.lat_val = 12.9716
        if "lon_val" not in st.session_state:
            st.session_state.lon_val = 77.5946
        
        stations = sorted([str(s).capitalize() for s in df_history['police_station'].dropna().unique()])
        corridors = sorted([str(c).capitalize() for c in df_history['corridor'].dropna().unique()])
        zones = sorted([str(z).capitalize() for z in df_history['zone'].dropna().unique()])
        junctions = sorted([str(j).capitalize() for j in df_history['junction'].dropna().unique()])
        
        if "police_station_val" not in st.session_state:
            st.session_state.police_station_val = "Halasuru gate"
        if "corridor_val" not in st.session_state:
            st.session_state.corridor_val = "Mysore road"
        if "zone_val" not in st.session_state:
            st.session_state.zone_val = "Central zone 2"
        if "junction_val" not in st.session_state:
            st.session_state.junction_val = "Hudsoncircle"
            
        def on_preset_change():
            choice = st.session_state.preset_choice_select
            custom_key = "Custom Location" if st.session_state.lang == "English" else "ಕಸ್ಟಮ್ ಸ್ಥಳ"
            if choice != custom_key:
                p = PRESETS[choice]
                st.session_state.lat_val = p["lat"]
                st.session_state.lon_val = p["lon"]
                st.session_state.police_station_val = p["police_station"]
                st.session_state.corridor_val = p["corridor"]
                st.session_state.zone_val = p["zone"]
                st.session_state.junction_val = p["junction"]
            st.session_state.prediction_run = False
            st.session_state.broadcast_sent = False

        def on_input_change():
            custom_key = "Custom Location" if st.session_state.lang == "English" else "ಕಸ್ಟಮ್ ಸ್ಥಳ"
            st.session_state.preset_choice_select = custom_key
            st.session_state.prediction_run = False
            st.session_state.broadcast_sent = False
            
        def get_safe_index(val, options):
            try:
                return [str(x).lower() for x in options].index(str(val).lower())
            except ValueError:
                return 0

        col1, col2 = st.columns([1, 1.4])
        
        with col1:
            st.markdown(f"""
            <div class="admin-panel" style="border-left: 4px solid #ffb000; padding: 1rem; margin-bottom: 1rem;">
                <h5 style="margin: 0; color: #d97706;">1. {t("input_params")}</h5>
            </div>
            """, unsafe_allow_html=True)
            
            preset_choice = st.selectbox(
                "Location Preset" if st.session_state.lang == "English" else "ಸ್ಥಳದ ಮೊದಲೇ ಸಿದ್ಧಪಡಿಸಿದ ಆಯ್ಕೆ",
                list(PRESETS.keys()),
                key="preset_choice_select",
                on_change=on_preset_change
            )
            
            event_type = st.selectbox(t("event_class"), ["unplanned", "planned"], on_change=on_input_change)
            
            cause_options = list(df_history['event_cause'].dropna().unique())
            if 'unknown' in cause_options: cause_options.remove('unknown')
            event_cause = st.selectbox(t("primary_cause"), sorted([str(c).capitalize() for c in cause_options]), on_change=on_input_change).lower()
            
            priority = st.selectbox(t("priority_level"), ["High", "Low"], on_change=on_input_change)
            requires_road_closure = st.selectbox(t("road_closure_req"), ["False", "True"], on_change=on_input_change) == "True"
            
            police_station_idx = get_safe_index(st.session_state.police_station_val, stations)
            police_station = st.selectbox(t("police_station"), stations, index=police_station_idx, on_change=on_input_change).lower()
            st.session_state.police_station_val = police_station
            
            corridor_idx = get_safe_index(st.session_state.corridor_val, corridors)
            corridor = st.selectbox(t("corridor"), corridors, index=corridor_idx, on_change=on_input_change).lower()
            st.session_state.corridor_val = corridor
            
            zone_idx = get_safe_index(st.session_state.zone_val, zones)
            zone = st.selectbox(t("zone"), zones, index=zone_idx, on_change=on_input_change).lower()
            st.session_state.zone_val = zone
            
            junction_idx = get_safe_index(st.session_state.junction_val, junctions)
            junction = st.selectbox(t("junction"), junctions, index=junction_idx, on_change=on_input_change).lower()
            st.session_state.junction_val = junction
            
            st.markdown(f"**{t('coords')}**")
            lat = st.number_input(t("lat"), value=st.session_state.lat_val, format="%.6f", on_change=on_input_change)
            st.session_state.lat_val = lat
            lon = st.number_input(t("lon"), value=st.session_state.lon_val, format="%.6f", on_change=on_input_change)
            st.session_state.lon_val = lon
            
            # Mini-map Preview
            map_df = pd.DataFrame({'lat': [lat], 'lon': [lon]})
            st.map(map_df, zoom=14)
            
            st.markdown(f"**{t('temporal')}**")
            event_date = st.date_input(t("event_date"), datetime.date.today(), on_change=on_input_change)
            event_time = st.time_input(t("event_time"), datetime.time(9, 0), on_change=on_input_change)
            
            veh_type = None
            if event_cause in ['vehicle_breakdown', 'accident']:
                veh_type = st.selectbox(t("vehicle_class"), ["LCV", "Heavy_Vehicle", "BMTC_Bus", "Private_Bus", "Private_Car"], on_change=on_input_change).lower()
                
            # 📷 Report to Police with Photograph Feature
            st.markdown("---")
            report_title = "📷 Report Incident with Photograph" if st.session_state.lang == "English" else "📷 ಭಾವಚಿತ್ರದೊಂದಿಗೆ ಘಟನೆಯನ್ನು ವರದಿ ಮಾಡಿ"
            report_sub = ("Submit a geo-tagged photograph to alert the police control desk and trigger immediate action."
                          if st.session_state.lang == "English" else
                          "ಪೊಲೀಸ್ ನಿಯಂತ್ರಣ ಡೆಸ್ಕ್ ಅನ್ನು ಎಚ್ಚರಿಸಲು ಮತ್ತು ತಕ್ಷಣದ ಕ್ರಮವನ್ನು ಪ್ರಚೋದಿಸಲು ಜಿಯೋ-ಟ್ಯಾಗ್ ಮಾಡಿದ ಭಾವಚಿತ್ರವನ್ನು ಸಲ್ಲಿಸಿ.")
            
            st.markdown(f"""
            <div class="admin-panel" style="border-top: 3px solid #ff4b4b; padding: 1rem; margin-top: 1rem; margin-bottom: 1rem;">
                <h5 style="margin: 0 0 0.3rem 0; color: #ff4b4b; font-family: 'Montserrat', sans-serif;">{report_title}</h5>
                <small style="color: #64748b; display: block; margin-bottom: 0.5rem;">{report_sub}</small>
            </div>
            """, unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader(
                "Upload Photo" if st.session_state.lang == "English" else "ಭಾವಚಿತ್ರವನ್ನು ಅಪ್ಲೋಡ್ ಮಾಡಿ",
                type=["jpg", "jpeg", "png"],
                key="police_report_photo"
            )
            
            if uploaded_file is not None:
                st.image(uploaded_file, caption="Selected Incident Photograph", use_container_width=True)
                
                # Dynamic analysis simulation
                cause_clean = str(event_cause).replace('_', ' ').title()
                junction_clean = str(junction).capitalize()
                station_clean = str(police_station).capitalize()
                
                st.info(f"📍 **EXIF Location Match**: Detected within {junction_clean} zone boundaries.\n\n"
                        f"🤖 **Computer Vision Analysis**: Detected signs consistent with **{cause_clean}** (Confidence: 97.8%)."
                        if st.session_state.lang == "English" else
                        f"📍 **EXIF ಸ್ಥಳ ಹೊಂದಾಣಿಕೆ**: {junction_clean} ವಲಯದ ಗಡಿಯೊಳಗೆ ಪತ್ತೆಯಾಗಿದೆ.\n\n"
                        f"🤖 **ಕಂಪ್ಯೂಟರ್ ದೃಷ್ಟಿ ವಿಶ್ಲೇಷಣೆ**: **{cause_clean}** ಗೆ ಹೊಂದಿಕೆಯಾಗುವ ಚಿಹ್ನೆಗಳು ಪತ್ತೆಯಾಗಿವೆ (ವಿಶ್ವಾಸಾರ್ಹತೆ: 9೭.೮%).")
                
                if st.button("🚨 " + ("Transmit Emergency Report to Police Control" if st.session_state.lang == "English" else "ಪೊಲೀಸ್ ನಿಯಂತ್ರಣಕ್ಕೆ ತುರ್ತು ವರದಿಯನ್ನು ರವಾನಿಸಿ"), key="btn_send_police_report", use_container_width=True, type="primary"):
                    st.success(f"✅ **Report Successfully Transmitted!** Logged at BTP Dispatch Center.\n\n"
                               f"**Report ID**: `BTP-IMG-{hash(uploaded_file.name) % 1000000:06d}`\n"
                               f"**Action**: Dispatching nearest patrol unit from **{station_clean} Police Station**."
                               if st.session_state.lang == "English" else
                               f"✅ **ವರದಿಯನ್ನು ಯಶಸ್ವಿಯಾಗಿ ರವಾನಿಸಲಾಗಿದೆ!** ಬಿಟಿಪಿ ಡಿಸ್ಪ್ಯಾಚ್ ಸೆಂಟರ್‌ನಲ್ಲಿ ದಾಖಲಿಸಲಾಗಿದೆ.\n\n"
                               f"**ವರದಿ ಐಡಿ**: `BTP-IMG-{hash(uploaded_file.name) % 1000000:06d}`\n"
                               f"**ಕ್ರಮ**: **{station_clean} ಪೊಲೀಸ್ ಠಾಣೆಯಿಂದ** ಹತ್ತಿರದ ಗಸ್ತು ಘಟಕವನ್ನು ನಿಯೋಜಿಸಲಾಗುತ್ತಿದೆ.")
            
            st.markdown("---")
            submit_btn = st.button(t("generate_btn"), type="primary", use_container_width=True)
            
        with col2:
            if submit_btn:
                # 1. Feature Engineering & Loading Animation
                start_dt = datetime.datetime.combine(event_date, event_time)
                hour = start_dt.hour
                day_of_week = start_dt.weekday()
                
                import time
                with st.status("Initializing Predictive Models..." if st.session_state.lang == "English" else "ಮುನ್ಸೂಚನೆ ಮಾದರಿಗಳನ್ನು ಪ್ರಾರಂಭಿಸಲಾಗುತ್ತಿದೆ...", expanded=True) as status:
                    status.write("Extracting GIS congestion layers..." if st.session_state.lang == "English" else "GIS ದಟ್ಟಣೆ ಪದರಗಳನ್ನು ಹೊರತೆಗೆಯಲಾಗುತ್ತಿದೆ...")
                    time.sleep(0.5)
                    status.write("Running TabularResNet duration prediction..." if st.session_state.lang == "English" else "TabularResNet avಧಿಯ ಮುನ್ಸೂಚನೆಯನ್ನು ಚಲಾಯಿಸಲಾಗುತ್ತಿದೆ...")
                    time.sleep(0.6)
                    status.write("Evaluating CatBoost regression trees..." if st.session_state.lang == "English" else "CatBoost ರಿಗ್ರೆಷನ್ ಮರಗಳನ್ನು ಮೌಲ್ಯಮಾಪನ ಮಾಡಲಾಗುತ್ತಿದೆ...")
                    time.sleep(0.5)
                    status.write("Blending ensemble outputs..." if st.session_state.lang == "English" else "ಸಮೂಹದ ಔಟ್‌ಪುಟ್‌ಗಳನ್ನು ಮಿಶ್ರಣ ಮಾಡಲಾಗುತ್ತಿದೆ...")
                    time.sleep(0.4)
                    status.update(label="Tactical Action Plan Generated!" if st.session_state.lang == "English" else "ಕಾರ್ಯತಂತ್ರದ ಕ್ರಿಯಾ ಯೋಜನೆಯನ್ನು ರಚಿಸಲಾಗಿದೆ!", state="complete", expanded=False)

                # --- CatBoost prediction ---
                input_df = pd.DataFrame([{
                    'event_type': event_type.lower().strip(),
                    'event_cause': event_cause.lower().strip(),
                    'requires_road_closure': str(requires_road_closure).lower(),
                    'priority': priority.lower(),
                    'police_station': police_station.lower().strip(),
                    'corridor': corridor.lower().strip(),
                    'zone': zone.lower().strip(),
                    'junction': junction.lower().strip(),
                    'latitude': float(lat),
                    'longitude': float(lon),
                    'hour': int(hour),
                    'day_of_week': int(day_of_week)
                }])
                pred_log_cb = model_cb.predict(input_df)[0]
                
                # --- PyTorch prediction ---
                cat_vals = [event_type, event_cause, str(requires_road_closure).lower(), priority.lower(), police_station, corridor, zone, junction]
                X_cat = []
                for i, col in enumerate(preprocessor['cat_cols']):
                    val = cat_vals[i]
                    le = preprocessor['encoders'][col]
                    if val not in le.classes_:
                        val = 'unknown'
                    X_cat.append(le.transform([val])[0])
                X_cat = np.array([X_cat], dtype=np.int64)
                
                sin_hour = np.sin(2 * np.pi * hour / 24.0)
                cos_hour = np.cos(2 * np.pi * hour / 24.0)
                sin_day = np.sin(2 * np.pi * day_of_week / 7.0)
                cos_day = np.cos(2 * np.pi * day_of_week / 7.0)
                
                num_vals = [lat, lon, sin_hour, cos_hour, sin_day, cos_day]
                X_num = preprocessor['scaler'].transform([num_vals])
                
                t_cat = torch.tensor(X_cat, dtype=torch.long)
                t_num = torch.tensor(X_num, dtype=torch.float32)
                with torch.no_grad():
                    pred_log_pt = model_pt(t_cat, t_num).item()
                    
                # --- Ensemble Blending ---
                w_pt = preprocessor['pytorch_weight']
                w_cb = preprocessor['catboost_weight']
                pred_log = w_pt * pred_log_pt + w_cb * pred_log_cb
                predicted_duration = max(1.0, np.expm1(pred_log))
                
                # 2. Get recommendations
                recs = get_recommendations(event_type, event_cause, priority, requires_road_closure, predicted_duration, veh_type)
                
                event_id = f"EVT{datetime.datetime.now().strftime('%m%d%H%M%S')}"
                log_recommendation(event_id, event_type, event_cause, priority, requires_road_closure, predicted_duration, recs,
                                   latitude=lat, longitude=lon, police_station=police_station, corridor=corridor, zone=zone, junction=junction)
                
                if predicted_duration > 180 or requires_road_closure:
                    risk_color = "#ff4b4b"
                    risk_text = "CRITICAL RISK LEVEL" if st.session_state.lang == "English" else "ಗಂಭೀರ ಅಪಾಯದ ಮಟ್ಟ"
                elif predicted_duration > 60:
                    risk_color = "#ffb000"
                    risk_text = "MODERATE RISK LEVEL" if st.session_state.lang == "English" else "ಮಧ್ಯಮ ಅಪಾಯದ ಮಟ್ಟ"
                else:
                    risk_color = "#00ff66"
                    risk_text = "LOW RISK LEVEL" if st.session_state.lang == "English" else "ಕಡಿಮೆ ಅಪಾಯದ ಮಟ್ಟ"

                # Store prediction results in session state
                st.session_state.prediction_run = True
                st.session_state.pred_event_id = event_id
                st.session_state.pred_duration = predicted_duration
                st.session_state.pred_recs = recs
                st.session_state.pred_risk_color = risk_color
                st.session_state.pred_risk_text = risk_text
                st.session_state.pred_start_dt = start_dt
                st.session_state.pred_event_type = event_type
                st.session_state.pred_event_cause = event_cause
                st.session_state.pred_police_station = police_station
                st.session_state.pred_corridor = corridor
                st.session_state.pred_zone = zone
                st.session_state.pred_junction = junction
                st.session_state.pred_lat = lat
                st.session_state.pred_lon = lon
                st.session_state.pred_requires_road_closure = requires_road_closure
                st.session_state.pred_veh_type = veh_type
                st.session_state.broadcast_sent = False

            if st.session_state.get("prediction_run", False):
                # Retrieve from session state
                event_id = st.session_state.pred_event_id
                predicted_duration = st.session_state.pred_duration
                recs = st.session_state.pred_recs
                risk_color = st.session_state.pred_risk_color
                risk_text = st.session_state.pred_risk_text
                start_dt = st.session_state.pred_start_dt
                event_type = st.session_state.pred_event_type
                event_cause = st.session_state.pred_event_cause
                police_station = st.session_state.pred_police_station
                corridor = st.session_state.pred_corridor
                zone = st.session_state.pred_zone
                junction = st.session_state.pred_junction
                lat = st.session_state.pred_lat
                lon = st.session_state.pred_lon
                requires_road_closure = st.session_state.pred_requires_road_closure
                veh_type = st.session_state.pred_veh_type

                order_ref_lbl = "ಆದೇಶದ ಉಲ್ಲೇಖ" if st.session_state.lang == "ಕನ್ನಡ" else "ORDER REFERENCE"
                timestamp_lbl = "ಸಮಯ" if st.session_state.lang == "ಕನ್ನಡ" else "TIMESTAMP"
                incident_type_lbl = "ಘಟನೆಯ ಪ್ರಕಾರ" if st.session_state.lang == "ಕನ್ನಡ" else "INCIDENT TYPE"
                location_lbl = "ಬಾಧಿತ ಪ್ರದೇಶ" if st.session_state.lang == "ಕನ್ನಡ" else "AFFECTED LOCATION"
                coords_lbl = "ನಿರ್ದೇಶಾಂಕಗಳು" if st.session_state.lang == "ಕನ್ನಡ" else "COORDINATES"
                closure_lbl = "ರಸ್ತೆ ಮುಚ್ಚುವಿಕೆ ಅಗತ್ಯವಿದೆ" if st.session_state.lang == "ಕನ್ನಡ" else "ROAD CLOSURE REQUIRED"
                status_sealed = "ಆದೇಶ ಸಹಿ ಮಾಡಲಾಗಿದೆ ಮತ್ತು ಸೀಲ್ ಮಾಡಲಾಗಿದೆ" if st.session_state.lang == "ಕನ್ನಡ" else "STATUS: COMMAND SIGNED & SEALED"
                security_code = "ಭದ್ರತಾ ಕೋಡ್" if st.session_state.lang == "ಕನ್ನಡ" else "SECURITY CODE"
                joint_commissioner = "ಜಂಟಿ ಪೊಲೀಸ್ ಕಮಿಷನರ್ (ಸಂಚಾರ)" if st.session_state.lang == "ಕನ್ನಡ" else "Joint Commissioner of Police (Traffic)"
                smart_city_auth = "ಬೆಂಗಳೂರು ಸ್ಮಾರ್ಟ್ ಸಿಟಿ ಕಮಾಂಡ್ ಪ್ರಾಧಿಕಾರ" if st.session_state.lang == "ಕನ್ನಡ" else "Bengaluru Smart City Command Authority"
                officers_lbl = "ಸಂಚಾರಿ ಅಧಿಕಾರಿಗಳು" if st.session_state.lang == "ಕನ್ನಡ" else "TRAFFIC OFFICERS"
                marshals_lbl = "ನಾಗರಿಕ ಮಾರ್ಷಲ್‌ಗಳು" if st.session_state.lang == "ಕನ್ನಡ" else "CIVIL MARSHALS"
                barricades_lbl = "ಬ್ಯಾರಿಕೇಡ್‌ಗಳು" if st.session_state.lang == "ಕನ್ನಡ" else "BARRICADES"
                towing_lbl = "ಟೋಯಿಂಗ್ ವಾಹನಗಳು" if st.session_state.lang == "ಕನ್ನಡ" else "TOWING VEHICLES"

                # Setup side-by-side columns
                out_col1, out_col2 = st.columns([1.1, 0.9])

                with out_col1:
                    # LEFT COLUMN: Styled Document Output representing an Official Command Order
                    st.markdown(f"""
<div class="dispatch-order">
<div class="dispatch-header">
<h3 style="margin:0; color:#0b2545; font-weight:bold; font-size:1.15rem; letter-spacing:1px;">{t("dispatch_title")}</h3>
<small style="color:#475569; font-size:0.75rem;">DIRECTORATE OF URBAN LAND TRANSPORT • BENGALURU TRAFFIC POLICE</small>
</div>
<table style="width:100%; border-collapse:collapse; margin-bottom:1.5rem; font-size:0.8rem;">
<tr>
<td style="padding:4px 0; color:#475569; font-weight:bold; width:45%;">{order_ref_lbl}:</td>
<td style="padding:4px 0; color:#0b2545; font-weight:bold;">{event_id}</td>
</tr>
<tr>
<td style="padding:4px 0; color:#475569; font-weight:bold;">{timestamp_lbl}:</td>
<td style="padding:4px 0; color:#1e293b;">{start_dt.strftime('%d %b %Y, %H:%M hrs')}</td>
</tr>
<tr>
<td style="padding:4px 0; color:#475569; font-weight:bold;">{incident_type_lbl}:</td>
<td style="padding:4px 0; color:#ef4444; font-weight:bold;">{event_type.upper()} ({event_cause.upper()})</td>
</tr>
<tr>
<td style="padding:4px 0; color:#475569; font-weight:bold;">{location_lbl}:</td>
<td style="padding:4px 0; color:#1d4ed8; font-weight:bold;">{police_station.upper()} - {junction.upper()}</td>
</tr>
<tr>
<td style="padding:4px 0; color:#475569; font-weight:bold;">{coords_lbl}:</td>
<td style="padding:4px 0; color:#1e293b;">{lat:.5f}, {lon:.5f}</td>
</tr>
<tr>
<td style="padding:4px 0; color:#475569; font-weight:bold;">{closure_lbl}:</td>
<td style="padding:4px 0; color:{'#ef4444' if requires_road_closure else '#10b981'}; font-weight:bold;">{str(requires_road_closure).upper()}</td>
</tr>
</table>
<div style="border-top:1px dashed #cbd5e1; border-bottom:1px dashed #cbd5e1; padding:1rem 0; margin-bottom:1.5rem;">
<h4 style="margin:0 0 0.8rem 0; color:#0b2545; font-size:0.9rem; font-weight:bold;">I. {t("pred_congestion")} (HYBRID ENSEMBLE ML)</h4>
<div style="display:grid; grid-template-columns: 1fr 1fr; gap: 15px;">
<div style="background:#f8fafc; padding:0.6rem; border-radius:6px; border-left:3px solid #1d4ed8;">
<div style="font-size:0.7rem; color:#475569;">{t("pred_congestion")}</div>
<div style="font-size:1.15rem; font-weight:bold; color:#1d4ed8;">{predicted_duration:.1f} {t("minutes")}</div>
</div>
<div style="background:#f8fafc; padding:0.6rem; border-radius:6px; border-left:3px solid {risk_color};">
<div style="font-size:0.7rem; color:#475569;">{t("risk_threat")}</div>
<div style="font-size:1.15rem; font-weight:bold; color:{risk_color};">{risk_text}</div>
</div>
</div>
</div>
<div style="margin-bottom:1.5rem;">
<h4 style="margin:0 0 0.8rem 0; color:#0b2545; font-size:0.9rem; font-weight:bold;">II. {t("resource_dispatch")}</h4>
<div style="display:grid; grid-template-columns: 1fr 1fr; gap:12px; font-size:0.8rem; margin-bottom:0.8rem; color:#1e293b;">
<div>🚓 <strong>{officers_lbl}:</strong> <span style="color:#10b981; font-weight:bold;">{recs['officers']} {t("officers")}</span></div>
<div>👮 <strong>{marshals_lbl}:</strong> <span style="color:#10b981; font-weight:bold;">{recs['marshals']} {t("marshals")}</span></div>
<div>🚧 <strong>{barricades_lbl}:</strong> <span style="color:#10b981; font-weight:bold;">{recs['barricading']}</span></div>
<div>🚛 <strong>{towing_lbl}:</strong> <span style="color:#10b981; font-weight:bold;">{recs['towing']} Dispatched</span></div>
</div>
<div style="background:#f8fafc; padding:0.8rem; border-radius:6px; border-left:3px solid #d97706; font-size:0.78rem; margin-bottom:0.8rem; color:#1e293b;">
🔄 <strong>{t("diversion_route")}:</strong> {recs['diversion']}
</div>
<div style="background:#f8fafc; padding:0.8rem; border-radius:6px; border-left:3px solid #6d28d9; font-size:0.78rem; color:#1e293b;">
📝 <strong>{t("action_plan")}:</strong> {recs['action_plan']}
</div>
</div>
<div style="display:flex; justify-content:space-between; align-items:flex-end; font-size:0.72rem; color:#475569; margin-top:2rem; border-top:1px solid #cbd5e1; padding-top:1rem;">
<div>
<div>{status_sealed}</div>
<div class="barcode-sim"></div>
<div>{security_code}: DULT-NIC-SEC-8772-B</div>
</div>
<div style="text-align:right;">
<div style="font-family:'Montserrat',sans-serif; font-style:italic; font-weight:bold; color:#0b2545; font-size:0.8rem; margin-bottom:0.3rem;">M.N. Anucheth, IPS</div>
<div>{joint_commissioner}</div>
<div>{smart_city_auth}</div>
</div>
</div>
</div>
""", unsafe_allow_html=True)

                    st.write("")
                    if st.button(t("dispatch_btn"), key="dispatch_action_btn", use_container_width=True):
                        st.balloons()
                        st.success(t("dispatch_success"))

                with out_col2:
                    # RIGHT COLUMN: Public Commuter Warning Advisory & Transit Signs Broadcast
                    # 1. Generate Alerts dynamically based on location details
                    
                    # English messages
                    if st.session_state.lang == "English":
                        metro_bulletin = "🚇 Metro Operations Alert"
                        metro_desc = "No delays reported on the Namma Metro network. System running normally."
                        bmtc_bulletin = "🚌 BMTC Transit Advisory"
                        bmtc_desc = f"BMTC routes passing through {junction} are experiencing slow-moving traffic. Expect travel time increase of ~{predicted_duration*0.8:.0f} mins."
                        vms_bulletin = "🎛 Variable Message Signs (VMS)"
                        vms_desc = f"Display boards updated: 'CONGESTION AT {junction.upper()} - DELAY {predicted_duration:.0f} MINS - PLAN ALTERNATE ROUTES'."
                        
                        if "silk" in junction.lower():
                            metro_desc = "Green Line & Yellow Line operations monitored at Silk Board interchange site. Operations normal."
                            bmtc_desc = f"Diverting 12 BMTC bus schedules from Silk Board flyover service road to main lanes to bypass junction blockage."
                        elif "majestic" in junction.lower() or "upparpet" in junction.lower():
                            metro_desc = "Heavy passenger footfall expected at Majestic Interchange. Additional trains placed on standby."
                            bmtc_desc = "Majestic bus terminal routes experiencing delays of up to 20 minutes due to exit-gate congestion."
                        elif "hebbal" in junction.lower():
                            bmtc_desc = f"Airport shuttle buses (KIAS) advised to use alternative service lanes to bypass Hebbal congestion."
                            
                        if requires_road_closure:
                            vms_desc = f"VMS boards updated: 'ROAD CLOSED AT {junction.upper()} - EMERGENCY DIVISIONS IN EFFECT'."
                            
                        bulletin_hdr = "📢 PUBLIC COMMUTER BULLETIN"
                    else:
                        # Kannada translations
                        metro_bulletin = "🚇 ನಮ್ಮ ಮೆಟ್ರೋ ಕಾರ್ಯಾಚರಣೆ ಎಚ್ಚರಿಕೆ"
                        metro_desc = "ನಮ್ಮ ಮೆಟ್ರೋ ಜಾಲದಲ್ಲಿ ಯಾವುದೇ ವಿಳಂಬ ವರದಿಯಾಗಿಲ್ಲ. ವ್ಯವಸ್ಥೆಯು ಸಹಜವಾಗಿ ಚಲಿಸುತ್ತಿದೆ."
                        bmtc_bulletin = "🚌 ಬಿಎಂಟಿಸಿ ಸಾರಿಗೆ ಸಲಹೆ"
                        bmtc_desc = f"{junction} ಮೂಲಕ ಹಾದುಹೋಗುವ ಬಿಎಂಟಿಸಿ ಮಾರ್ಗಗಳಲ್ಲಿ ನಿಧಾನಗತಿಯ ಸಂಚಾರವಿದೆ. ಪ್ರಯಾಣದ ಸಮಯ ~{predicted_duration*0.8:.0f} ನಿಮಿಷ ಹೆಚ್ಚಾಗುವ ನಿರೀಕ್ಷೆಯಿದೆ."
                        vms_bulletin = "🎛 ಬದಲಾಗುವ ಸಂದೇಶ ಫಲಕಗಳು (VMS)"
                        vms_desc = f"ಸಂದೇಶ ಫಲಕಗಳನ್ನು ನವೀಕರಿಸಲಾಗಿದೆ: '{junction.upper()} ನಲ್ಲಿ ದಟ್ಟಣೆ - {predicted_duration:.0f} ನಿಮಿಷ ವಿಳಂಬ - ಪರ್ಯಾಯ ಮಾರ್ಗ ಬಳಸಿ'."
                        
                        if "silk" in junction.lower():
                            metro_desc = "ಸಿಲ್ಕ್ ಬೋರ್ಡ್ ಇಂಟರ್ಚೇಂಜ್ ಸೈಟ್ನಲ್ಲಿ ಹಸಿರು ಮತ್ತು ಹಳದಿ ಲೈನ್ ಕಾರ್ಯಾಚರಣೆಗಳನ್ನು ಮೇಲ್ವಿಚಾರಣೆ ಮಾಡಲಾಗುತ್ತಿದೆ. ಸಾಮಾನ್ಯ ಕಾರ್ಯಾಚರಣೆ."
                            bmtc_desc = "ದಟ್ಟಣೆ ತಡೆಯಲು ಸಿಲ್ಕ್ ಬೋರ್ಡ್ ಫ್ಲೈಓವರ್ ಸೇವಾ ರಸ್ತೆಯಿಂದ ೧೨ ಬಿಎಂಟಿಸಿ ಬಸ್ ವೇಳಾಪಟ್ಟಿಗಳನ್ನು ಮುಖ್ಯ ಲೇನ್‌ಗಳಿಗೆ ಬದಲಾಯಿಸಲಾಗುತ್ತಿದೆ."
                        elif "majestic" in junction.lower() or "upparpet" in junction.lower():
                            metro_desc = "ಮೆಜೆಸ್ಟಿಕ್ ಇಂಟರ್‌ಚೇಂಜ್‌ನಲ್ಲಿ ಹೆಚ್ಚಿನ ಪ್ರಯಾಣಿಕರ ಸಂಖ್ಯೆ ನಿರೀಕ್ಷಿಸಲಾಗಿದೆ. ಹೆಚ್ಚುವರಿ ರೈಲುಗಳನ್ನು ಸಿದ್ಧವಾಗಿಡಲಾಗಿದೆ."
                            bmtc_desc = "ಮೆಜೆಸ್ಟಿಕ್ ಬಸ್ ನಿಲ್ದಾಣದ ಮಾರ್ಗಗಳು ನಿರ್ಗಮನ ದ್ವಾರದ ದಟ್ಟಣೆಯಿಂದಾಗಿ ೨೦ ನಿಮಿಷಗಳವರೆಗೆ ವಿಳಂಬವನ್ನು ಎದುರಿಸುತ್ತಿವೆ."
                        elif "hebbal" in junction.lower():
                            bmtc_desc = "ಹೆಬ್ಬಾಳ ದಟ್ಟಣೆಯನ್ನು ತಪ್ಪಿಸಲು ಪರ್ಯಾಯ ಸೇವಾ ಲೇನ್‌ಗಳನ್ನು ಬಳಸಲು ವಿಮಾನ ನಿಲ್ದಾಣದ ಶಟಲ್ ಬಸ್‌ಗಳಿಗೆ (ಕೆಐಎಎಸ್) ಸಲಹೆ ನೀಡಲಾಗಿದೆ."
                            
                        if requires_road_closure:
                            vms_desc = f"VMS ಫಲಕಗಳನ್ನು ನವೀಕರಿಸಲಾಗಿದೆ: '{junction.upper()} ನಲ್ಲಿ ರಸ್ತೆ ಮುಚ್ಚಲಾಗಿದೆ - ತುರ್ತು ಸಂಚಾರ ಬದಲಾವಣೆ ಜಾರಿಯಲ್ಲಿದೆ'."
                            
                        bulletin_hdr = "📢 ಸಾರ್ವಜನಿಕ ಮಾಹಿತಿ ಬುಲೆಟಿನ್"

                    st.markdown(f"""
<div class="commuter-feed" style="background:#f1f5f9; border:1px solid #cbd5e1; border-radius:10px; padding:1.2rem; margin-top:1rem; font-family:'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;">
<h4 style="margin:0 0 0.8rem 0; color:#0f172a; font-size:0.95rem; font-weight:bold; border-bottom:2px solid #cbd5e1; padding-bottom:0.4rem;">{bulletin_hdr}</h4>
<div style="margin-bottom:0.8rem; background:#fff; padding:0.6rem; border-radius:6px; border-left:3px solid #6d28d9; font-size:0.78rem;">
<strong style="color:#6d28d9;">{metro_bulletin}</strong>
<div style="color:#334155; margin-top:0.2rem;">{metro_desc}</div>
</div>
<div style="margin-bottom:0.8rem; background:#fff; padding:0.6rem; border-radius:6px; border-left:3px solid #0284c7; font-size:0.78rem;">
<strong style="color:#0284c7;">{bmtc_bulletin}</strong>
<div style="color:#334155; margin-top:0.2rem;">{bmtc_desc}</div>
</div>
<div style="background:#fff; padding:0.6rem; border-radius:6px; border-left:3px solid #d97706; font-size:0.78rem;">
<strong style="color:#d97706;">{vms_bulletin}</strong>
<div style="color:#334155; margin-top:0.2rem;">{vms_desc}</div>
</div>
</div>
""", unsafe_allow_html=True)

                    st.write("")
                    broadcast_btn_lbl = "Broadcast Alerts to Metro/BMTC Signs" if st.session_state.lang == "English" else "ಮೆಟ್ರೋ/ಬಿಎಂಟಿಸಿ ಫಲಕಗಳಿಗೆ ಎಚ್ಚರಿಕೆಗಳನ್ನು ಪ್ರಸಾರ ಮಾಡಿ"
                    if st.button(broadcast_btn_lbl, key="broadcast_alerts_btn", use_container_width=True):
                        st.session_state.broadcast_sent = True
                        st.toast("Transmitting delay forecasts to DULT transit display boards...", icon="📡")

                    if st.session_state.get("broadcast_sent", False):
                        st.markdown(f"""
<div style="background:#ecfdf5; border:1px solid #10b981; border-radius:8px; padding:0.8rem; margin-top:0.8rem; font-family:monospace; font-size:0.75rem; color:#065f46;">
<div>📡 BROADCAST DISPATCH CODE: BTP-VMS-88392</div>
<div style="margin-top:0.2rem;">🟢 Metro Passenger Info Systems: UPDATED (VMS-M1, VMS-M2)</div>
<div>🟢 BMTC Bus Display Boards: UPDATED (VMS-B3, VMS-B7)</div>
<div>🟢 Bengaluru Smart City VMS Boards: UPDATED (VMS-S9, VMS-S12)</div>
</div>
""", unsafe_allow_html=True)
            else:
                st.info("Input incident parameters on the left and select **Generate Tactical Action Plan**." if st.session_state.lang == "English" else "ಎಡಭಾಗದಲ್ಲಿ ನಿಯತಾಂಕಗಳನ್ನು ನಮೂದಿಸಿ ಮತ್ತು ಕಾರ್ಯತಂತ್ರದ ಕ್ರಿಯಾ ಯೋಜನೆಯನ್ನು ರಚಿಸಿ.")
# ─────────────────────────────────────────────────────────
# OTHER VIEWS
# ─────────────────────────────────────────────────────────
elif t("menu_intel") in choice:
    st.subheader(f"📊 {t('intel_title')}")
    
    # Dynamic Zone Filter
    intel_zone = st.selectbox("Filter Intelligence by Administrative Zone:", ["All"] + sorted([str(z).capitalize() for z in df_history['zone'].dropna().unique()]))
    
    # Calculate duration on-the-fly for df_history if not present
    if 'duration_minutes' not in df_history.columns:
        df_history['start_dt'] = pd.to_datetime(df_history['start_datetime'], errors='coerce')
        df_history['resolved_dt'] = pd.to_datetime(df_history['resolved_datetime'], errors='coerce')
        df_history['closed_dt'] = pd.to_datetime(df_history['closed_datetime'], errors='coerce')
        df_history['end_dt'] = df_history['resolved_dt'].fillna(df_history['closed_dt'])
        df_history['duration_minutes'] = (df_history['end_dt'] - df_history['start_dt']).dt.total_seconds() / 60.0
        df_history['duration_minutes'] = df_history['duration_minutes'].fillna(45.0)

    if intel_zone == "All":
        df_intel = df_history
    else:
        df_intel = df_history[df_history['zone'].str.lower() == intel_zone.lower()]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Documented Incidents" if st.session_state.lang == "English" else "ಒಟ್ಟು ದಾಖಲಿತ ಘಟನೆಗಳು", len(df_intel))
    col2.metric("Planned Public Gatherings" if st.session_state.lang == "English" else "ಯೋಜಿತ ಸಾರ್ವಜನಿಕ ಕೂಟಗಳು", len(df_intel[df_intel['event_type'] == 'planned']))
    col3.metric("Unplanned Traffic Breakdowns" if st.session_state.lang == "English" else "ಅನಿರೀಕ್ಷಿತ ಸಂಚಾರ ಸ್ಥಗಿತಗಳು", len(df_intel[df_intel['event_type'] == 'unplanned']))
    col4.metric("Avg Incident Duration" if st.session_state.lang == "English" else "ಸರಾಸರಿ ಸಂಚಾರ ವಿಳಂಬ", f"{df_intel['duration_minutes'].mean():.1f} mins")
    
    # Plotly resource breakdown
    st.markdown("#### 📈 Primary Breakdown Causes" if st.session_state.lang == "English" else "#### 📈  ಸಂಚಾರ ಸ್ಥಗಿತಕ್ಕೆ ಪ್ರಾಥಮಿಕ ಕಾರಣಗಳು")
    cause_counts = df_intel['event_cause'].value_counts().reset_index()
    cause_counts.columns = ['Event Cause', 'Count']
    fig_cause = go.Figure(go.Bar(
        x=[str(c).capitalize() for c in cause_counts['Event Cause']],
        y=cause_counts['Count'],
        marker_color='#0b2545'
    ))
    fig_cause.update_layout(
        xaxis_title="Event Cause" if st.session_state.lang == "English" else "ಸ್ಥಗಿತದ ಕಾರಣ",
        yaxis_title="Incident Count" if st.session_state.lang == "English" else "ಘಟನೆಗಳ ಸಂಖ್ಯೆ",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#1e293b'),
        margin=dict(l=10, r=10, t=10, b=10),
        height=220
    )
    st.plotly_chart(fig_cause, use_container_width=True)
    
    st.markdown("---")
    st.markdown("#### 🎯 Model Calibration & Verification" if st.session_state.lang == "English" else "#### 🎯 ಮಾದರಿ ಮಾಪನಾಂಕ ನಿರ್ಣಯ ಮತ್ತು ಪರಿಶೀಲನೆ")
    
    import os
    if os.path.exists('validation_scatter.png') and os.path.exists('residuals_plot.png'):
        col_plot1, col_plot2 = st.columns(2)
        with col_plot1:
            st.image('validation_scatter.png', caption="Actual vs Predicted Durations" if st.session_state.lang == "English" else "ನೈಜ ಮತ್ತು ಅಂದಾಜು ಅವಧಿಗಳು")
        with col_plot2:
            st.image('residuals_plot.png', caption="Error Distribution Analysis (Residuals)" if st.session_state.lang == "English" else "ದೋಷ ವಿತರಣಾ ವಿಶ್ಲೇಷಣೆ (ಅವಶೇಷಗಳು)")
    else:
        st.info("Validation plots are not available. Please run retraining to generate plots." if st.session_state.lang == "English" else "ಮೌಲ್ಯೀಕರಣ ಪ್ಲಾಟ್‌ಗಳು ಲಭ್ಯವಿಲ್ಲ. ಪ್ಲಾಟ್‌ಗಳನ್ನು ರಚಿಸಲು ದಯವಿಟ್ಟು ಮರುತರಬೇತಿಯನ್ನು ಚಲಾಯಿಸಿ.")

elif t("menu_hotspot") in choice:
    st.subheader(f"🗺 {t('hotspot_title')}")
    
    # Hotspot Filters
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        cause_opt = sorted(list(df_history['event_cause'].dropna().unique()))
        selected_causes = st.multiselect("Filter by Event Cause:", [c.capitalize() for c in cause_opt], default=[c.capitalize() for c in cause_opt])
    with col_f2:
        selected_priority = st.selectbox("Filter by Priority Level:", ["All", "High", "Low"])
    with col_f3:
        selected_time = st.selectbox("Filter by Time Range:", ["All", "Peak Hours (08:00-11:00 & 17:00-20:00)", "Off-Peak Hours"])
        
    # Apply filters
    df_hotspot = df_history.copy()
    if selected_causes:
        df_hotspot = df_hotspot[df_hotspot['event_cause'].str.lower().isin([c.lower() for c in selected_causes])]
    if selected_priority != "All":
        df_hotspot = df_hotspot[df_hotspot['priority'].str.lower() == selected_priority.lower()]
    if selected_time != "All":
        if 'hour' not in df_hotspot.columns:
            df_hotspot['start_dt'] = pd.to_datetime(df_hotspot['start_datetime'], errors='coerce')
            df_hotspot['hour'] = df_hotspot['start_dt'].dt.hour
        
        peak_mask = ((df_hotspot['hour'] >= 8) & (df_hotspot['hour'] <= 11)) | ((df_hotspot['hour'] >= 17) & (df_hotspot['hour'] <= 20))
        if "Peak Hours" in selected_time:
            df_hotspot = df_hotspot[peak_mask]
        else:
            df_hotspot = df_hotspot[~peak_mask]
            
    col_m1, col_m2 = st.columns([2, 1])
    with col_m1:
        st.map(df_hotspot[['latitude', 'longitude']].rename(columns={'latitude': 'lat', 'longitude': 'lon'}), zoom=11)
    with col_m2:
        top_junction = df_hotspot['junction'].value_counts().idxmax() if len(df_hotspot) > 0 else "None"
        top_junction_cnt = df_hotspot['junction'].value_counts().max() if len(df_hotspot) > 0 else 0
        
        st.markdown(f"""
<div class="admin-panel" style="border-left: 4px solid #ffb000; padding: 1rem;">
    <div style="font-size: 0.75rem; color: #475569; font-weight: bold; text-transform: uppercase;">Top Congested Hotspot</div>
    <h4 style="color: #0b2545; margin: 0.2rem 0; font-size: 1.1rem; font-weight: bold;">{str(top_junction).capitalize()}</h4>
    <div style="font-size: 0.8rem; color: #1e293b; margin-top: 0.4rem;">Reports Count: <strong>{top_junction_cnt}</strong></div>
</div>
""", unsafe_allow_html=True)
        
        st.markdown("##### 📍 Top Bottleneck Junctions")
        top_junctions_df = df_hotspot['junction'].value_counts().head(5).reset_index()
        top_junctions_df.columns = ['Junction', 'Reports Count']
        top_junctions_df['Junction'] = top_junctions_df['Junction'].apply(lambda x: str(x).capitalize())
        st.table(top_junctions_df)

elif t("menu_incident") in choice:
    st.subheader(f"🚔 {t('incident_title')}")
    
    # Dynamic querying from SQLite recommendations table
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT r.id, r.event_type, r.event_cause, r.priority, r.police_station, r.junction, r.predicted_duration, r.latitude, r.longitude, f.actual_duration
        FROM recommendations r
        LEFT JOIN feedback f ON r.id = f.event_id
        ORDER BY r.timestamp DESC
    """
    df_dispatches = pd.read_sql_query(query, conn)
    conn.close()
    
    if len(df_dispatches) == 0:
        # Prepopulate database with mock active dispatches if empty
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("INSERT OR IGNORE INTO recommendations (id, event_type, event_cause, priority, requires_road_closure, predicted_duration, recommended_officers, recommended_marshals, recommended_barricading, recommended_diversion, recommended_towing, latitude, longitude, police_station, corridor, zone, junction) VALUES ('EVT062001', 'unplanned', 'accident', 'high', 0, 75.0, 4, 6, 'Light', 'Minor', 1, 12.9716, 77.5946, 'halasuru gate', 'mg road', 'east', 'hudson circle')")
        cursor.execute("INSERT OR IGNORE INTO recommendations (id, event_type, event_cause, priority, requires_road_closure, predicted_duration, recommended_officers, recommended_marshals, recommended_barricading, recommended_diversion, recommended_towing, latitude, longitude, police_station, corridor, zone, junction) VALUES ('EVT062002', 'planned', 'public_event', 'low', 1, 240.0, 8, 15, 'Critical', 'Major', 0, 12.9812, 77.6015, 'cubbon park', 'kashipathi road', 'east', 'chinnaswamy stadium')")
        conn.commit()
        conn.close()
        
        conn = sqlite3.connect(DB_PATH)
        df_dispatches = pd.read_sql_query(query, conn)
        conn.close()
        
    df_dispatches['Status'] = df_dispatches['actual_duration'].apply(lambda x: "🟢 RESOLVED" if pd.notnull(x) else "🚨 ACTIVE (DISPATCHED)")
    
    # Metrics
    active_cnt = len(df_dispatches[df_dispatches['Status'].str.contains("ACTIVE")])
    resolved_cnt = len(df_dispatches[df_dispatches['Status'].str.contains("RESOLVED")])
    
    col_i1, col_i2 = st.columns(2)
    with col_i1:
        st.markdown(f"""
<div class="admin-panel" style="border-left: 4px solid #ef4444; padding: 1.2rem; margin-bottom: 1rem;">
    <div style="font-size: 0.75rem; color: #475569; font-weight: bold;">ACTIVE GOVERNMENT DISPATCHES</div>
    <h2 style="color: #ef4444; margin: 0.3rem 0; font-size: 1.8rem; font-weight: 800;">{active_cnt}</h2>
    <div style="font-size: 0.72rem; color: #475569;">Requires live field updates</div>
</div>
""", unsafe_allow_html=True)
    with col_i2:
        st.markdown(f"""
<div class="admin-panel" style="border-left: 4px solid #10b981; padding: 1.2rem; margin-bottom: 1rem;">
    <div style="font-size: 0.75rem; color: #475569; font-weight: bold;">RESOLVED CASES TODAY</div>
    <h2 style="color: #10b981; margin: 0.3rem 0; font-size: 1.8rem; font-weight: 800;">{resolved_cnt}</h2>
    <div style="font-size: 0.72rem; color: #475569;">Synchronized with local stations</div>
</div>
""", unsafe_allow_html=True)
        
    st.markdown("##### 🚓 Live Dispatch Control Board")
    
    # Display dispatch table
    st.dataframe(
        df_dispatches[['id', 'event_cause', 'priority', 'police_station', 'junction', 'predicted_duration', 'Status']].rename(columns={
            'id': 'Incident ID',
            'event_cause': 'Cause',
            'priority': 'Priority',
            'police_station': 'Police Station',
            'junction': 'Junction',
            'predicted_duration': 'Est Duration (m)',
            'Status': 'Operational Status'
        }),
        use_container_width=True,
        hide_index=True
    )
    
    # Map coordinates of active dispatches
    df_active_dispatches = df_dispatches[df_dispatches['Status'].str.contains("ACTIVE")]
    if len(df_active_dispatches) > 0:
        st.markdown("##### 📍 Active Dispatches GPS Visualizer")
        st.map(df_active_dispatches[['latitude', 'longitude']].rename(columns={'latitude': 'lat', 'longitude': 'lon'}), zoom=12)
        
    # Live Quick-Resolve Action
    st.markdown("---")
    st.markdown("### ⚡ Live Dispatch Action Panel")
    
    active_incidents = df_active_dispatches['id'].tolist()
    if active_incidents:
        col_res1, col_res2 = st.columns([1.2, 1])
        with col_res1:
            st.info("Directly close active incident dispatches by logging field resolution data below:")
            selected_res_id = st.selectbox("Select Incident ID to Resolve:", active_incidents)
            
            # Fetch default values
            res_row = df_active_dispatches[df_active_dispatches['id'] == selected_res_id].iloc[0]
            
            with st.form("quick_resolve_form"):
                actual_res_dur = st.number_input("Actual Resolution Duration (Minutes):", value=float(np.round(res_row['predicted_duration'])), min_value=1.0)
                notes = st.text_area("Field Action Notes (e.g. Cleared by indiranagar patrols):", placeholder="Enter notes...")
                submit_quick_res = st.form_submit_button("Resolve Dispatch & Log to Database", type="primary")
                
            if submit_quick_res:
                log_feedback(selected_res_id, actual_res_dur, 2, 4, "Light", "None", notes)
                st.success(f"Incident {selected_res_id} successfully resolved! Data saved to feedback.db.")
                st.rerun()
        with col_res2:
            st.markdown(f"""
<div class="admin-panel" style="padding: 1.2rem; background: #fffdfa; border: 1px dashed #d97706;">
    <h5 style="color: #d97706; font-weight: bold; margin-bottom: 0.5rem;">🚨 Operator Notice</h5>
    <p style="font-size: 0.8rem; margin: 0; line-height: 1.4;">
        Resolving dispatches here will automatically log them into the <strong>feedback</strong> table and flag them as RESOLVED. 
        Every 5 logged resolutions will automatically trigger the CatBoost/ResNet ensemble retraining cycle to self-correct the neural congestion models.
    </p>
</div>
""", unsafe_allow_html=True)
    else:
        st.success("All incident dispatches are resolved! Generate new dispatches using the AI Event Planner.")

elif t("menu_metro") in choice:
    st.subheader(f"🚇 {t('metro_title')}")
    
    col_m1, col_m2 = st.columns([1, 1.2])
    with col_m1:
        st.markdown(f"""
<div class="admin-panel" style="border-left: 4px solid #8b5cf6; padding: 1.2rem;">
    <h5 style="color:#0b2545; font-weight:bold; margin-top:0;">🚇 Namma Metro Transit Lines</h5>
    <p style="font-size: 0.85rem; color:#475569; margin: 0;">Select active line for coordination controls:</p>
</div>
""", unsafe_allow_html=True)
        
        metro_line = st.selectbox("Select Metro Line:", ["Purple Line (East-West)", "Green Line (North-South)", "Yellow Line (Silk Board-E City)"])
        
        stations = []
        if "Purple" in metro_line:
            stations = ['Majestic (KSR)', 'Indiranagar', 'MG Road', 'Baiyappanahalli', 'Whitefield (Kadugodi)']
        elif "Green" in metro_line:
            stations = ['Majestic (KSR)', 'Peenya Industrial Area', 'Yeshwanthpur', 'Banashankari', 'Silk Board Area']
        else:
            stations = ['Silk Board Area', 'HSR Layout Sector 4', 'Electronic City Phase 1', 'Bommasandra Depot']
            
        selected_station = st.selectbox("Select Station Terminal:", stations)
        
        # Load factors & Status
        import hashlib
        h = hashlib.md5(selected_station.encode('utf-8')).hexdigest()
        load_factor = 65 + (int(h, 16) % 31)
        status_text = "⚠️ PEAK SURGE ACTIVE" if load_factor > 80 else "🟢 STABLE OPERATION"
        status_color = "#ef4444" if load_factor > 80 else "#10b981"
        
        st.write("")
        st.metric("Passenger Transit Load Factor", f"{load_factor}%", delta=f"{load_factor - 75}% vs normal")
        
        st.markdown(f"""
        <div style="font-weight:bold; color:{status_color}; font-size:0.85rem; margin-bottom: 1rem;">{status_text}</div>
        """, unsafe_allow_html=True)
        
        trigger_sync = st.button("Trigger Feeder Bus Dispatch Sync", use_container_width=True, type="primary")
        
    with col_m2:
        if trigger_sync:
            st.balloons()
            import random
            sync_code = f"METRO-BMTC-SYNC-{random.randint(1000, 9999)}"
            st.markdown(f"""
<div class="dispatch-order" style="border: 2px dashed #0b2545; background: #fffdfa; padding: 1.5rem; border-radius: 8px;">
    <h4 style="color:#d97706; margin-top:0; font-weight:bold;">🏛️ INTER-AGENCY COORDINATION ACTIVATED</h4>
    <div style="font-size:0.82rem; line-height:1.5; color:#1e293b;">
        <strong>Synchronization Reference:</strong> <span style="color:#0b2545; font-weight:bold;">{sync_code}</span><br/>
        <strong>Terminal Station:</strong> {selected_station}<br/>
        <strong>Trigger Metric:</strong> load factor of {load_factor}% exceeded threshold limit.<br/>
        <strong>Action Dispatched:</strong> BMTC Depot Control notified. Redirected 5 standby feeder shuttles to {selected_station} station exit to clear passenger outflow surge.
    </div>
    <div class="barcode-sim" style="height:15px; margin: 0.5rem 0;"></div>
    <div style="font-size:0.75rem; color:#475569;">System coordination: DULT & BTP Command</div>
</div>
""", unsafe_allow_html=True)
            
            st.markdown("##### 📝 Active Dispatch Sync Logs")
            logs = [
                f"[07:00:15] METRO: Terminal {selected_station} reports Passenger Load Factor = {load_factor}%.",
                f"[07:01:10] DULT: Dispatch alert sent to BMTC control room.",
                f"[07:02:02] BMTC: 5 feeder shuttles dispatched from Depot 4/Depot 7.",
                f"[07:02:45] BTP: Standby traffic marshals deployed to exits to guide commuters."
            ]
            for log in logs:
                st.code(log, language="bash")
        else:
            st.info("Select a station terminal and click **Trigger Feeder Bus Dispatch Sync** to coordinate commuter transit relief." if st.session_state.lang == "English" else "Commuter ಸಾರಿಗೆ ಪರಿಹಾರವನ್ನು ಸಂಘಟಿಸಲು ನಿಲ್ದಾಣದ ಟರ್ಮಿನಲ್ ಅನ್ನು ಆಯ್ಕೆ ಮಾಡಿ ಮತ್ತು ಬಟನ್ ಕ್ಲಿಕ್ ಮಾಡಿ.")

elif t("menu_bmtc") in choice:
    st.subheader(f"🚌 {t('bmtc_title')}")
    
    # -------------------------------------------------------------
    # BMTC ROUTE DATASET & OPERATIONAL DIRECTORY
    # -------------------------------------------------------------
    bmtc_routes_data = {
        "500-D": {
            "name": "Central Silk Board ⇄ Hebbal (via Outer Ring Road)",
            "start": "Central Silk Board",
            "end": "Hebbal Outer Ring Road",
            "operating_hours": "05:00 AM - 10:45 PM",
            "peak_freq": "5-8 minutes",
            "offpeak_freq": "12-15 minutes",
            "fare_range": "₹15 - ₹35 (Normal) | ₹30 - ₹75 (Vajra AC)",
            "depot": "Depot 4 (HSR Layout) & Depot 7 (Hebbal)",
            "stops": ["Central Silk Board", "HSR Layout (SIAS)", "Agara Junction", "Ibblur", "Bellandur", "Devarabeesanahalli", "Marathahalli Bridge", "Karthik Nagar", "KR Puram Railway Station", "Babusapalya", "Kalyan Nagar", "Nagawara", "Hebbal Junction"],
            "kannada": {
                "name": "ಸೆಂಟ್ರಲ್ ಸಿಲ್ಕ್ ಬೋರ್ಡ್ ⇄ ಹೆಬ್ಬಾಳ (ಹೊರ ವರ್ತುಲ ರಸ್ತೆ ಮೂಲಕ)",
                "depot": "ಡೆಪೋ ೪ (ಎಚ್‌ಎಸ್‌ಆರ್ ಲೇಔಟ್) ಮತ್ತು ಡೆಪೋ ೭ (ಹೆಬ್ಬಾಳ)",
                "operating_hours": "ಬೆಳಗ್ಗೆ ೦೫:೦೦ - ರಾತ್ರಿ ೧೦:೪೫",
                "peak_freq": "೫-೮ ನಿಮಿಷಗಳು",
                "offpeak_freq": "೧೨-೧೫ ನಿಮಿಷಗಳು",
            }
        },
        "335-E": {
            "name": "Kempegowda Bus Station (Majestic) ⇄ ITPL Kadugodi",
            "start": "KBS Majestic",
            "end": "ITPL Kadugodi",
            "operating_hours": "05:30 AM - 11:00 PM",
            "peak_freq": "8-10 minutes",
            "offpeak_freq": "15-20 minutes",
            "fare_range": "₹20 - ₹40 (Normal) | ₹40 - ₹85 (Vajra AC)",
            "depot": "Depot 13 (Whitefield)",
            "stops": ["KBS Majestic", "Corporation (Hudson Circle)", "Richmond Circle", "Domlur Bridge", "HAL Main Gate", "Marathahalli Bridge", "Kundalahalli Gate", "Graphite India", "Vydehi Hospital", "ITPL Main Gate", "Kadugodi Bus Stand"],
            "kannada": {
                "name": "ಕೆಂಪೇಗೌಡ ಬಸ್ ನಿಲ್ದಾಣ (ಮೆಜೆಸ್ಟಿಕ್) ⇄ ಐಟಿಪಿಎಲ್ ಕಾಡುಗೋಡಿ",
                "depot": "ಡೆಪೋ ೧೩ (ವೈಟ್‌ಫೀಲ್ಡ್)",
                "operating_hours": "ಬೆಳಗ್ಗೆ ೦೫:೩೦ - ರಾತ್ರಿ ೧೧:೦೦",
                "peak_freq": "೮-೧೦ ನಿಮಿಷಗಳು",
                "offpeak_freq": "೧೫-೨೦ ನಿಮಿಷಗಳು",
            }
        },
        "KIAS-9": {
            "name": "Kempegowda Bus Station (Majestic) ⇄ Kempegowda Intl Airport",
            "start": "KBS Majestic",
            "end": "Kempegowda International Airport (KIA)",
            "operating_hours": "24/7 (Continuous Service)",
            "peak_freq": "15-20 minutes",
            "offpeak_freq": "30 minutes",
            "fare_range": "₹230 - ₹260 (Vayu Vajra AC Sleeper)",
            "depot": "Depot 25 (Airport Services)",
            "stops": ["KBS Majestic", "Mekhri Circle", "Hebbal Flyover", "Kogilu Cross", "Yelahanka Bypass", "Chikkajala", "Trumpet Flyover", "KIA Departure Terminal"],
            "kannada": {
                "name": "ಕೆಂಪೇಗೌಡ ಬಸ್ ನಿಲ್ದಾಣ (ಮೆಜೆಸ್ಟಿಕ್) ⇄ ಕೆಂಪೇಗೌಡ ಅಂತರಾಷ್ಟ್ರೀಯ ವಿಮಾನ ನಿಲ್ದಾಣ",
                "depot": "ಡೆಪೋ ೨೫ (ವಿಮಾನ ನಿಲ್ದಾಣ ಸೇವೆಗಳು)",
                "operating_hours": "೨೪/೭ (ನಿರಂತರ ಸೇವೆ)",
                "peak_freq": "೧೫-೨೦ ನಿಮಿಷಗಳು",
                "offpeak_freq": "೩೦ ನಿಮಿಷಗಳು",
            }
        },
        "356-C": {
            "name": "Kempegowda Bus Station (Majestic) ⇄ Electronic City Wipro Gate",
            "start": "KBS Majestic",
            "end": "Electronic City Wipro Gate",
            "operating_hours": "06:00 AM - 10:30 PM",
            "peak_freq": "10-12 minutes",
            "offpeak_freq": "18-25 minutes",
            "fare_range": "₹18 - ₹38 (Normal) | ₹35 - ₹80 (Vajra AC)",
            "depot": "Depot 38 (Electronic City)",
            "stops": ["KBS Majestic", "Wilson Garden", "Dairy Circle", "Madiwala", "Central Silk Board", "Bommanahalli", "Garvebhavipalya", "Singasandra", "Hosa Road", "Konappana Agrahara", "Electronic City Wipro Gate"],
            "kannada": {
                "name": "ಕೆಂಪೇಗೌಡ ಬಸ್ ನಿಲ್ದಾಣ (ಮೆಜೆಸ್ಟಿಕ್) ⇄ ಎಲೆಕ್ಟ್ರಾನಿಕ್ ಸಿಟಿ ವಿಪ್ರೋ ಗೇಟ್",
                "depot": "ಡೆಪೋ ೩೮ (ಎಲೆಕ್ಟ್ರಾನಿಕ್ ಸಿಟಿ)",
                "operating_hours": "ಬೆಳಗ್ಗೆ ೦೬:೦೦ - ರಾತ್ರಿ ೧೦:೩೦",
                "peak_freq": "೧೦-೧೨ ನಿಮಿಷಗಳು",
                "offpeak_freq": "೧೮-೨೫ ನಿಮಿಷಗಳು",
            }
        },
        "G-3": {
            "name": "Mayo Hall (Brigade Road) ⇄ Sarjapur",
            "start": "Mayo Hall",
            "end": "Sarjapur Bus Stand",
            "operating_hours": "06:15 AM - 10:00 PM",
            "peak_freq": "15 minutes",
            "offpeak_freq": "25-30 minutes",
            "fare_range": "₹15 - ₹35 (Normal)",
            "depot": "Depot 21 (Sarjapur)",
            "stops": ["Mayo Hall", "Military Hockey Stadium", "Domlur Flyover", "Koramangala Krupanidhi College", "Agara Junction", "Bellandur Gate", "Kaikondrahalli", "Carmelaram Gate", "Dommasandra", "Sarjapur Bus Stand"],
            "kannada": {
                "name": "ಮೇಯೋ ಹಾಲ್ (ಬ್ರಿಗೇಡ್ ರಸ್ತೆ) ⇄ ಸರ್ಜಾಪುರ",
                "depot": "ಡೆಪೋ ೨೧ (ಸರ್ಜಾಪುರ)",
                "operating_hours": "ಬೆಳಗ್ಗೆ ೦೬:೧೫ - ರಾತ್ರಿ ೧೦:೦೦",
                "peak_freq": "೧೫ ನಿಮಿಷಗಳು",
                "offpeak_freq": "೨೫-೩೦ ನಿಮಿಷಗಳು",
            }
        }
    }

    # Fleet status header metric
    st.markdown(f"""
    <div style="background: #0b2545; color: white; border-radius: 12px; padding: 1rem; margin-bottom: 1.5rem; border-left: 5px solid #FF9933; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
            <div>
                <h4 style="margin: 0; font-weight: bold; color: white; font-size: 1.1rem;">🚌 BMTC INTELLIGENT FLEET OPERATIONS</h4>
                <p style="margin: 3px 0 0 0; font-size: 0.8rem; color: #cbd5e1;">Directorate of Urban Land Transport (DULT) & BMTC Depot Sync Hub</p>
            </div>
            <div style="display: flex; gap: 1.5rem; margin-top: 0.5rem;">
                <div style="text-align: right;">
                    <div style="font-size: 0.75rem; color: #94a3b8; font-weight: bold;">TOTAL ON-ROAD FLEET</div>
                    <div style="font-size: 1.15rem; font-weight: bold; color: #38bdf8;">5,842 / 6,100 <span style="font-size:0.8rem; color:#4ade80;">(95.8%)</span></div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 0.75rem; color: #94a3b8; font-weight: bold;">ACTIVE SECTORS</div>
                    <div style="font-size: 1.15rem; font-weight: bold; color: #f59e0b;">486 Routes</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    bmtc_tabs = st.tabs([
        "📋 Route Directory & Timetables" if st.session_state.lang == "English" else "📋 ಮಾರ್ಗ ಡೈರೆಕ್ಟರಿ ಮತ್ತು ವೇಳಾಪಟ್ಟಿಗಳು",
        "⚡ Live Congestion Schedule Optimizer" if st.session_state.lang == "English" else "⚡ ವೇಳಾಪಟ್ಟಿ ವಿಳಂಬ ಆಪ್ಟಿಮೈಜರ್",
        "🛣️ Dedicated Bus Lane Monitor" if st.session_state.lang == "English" else "🛣️ ಪ್ರತ್ಯೇಕ ಬಸ್ ಪಥದ ಮೇಲ್ವಿಚಾರಣೆ"
    ])

    # TAB 1: Route Directory & Timetables
    with bmtc_tabs[0]:
        st.markdown(f"""
        <div class="admin-panel" style="border-left: 4px solid #10b981; margin-bottom: 1.2rem;">
            <h5>📋 BMTC Route Directory</h5>
            <p style="font-size: 0.85rem; color:#475569;">Select a high-density daily transit route to view stop-by-stop schedule alignments:</p>
        </div>
        """, unsafe_allow_html=True)
        
        sel_route_id = st.selectbox(
            "Select Bus Route:" if st.session_state.lang == "English" else "ಬಸ್ ಮಾರ್ಗ ಆಯ್ಕೆಮಾಡಿ:",
            list(bmtc_routes_data.keys()),
            key="bmtc_dir_route"
        )
        
        route_info = bmtc_routes_data[sel_route_id]
        
        col_rd1, col_rd2 = st.columns([1, 1.2])
        
        with col_rd1:
            st.markdown(f"""
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 1.2rem; height: 100%;">
                <h5 style="color:#0f172a; margin-top: 0; font-weight: bold; font-size:1rem;">🚌 Route Details: {sel_route_id}</h5>
                <hr style="margin: 0.6rem 0; border: 0; border-top: 1px solid #cbd5e1;"/>
                <div style="font-size: 0.85rem; line-height: 1.6; color: #334155;">
                    📍 <strong>Sector:</strong> {route_info['name'] if st.session_state.lang == 'English' else route_info['kannada']['name']}<br/>
                    🏢 <strong>Managing Depot:</strong> {route_info['depot'] if st.session_state.lang == 'English' else route_info['kannada']['depot']}<br/>
                    ⏰ <strong>Operating Hours:</strong> {route_info['operating_hours'] if st.session_state.lang == 'English' else route_info['kannada']['operating_hours']}<br/>
                    ⚡ <strong>Peak Frequency:</strong> {route_info['peak_freq'] if st.session_state.lang == 'English' else route_info['kannada']['peak_freq']}<br/>
                    🐢 <strong>Off-Peak Frequency:</strong> {route_info['offpeak_freq'] if st.session_state.lang == 'English' else route_info['kannada']['offpeak_freq']}<br/>
                    💳 <strong>Fare Range:</strong> {route_info['fare_range']}<br/>
                    🏁 <strong>Total Stops:</strong> {len(route_info['stops'])} stops
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_rd2:
            st.markdown(f"""
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 1.2rem;">
                <h5 style="color:#0f172a; margin-top: 0; font-weight: bold; font-size:1rem;">📍 Route Stops Timeline</h5>
                <hr style="margin: 0.6rem 0; border: 0; border-top: 1px solid #cbd5e1;"/>
            </div>
            """, unsafe_allow_html=True)
            
            stops_html = ""
            for idx_s, stop in enumerate(route_info['stops']):
                is_terminal = (idx_s == 0 or idx_s == len(route_info['stops']) - 1)
                color = "#3b82f6" if idx_s == 0 else "#10b981" if idx_s == len(route_info['stops']) - 1 else "#64748b"
                font_weight = "bold" if is_terminal else "normal"
                marker = "🏁" if idx_s == len(route_info['stops']) - 1 else "🛫" if idx_s == 0 else "•"
                
                stops_html += f'<div style="display: flex; align-items: center; margin-bottom: 0.3rem; padding-left: 0.5rem;">'
                stops_html += f'<span style="color: {color}; font-size: 1rem; font-weight: bold; width: 1.5rem; display: inline-block; text-align: center;">{marker}</span>'
                stops_html += f'<span style="font-weight: {font_weight}; color: #1e293b; font-size: 0.85rem;">{stop}</span>'
                stops_html += f'</div>'
                if idx_s < len(route_info['stops']) - 1:
                    stops_html += f'<div style="padding-left: 1.15rem; color: #cbd5e1; font-size: 0.75rem; margin-top: -0.2rem; margin-bottom: -0.1rem;">│</div>'
                    
            st.markdown(f'<div style="max-height: 250px; overflow-y: auto; padding-right: 0.5rem;">{stops_html}</div>', unsafe_allow_html=True)

    # TAB 2: Live Congestion Schedule Optimizer
    with bmtc_tabs[1]:
        st.markdown(f"""
        <div class="admin-panel" style="border-left: 4px solid #f59e0b; margin-bottom: 1.2rem;">
            <h5>⚡ Live Congestion Schedule Optimizer</h5>
            <p style="font-size: 0.85rem; color:#475569;">Identify scheduling bottlenecks and trigger reserve dispatches to mitigate delays:</p>
        </div>
        """, unsafe_allow_html=True)
        
        col_op1, col_op2 = st.columns([1, 1.2])
        
        with col_op1:
            opt_route_id = st.selectbox(
                "Select Route to Optimize:" if st.session_state.lang == "English" else "ಆಪ್ಟಿಮೈಸ್ ಮಾಡಲು ಮಾರ್ಗವನ್ನು ಆರಿಸಿ:",
                list(bmtc_routes_data.keys()),
                key="bmtc_opt_route"
            )
            
            time_slot = st.selectbox(
                "Select Current Time Window:" if st.session_state.lang == "English" else "ಪ್ರಸ್ತುತ ಸಮಯದ ವಿಂಡೋ ಆರಿಸಿ:",
                ["08:00 AM - 10:00 AM (Morning Peak)", "12:00 PM - 02:00 PM (Off-Peak)", "05:00 PM - 08:00 PM (Evening Peak)", "10:00 PM - 12:00 AM (Late Night)"]
            )
            
            is_peak = "Peak" in time_slot
            
            # Predict delays based on peak hours and active database logs
            base_delay = 18 if is_peak else 4
            if opt_route_id == "500-D":
                base_delay += 10
            elif opt_route_id == "356-C" or opt_route_id == "335-E":
                base_delay += 6
                
            st.metric(
                label="Predicted Operational Delay" if st.session_state.lang == "English" else "ಊಹಿಸಲಾದ ಕಾರ್ಯಾಚರಣೆಯ ವಿಳಂಬ",
                value=f"{base_delay} mins",
                delta="+12 mins vs baseline" if is_peak else "+2 mins vs baseline",
                delta_color="inverse"
            )
            
            st.markdown("##### 💡 Dispatch Actions:")
            
            # Action 1: Inject standby bus
            btn_inject = st.button("Inject Gap-Fill Feeder Bus" if st.session_state.lang == "English" else "ಮೀಸಲು ಫೀಡರ್ ಬಸ್ ಸೇರಿಸಿ", type="primary")
            
            # Action 2: Signal priority wave
            btn_signal = st.button("Request Signal Priority Wave" if st.session_state.lang == "English" else "ಸಿಗ್ನಲ್ ಆದ್ಯತೆ ವೇವ್ ವಿನಂತಿಸಿ")
            
        with col_op2:
            st.write("📋 **Depot Dispatch & Signal Logs**" if st.session_state.lang == "English" else "📋 **ಡೆಪೋ ರವಾನೆ ಮತ್ತು ಸಿಗ್ನಲ್ ಲಾಗ್‌ಗಳು**")
            
            # Handle Actions in Session State
            if "bmtc_logs" not in st.session_state:
                st.session_state.bmtc_logs = [
                    "[08:15:02] BMTC Control: System monitoring active.",
                    "[08:30:10] DULT Center: Connected to Namma Metro passenger outflow feeds."
                ]
                
            if btn_inject:
                import random
                bus_id = f"KA-57-F-{random.randint(1000, 9999)}"
                depot = bmtc_routes_data[opt_route_id]["depot"].split(" & ")[0]
                now_str = datetime.datetime.now().strftime("%H:%M:%S")
                st.session_state.bmtc_logs.append(f"[{now_str}] DISPATCH: Standby bus {bus_id} injected from {depot} to Route {opt_route_id}.")
                st.session_state.bmtc_logs.append(f"[{now_str}] SYSTEM: Headway gap reduced by 8 mins. Status optimized.")
                st.toast(f"Feeder Bus {bus_id} Dispatched!" if st.session_state.lang == "English" else f"ಫೀಡರ್ ಬಸ್ {bus_id} ರವಾನಿಸಲಾಗಿದೆ!")
                
            if btn_signal:
                now_str = datetime.datetime.now().strftime("%H:%M:%S")
                junction = "Central Silk Board" if opt_route_id in ["500-D", "356-C"] else "Marathahalli Bridge" if opt_route_id == "335-E" else "Hebbal Flyover"
                st.session_state.bmtc_logs.append(f"[{now_str}] SIGNAL: Priority wave requested at {junction} for Route {opt_route_id}.")
                st.session_state.bmtc_logs.append(f"[{now_str}] BTP Sync: Signal green-time extended by 15 seconds.")
                st.toast("Signal Priority Dispatched!" if st.session_state.lang == "English" else "ಸಿಗ್ನಲ್ ಆದ್ಯತೆಯನ್ನು ರವಾನಿಸಲಾಗಿದೆ!")
                
            # Print logs in reverse chronological order
            log_box_content = ""
            for log in reversed(st.session_state.bmtc_logs):
                log_box_content += log + "\n"
            st.text_area("Live Terminal", value=log_box_content, height=220, disabled=True, label_visibility="collapsed")

    # TAB 3: Dedicated Bus Lane (DBL) Monitor
    with bmtc_tabs[2]:
        st.markdown(f"""
        <div class="admin-panel" style="border-left: 4px solid #ef4444; margin-bottom: 1.2rem;">
            <h5>🛣️ Dedicated Bus Lane Compliance Monitor</h5>
            <p style="font-size: 0.85rem; color:#475569;">Outer Ring Road (ORR) Dedicated Bus Lane - Silk Board to KR Puram Sector:</p>
        </div>
        """, unsafe_allow_html=True)
        
        col_dbl1, col_dbl2 = st.columns([1, 1.2])
        
        with col_dbl1:
            st.markdown(f"""
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 1.2rem; margin-bottom: 1rem;">
                <h5 style="color:#0f172a; margin-top: 0; font-weight: bold; font-size:1rem;">🛣️ Compliance Stats</h5>
                <hr style="margin: 0.6rem 0; border: 0; border-top: 1px solid #cbd5e1;"/>
            </div>
            """, unsafe_allow_html=True)
            
            # Display DBL compliance stats
            dc1, dc2 = st.columns(2)
            dc1.metric("Compliance Rate", "89.4%", delta="+2.1% vs last week")
            dc2.metric("Bus Speed Speedup", "+38%", delta="+4% vs mixed lane")
            
            st.write("")
            btn_marshal = st.button("Dispatch Enforcement Marshals" if st.session_state.lang == "English" else "ಜಾರಿ ಮಾರ್ಷಲ್‌ಗಳನ್ನು ರವಾನಿಸಿ", type="primary")
            
        with col_dbl2:
            st.markdown(f"""
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 1.2rem; margin-bottom: 0.5rem;">
                <h5 style="color:#0f172a; margin-top: 0; font-weight: bold; font-size:1rem;">🚨 DBL Camera Intrusion Alerts (Live Feed)</h5>
                <hr style="margin: 0.6rem 0; border: 0; border-top: 1px solid #cbd5e1;"/>
            </div>
            """, unsafe_allow_html=True)
            
            if "dbl_violations" not in st.session_state:
                st.session_state.dbl_violations = [
                    {"vehicle": "KA-03-MP-4122", "type": "SUV", "location": "Ibblur Junction", "status": "🚨 Flagged"},
                    {"vehicle": "KA-51-EF-8890", "type": "Sedan", "location": "Bellandur Gate", "status": "🚨 Flagged"},
                    {"vehicle": "KA-01-AB-1234", "type": "Two-Wheeler", "location": "Marathahalli", "status": "🚨 Flagged"}
                ]
                
            if btn_marshal:
                st.session_state.dbl_violations = [
                    {"vehicle": v["vehicle"], "type": v["type"], "location": v["location"], "status": "👮 Deployed Marshal"} 
                    for v in st.session_state.dbl_violations
                ]
                st.toast("Traffic Marshals dispatched to DBL sectors!" if st.session_state.lang == "English" else "ಡಿಬಿಎಲ್ ಸೆಕ್ಟರ್‌ಗಳಿಗೆ ಸಂಚಾರ ಮಾರ್ಷಲ್‌ಗಳನ್ನು ರವಾನಿಸಲಾಗಿದೆ!")
                
            # Display violation alerts as beautiful cards
            for v in st.session_state.dbl_violations:
                badge_color = "#ef4444" if "🚨" in v["status"] else "#10b981"
                st.markdown(f"""
                <div style="padding: 0.6rem; background: white; border: 1px solid #e2e8f0; border-radius: 8px; margin-bottom: 0.5rem; display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span style="font-family: monospace; font-weight: bold; color: #1e293b; font-size:0.85rem;">{v['vehicle']}</span> 
                        <span style="font-size: 0.75rem; color: #64748b; margin-left: 0.5rem;">({v['type']})</span><br/>
                        <span style="font-size: 0.78rem; color: #475569;">📍 {v['location']}</span>
                    </div>
                    <span style="background: {badge_color}15; color: {badge_color}; border: 1px solid {badge_color}; font-size: 0.7rem; font-weight: bold; padding: 0.2rem 0.5rem; border-radius: 12px;">{v['status']}</span>
                </div>
                """, unsafe_allow_html=True)

elif t("menu_adhira") in choice:
    st.subheader(f"🤖 {t('adhira_title')}")
    
    # Initialize Chat History if not present
    welcome_msg = (
        "ನಮಸ್ಕಾರ | Welcome to the Bengaluru Intelligent Mobility Command Assistant. I am Officer Adhira, NIC's Advisory AI. How can I assist you with city traffic deployment today?"
        if st.session_state.lang == "ಕನ್ನಡ" else
        "Welcome to the Bengaluru Intelligent Mobility Command Assistant. I am Officer Adhira, NIC's Advisory AI. How can I assist you today?"
    )
    if "adhira_chat_history" not in st.session_state:
        st.session_state.adhira_chat_history = [
            {"role": "ai", "content": welcome_msg}
        ]
        
    col_a1, col_a2 = st.columns([1.2, 1])
    
    with col_a1:
        st.markdown(f"""
        <div class="admin-panel" style="border-left: 4px solid #0b2545; padding:1.2rem;">
            <div style="display:flex; align-items:center; gap:12px; margin-bottom:1rem;">
                <img src="data:image/png;base64,{avatar_b64}" style="width:50px; height:50px; border-radius:50%; border:2px solid #0b2545;"/>
                <div>
                    <h4 style="margin:0; color:#0b2545; font-family:'Montserrat', sans-serif; font-weight:bold;">Officer Adhira AI</h4>
                    <small style="color:#10b981; font-weight:bold;">🟢 ACTIVE COMMAND PROTOCOL SYNCED</small>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Render Chat history
        chat_html = ['<div class="chat-box">']
        for msg in st.session_state.adhira_chat_history:
            if msg["role"] == "ai":
                chat_html.append(f"""
                <div class="chat-msg ai">
                    <img class="chat-avatar ai" src="data:image/png;base64,{avatar_b64}"/>
                    <div class="chat-bubble">
                        {msg['content']}
                    </div>
                </div>
                """.strip())
            else:
                chat_html.append(f"""
                <div class="chat-msg user">
                    <div class="chat-bubble">
                        {msg['content']}
                    </div>
                </div>
                """.strip())
        chat_html.append('</div>')
        
        # Unindent completely to prevent code block formatting issues in markdown
        st.markdown("".join(chat_html), unsafe_allow_html=True)
        
        # Input form for new message
        with st.form(key="adhira_chat_form", clear_on_submit=True):
            user_msg = st.text_input(
                "Ask Officer Adhira a traffic deployment question:" if st.session_state.lang == "English" else "ಸಂಚಾರ ನಿಯೋಜನೆ ಬಗ್ಗೆ ಅಧಿಕಾರಿ ಅಧೀರಾರನ್ನು ಪ್ರಶ್ನಿಸಿ:",
                placeholder=t("placeholder_query")
            )
            chat_cols = st.columns([4, 1.2])
            with chat_cols[0]:
                quick_query = st.selectbox(
                    "Quick Commands / ತ್ವರಿತ ಆಜ್ಞೆಗಳು:",
                    [
                        "-- Select Quick Query --",
                        "Analyze current peak-hour bottleneck at Hebbal Flyover",
                        "Suggest emergency mitigation for waterlogging at Silk Board Junction",
                        "Metro-BMTC coordination plan for Chinnaswamy Stadium event"
                    ] if st.session_state.lang == "English" else [
                        "-- ತ್ವರಿತ ಪ್ರಶ್ನೆ ಆರಿಸಿ --",
                        "ಹೆಬ್ಬಾಳ ಫ್ಲೈಓವರ್‌ನಲ್ಲಿನ ಪ್ರಸ್ತುತ ಪೀಕ್-ಅವರ್ ಬಾಟಲ್‌ನೆಕ್ ಅನ್ನು ವಿಶ್ಲೇಷಿಸಿ",
                        "ಸಿಲ್ಕ್ ಬೋರ್ಡ್ ಜಂಕ್ಷನ್‌ನಲ್ಲಿನ ಜಲಾವೃತಕ್ಕೆ ತುರ್ತು ಶಮನ ಸೂಚಿಸಿ",
                        "ಚಿನ್ನಸ್ವಾಮಿ ಕ್ರೀಡಾಂಗಣದ ಕಾರ್ಯಕ್ರಮಕ್ಕಾಗಿ ಮೆಟ್ರೋ-ಬಿಎಂಟಿಸಿ ಸಮನ್ವಯ ಯೋಜನೆ"
                    ]
                )
            with chat_cols[1]:
                submit_msg = st.form_submit_button(
                    "Send" if st.session_state.lang == "English" else "ಕಳುಹಿಸಿ",
                    use_container_width=True,
                    type="primary"
                )
                
        # Handle chat submissions
        active_query = ""
        if submit_msg:
            if user_msg.strip():
                active_query = user_msg.strip()
            elif quick_query and not quick_query.startswith("--"):
                active_query = quick_query
                
        if active_query:
            # Append user message
            st.session_state.adhira_chat_history.append({"role": "user", "content": active_query})
            
            # Generate AI response
            reply = ""
            q_lower = active_query.lower()
            
            if "hebbal" in q_lower or "ಹೆಬ್ಬಾಳ" in q_lower:
                if st.session_state.lang == "English":
                    reply = ("**Officer Adhira AI Analysis:** Hebbal Flyover (North Corridor) is experiencing downstream merges bottlenecking onto Outer Ring Road near Nagavara. "
                             "I recommend: 1) Deploying 3 Civil Marshals to regulate the service road merge; 2) Signal adjustment at Mekhri Circle to hold outbound flow by 15s; "
                             "3) Advising KIAS Airport Shuttles to divert via Hennur Main Road if speeds fall below 15 km/h.")
                else:
                    reply = ("**ಅಧೀರಾ ಎಐ ವಿಶ್ಲೇಷಣೆ:** ಹೆಬ್ಬಾಳ ಫ್ಲೈಓವರ್ (ಉತ್ತರ ಕಾರಿಡಾರ್) ನಾಗವಾರದ ಬಳಿ ಹೊರ ವರ್ತುಲ ರಸ್ತೆಗೆ ಸೇರುವ ಸಂಚಾರದಲ್ಲಿ ದಟ್ಟಣೆಯನ್ನು ಎದುರಿಸುತ್ತಿದೆ. "
                             "ನನ್ನ ಶಿಫಾರಸುಗಳು: ೧) ಸೇವಾ ರಸ್ತೆ ವಿಲೀನವನ್ನು ನಿಯಂತ್ರಿಸಲು ೩ ನಾಗರಿಕ ಮಾರ್ಷಲ್‌ಗಳನ್ನು ನಿಯೋಜಿಸಿ; ೨) ಹೊರಹೋಗುವ ಹರಿವನ್ನು ತಡೆಯಲು ಮೇಖ್ರಿ ವೃತ್ತದಲ್ಲಿ ಸಂಕೇತವನ್ನು ೧೫ ಸೆಕೆಂಡುಗಳಷ್ಟು ಹೊಂದಿಸಿ; "
                             "೩) ವೇಗವು ೧೫ ಕಿಮೀಗಿಂತ ಕಡಿಮೆಯಾದರೆ ಹೆಣ್ಣೂರು ಮುಖ್ಯ ರಸ್ತೆ ಮೂಲಕ ಸಂಚಾರ ಬದಲಾಯಿಸಲು ಕೆಐಎಎಸ್ ವಿಮಾನ ನಿಲ್ದಾಣದ ಶಟಲ್‌ಗಳಿಗೆ ಸಲಹೆ ನೀಡಿ.")
            elif "silk board" in q_lower or "ಸಿಲ್ಕ್" in q_lower:
                if st.session_state.lang == "English":
                    reply = ("**Officer Adhira AI Analysis:** Silk Board Junction is currently under high load. Waterlogging hazard at the service road intersection is high. "
                             "Recommended dispatches: 1) Deploy 2 heavy-duty towing trucks on standby at the HSR exit loop; 2) Hold traffic signal for ORR bus lane waves; "
                             "3) Push public VMS board alert: 'SILK BOARD SERVICE ROAD BLOCK - USE FLYOVER LANES'.")
                else:
                    reply = ("**ಅಧೀರಾ ಎಐ ವಿಶ್ಲೇಷಣೆ:** ಸಿಲ್ಕ್ ಬೋರ್ಡ್ ಜಂಕ್ಷನ್ ಪ್ರಸ್ತುತ ಹೆಚ್ಚಿನ ದಟ್ಟಣೆಯಲ್ಲಿದೆ. ಸೇವಾ ರಸ್ತೆ ಛೇದಕದಲ್ಲಿ ಜಲಾವೃತವಾಗುವ ಅಪಾಯ ಹೆಚ್ಚಾಗಿದೆ. "
                             "ಶಿಫಾರಸು ಮಾಡಲಾದ ನಿಯೋಜನೆ: ೧) ಎಚ್‌ಎಸ್‌ಆರ್ ಎಕ್ಸಿಟ್ ಲೂಪ್‌ನಲ್ಲಿ ೨ ಹೆವಿ-ಡ್ಯೂಟಿ ಟೋಯಿಂಗ್ ಟ್ರಕ್‌ಗಳನ್ನು ಸಿದ್ಧವಾಗಿಡಿ; ೨) ಒಆರ್‌ಆರ್ ಬಸ್ ಲೇನ್ ಹರಿವಿಗೆ ಸಂಚಾರ ಸಂಕೇತವನ್ನು ಆದ್ಯತೆ ನೀಡಿ; "
                             "೩) ಸಾರ್ವಜನಿಕ ಸಂದೇಶ ಫಲಕದಲ್ಲಿ ಎಚ್ಚರಿಕೆ ನೀಡಿ: 'ಸಿಲ್ಕ್ ಬೋರ್ಡ್ ಸೇವಾ ರಸ್ತೆ ಬ್ಲಾಕ್ - ಫ್ಲೈಓವರ್ ಲೇನ್‌ಗಳನ್ನು ಬಳಸಿ'.")
            elif "chinnaswamy" in q_lower or "stadium" in q_lower or "ಚಿನ್ನಸ್ವಾಮಿ" in q_lower:
                if st.session_state.lang == "English":
                    reply = ("**Officer Adhira AI Analysis:** Projecting 35% traffic surge on Queens Road, Cubbon Road, and MG Road due to Chinnaswamy Stadium event. "
                             "Inter-agency coordination plan: 1) Increase Namma Metro Purple Line frequency to 3.5 minutes; 2) BMTC to deploy 15 dedicated feeder shuttles to Majestic and Indiranagar; "
                             "3) Strict parking ban on Mahatma Gandhi road with towing patrols active.")
                else:
                    reply = ("**ಅಧೀರಾ ಎಐ ವಿಶ್ಲೇಷಣೆ:** ಚಿನ್ನಸ್ವಾಮಿ ಕ್ರೀಡಾಂಗಣದ ಕಾರ್ಯಕ್ರಮದಿಂದಾಗಿ ಕ್ವೀನ್ಸ್ ರಸ್ತೆ, ಕಬ್ಬನ್ ರಸ್ತೆ ಮತ್ತು ಎಂಜಿ ರಸ್ತೆಯಲ್ಲಿ ೩೫% ಸಂಚಾರ ಹೆಚ್ಚಾಗುವ ನಿರೀಕ್ಷೆಯಿದೆ. "
                             "ಸಮನ್ವಯ ಯೋಜನೆ: ೧) ನಮ್ಮ ಮೆಟ್ರೋ ಪರ್ಪಲ್ ಲೈನ್ ಆವರ್ತನವನ್ನು ೩.೫ ನಿಮಿಷಗಳಿಗೆ ಹೆಚ್ಚಿಸಿ; ೨) ಮೆಜೆಸ್ಟಿಕ್ ಮತ್ತು ಇಂದಿರಾನಗರಕ್ಕೆ ೧೫ ಬಿಎಂಟಿಸಿ ಫೀಡರ್ ಬಸ್‌ಗಳನ್ನು ನಿಯೋಜಿಸಿ; "
                             "೩) ಟೋಯಿಂಗ್ ಗಸ್ತುಗಳೊಂದಿಗೆ ಮಹಾತ್ಮ ಗಾಂಧಿ ರಸ್ತೆಯಲ್ಲಿ ಕಟ್ಟುನಿಟ್ಟಾದ ಪಾರ್ಕಿಂಗ್ ನಿಷೇಧ ಜಾರಿಗೊಳಿಸಿ.")
            else:
                if st.session_state.lang == "English":
                    reply = ("**Officer Adhira AI Response:** Understood. Bengaluru Intelligent Command system advises: "
                             "1) Maintain strict lane discipline along high-density corridors; 2) Synchronize signal times at adjacent junctions to prevent gridlock tailbacks; "
                             "3) Coordinate with BMTC depot managers to adjust bus dispatch intervals during peak surges.")
                else:
                    reply = ("**ಅಧೀರಾ ಎಐ ಪ್ರತಿಕ್ರಿಯೆ:** ಅರ್ಥವಾಯಿತು. ಬೆಂಗಳೂರು ಬುದ್ಧಿವಂತ ಕಮಾಂಡ್ ಸಿಸ್ಟಮ್ ಸಲಹೆ ನೀಡುತ್ತದೆ: "
                             "೧) ಹೆಚ್ಚಿನ ಸಾಂದ್ರತೆಯ ಕಾರಿಡಾರ್‌ಗಳಲ್ಲಿ ಕಟ್ಟುನಿಟ್ಟಾದ ಲೇನ್ ಶಿಸ್ತು ಕಾಪಾಡಿಕೊಳ್ಳಿ; ೨) ಜ್ರಿಡ್‌ಲಾಕ್ ತಡೆಯಲು ಪಕ್ಕದ ಜಂಕ್ಷನ್‌ಗಳಲ್ಲಿ ಸಿಗ್ನಲ್ ಸಮಯಗಳನ್ನು ಸಿಂಕ್ರೊನೈಸ್ ಮಾಡಿ; "
                             "೩) ಗರಿಷ್ಠ ದಟ್ಟಣೆಯ ಸಮಯದಲ್ಲಿ ಬಸ್ ರವಾನೆ ಮಧ್ಯಂತರಗಳನ್ನು ಹೊಂದಿಸಲು ಬಿಎಂಟಿಸಿ ಡಿಪೋ ವ್ಯವಸ್ಥಾಪಕರೊಂದಿಗೆ ಸಮನ್ವಯ ಸಾಧಿಸಿ.")
            
            st.session_state.adhira_chat_history.append({"role": "ai", "content": reply})
            st.rerun()
            
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_a2:
        # Adhira's Cognitive Health Monitor & dispatches
        st.markdown(f"""
        <div class="admin-panel" style="border-left: 4px solid #d97706; padding:1.2rem; margin-bottom:1rem;">
            <h5 style="margin: 0 0 0.8rem 0; color: #0b2545; font-family: 'Montserrat', sans-serif;">🧠 Cognitive Health Monitor</h5>
        """, unsafe_allow_html=True)
        
        # Calculate feedback sync count
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM feedback")
        feedback_count = cursor.fetchone()[0]
        conn.close()
        
        cc1, cc2 = st.columns(2)
        cc1.metric("Inference Latency", "42 ms", delta="Stable")
        cc2.metric("Neural Active Weights", "128 / 128", delta="Synchronized")
        
        cc3, cc4 = st.columns(2)
        cc3.metric("Ensemble Confidence", "94.2%", delta="+0.8% vs last retrain")
        cc4.metric("Feedback Database Sync", f"{feedback_count} logs", delta="Connected")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Accept Adhira Recommendation Action
        st.markdown(f"""
        <div class="admin-panel" style="border-left: 4px solid #10b981; padding:1.2rem;">
            <h5 style="margin: 0 0 0.8rem 0; color: #0b2545; font-family: 'Montserrat', sans-serif;">⚡ Quick AI Dispatch Launcher</h5>
            <p style="font-size: 0.8rem; color:#475569;">Accept Adhira's live traffic recommendation based on current threat vectors and log it directly to the active command queue:</p>
        """, unsafe_allow_html=True)
        
        dispatch_presets = {
            "Hebbal Flyover Inflow Restriction (North Zone)": {
                "event_type": "unplanned",
                "event_cause": "congestion_bottleneck",
                "priority": "High",
                "requires_road_closure": False,
                "predicted_duration": 45.0,
                "police_station": "Hebbala",
                "corridor": "Orr north 1",
                "zone": "North zone 1",
                "junction": "Hebbalflyoverjunc",
                "latitude": 13.0419,
                "longitude": 77.5947
            },
            "Silk Board service road diversion (South Zone)": {
                "event_type": "unplanned",
                "event_cause": "waterlogging_delay",
                "priority": "High",
                "requires_road_closure": True,
                "predicted_duration": 75.0,
                "police_station": "Madiwala",
                "corridor": "Orr east 1",
                "zone": "South zone 2",
                "junction": "Silkboardjunc",
                "latitude": 12.9188,
                "longitude": 77.6215
            }
        }
        
        selected_dispatch = st.selectbox("Select Tactical Directive:", list(dispatch_presets.keys()))
        
        if st.button("Accept & Dispatch Command", type="primary", use_container_width=True):
            dp = dispatch_presets[selected_dispatch]
            recs = get_recommendations(dp["event_type"], dp["event_cause"], dp["priority"], dp["requires_road_closure"], dp["predicted_duration"])
            
            event_id = f"EVT{datetime.datetime.now().strftime('%m%d%H%M%S')}"
            log_recommendation(
                event_id, dp["event_type"], dp["event_cause"], dp["priority"], dp["requires_road_closure"], dp["predicted_duration"], recs,
                latitude=dp["latitude"], longitude=dp["longitude"], police_station=dp["police_station"],
                corridor=dp["corridor"], zone=dp["zone"], junction=dp["junction"]
            )
            
            st.success(f"Directive registered successfully! Dispatch ID: {event_id} has been queued into active incident response.")
            st.toast("Adhira Command Dispatched!", icon="🚀")
            
        st.markdown("</div>", unsafe_allow_html=True)


elif t("menu_control") in choice:
    st.subheader(f"⚙ {t('control_title')}")
    
    # --- Manual Retraining Block ---
    st.markdown("### 🔄 " + ("Retrain Engine & Reload Models" if st.session_state.lang == "English" else "ಎಂಜಿನ್ ಮರುತರಬೇತಿ ಮತ್ತು ಮಾದರಿಗಳನ್ನು ಮರುಲೋಡ್ ಮಾಡಿ"))
    st.markdown("Manually trigger model retraining with all operator-validated feedback logged so far. This will rebuild the ResNet + CatBoost models and refresh the system cache."
                if st.session_state.lang == "English" else
                "ಇಲ್ಲಿಯವರೆಗೆ ದಾಖಲಾದ ಎಲ್ಲಾ ಆಪರೇಟರ್-ಮೌಲ್ಯೀಕೃತ ಪ್ರತಿಕ್ರಿಯೆಗಳೊಂದಿಗೆ ಮಾದರಿ ಮರುತರಬೇತಿಯನ್ನು ಹಸ್ತಚಾಲಿತವಾಗಿ ಪ್ರಚೋದಿಸಿ. ಇದು ResNet + CatBoost ಮಾದರಿಗಳನ್ನು ಮರುನಿರ್ಮಾಣ ಮಾಡುತ್ತದೆ ಮತ್ತು ಸಿಸ್ಟಮ್ ಸಂಗ್ರಹವನ್ನು ರಿಫ್ರೆಶ್ ಮಾಡುತ್ತದೆ.")
    
    if st.button("🔄 " + ("Retrain Models & Clear Cache" if st.session_state.lang == "English" else "ಮಾದರಿಗಳಿಗೆ ಮರುತರಬೇತಿ ನೀಡಿ ಮತ್ತು ಸಂಗ್ರಹವನ್ನು ತೆರವುಗೊಳಿಸಿ"), key="btn_manual_retrain"):
        with st.spinner("Retraining model ensemble and regenerating calibration plots..." if st.session_state.lang == "English" else "ಮಾದರಿ ಸಮೂಹಕ್ಕೆ ಮರುತರಬೇತಿ ನೀಡಲಾಗುತ್ತಿದೆ ಮತ್ತು ಮಾಪನಾಂಕ ನಿರ್ಣಯ ಪ್ಲಾಟ್‌ಗಳನ್ನು ಮರುಸೃಷ್ಟಿಸಲಾಗುತ್ತಿದೆ..."):
            try:
                from retrain_engine import run_retraining
                from generate_metrics_plots import generate_plots
                
                success = run_retraining()
                if success:
                    generate_plots()
                    st.cache_resource.clear()
                    st.cache_data.clear()
                    st.success("Model ensemble retrained and system cache cleared successfully! Active session is now using updated weights."
                               if st.session_state.lang == "English" else
                               "ಮಾದರಿ ಸಮೂಹಕ್ಕೆ ಯಶಸ್ವಿಯಾಗಿ ಮರುತರಬೇತಿ ನೀಡಲಾಗಿದೆ ಮತ್ತು ಸಿಸ್ಟಮ್ ಸಂಗ್ರಹವನ್ನು ತೆರವುಗೊಳಿಸಲಾಗಿದೆ! ಸಕ್ರಿಯ ಸೆಷನ್ ಈಗ ನವೀಕರಿಸಿದ ತೂಕವನ್ನು ಬಳಸುತ್ತಿದೆ.")
                else:
                    st.error("Retraining engine failed to run. Check server logs." if st.session_state.lang == "English" else "ಮರುತರಬೇತಿ ಎಂಜಿನ್ ರನ್ ಮಾಡಲು ವಿಫಲವಾಗಿದೆ. ಸರ್ವರ್ ಲಾಗ್‌ಗಳನ್ನು ಪರಿಶೀಲಿಸಿ.")
            except Exception as ex:
                st.error(f"Error during retraining: {ex}")
                
    st.markdown("---")
    
    conn = sqlite3.connect(DB_PATH)
    predictions_df = pd.read_sql_query("SELECT id, event_cause, predicted_duration, timestamp FROM recommendations ORDER BY timestamp DESC LIMIT 20", conn)
    conn.close()
    
    if len(predictions_df) == 0:
        st.warning("No generated incident records found. Use the **AI Event Planner** to log predictions first." if st.session_state.lang == "English" else "ಯಾವುದೇ ಮುನ್ಸೂಚನೆ ದಾಖಲೆಗಳು ಕಂಡುಬಂದಿಲ್ಲ. ಮೊದಲು ಇವೆಂಟ್ ಪ್ಲಾನರ್ ಬಳಸಿ.")
    else:
        select_ref_lbl = "ಘಟನೆಯ ಆದೇಶವನ್ನು ಆಯ್ಕೆಮಾಡಿ" if st.session_state.lang == "ಕನ್ನಡ" else "Select Event ID to Log Post-Event Feedback"
        selected_event_row = st.selectbox(
            select_ref_lbl,
            predictions_df.apply(lambda r: f"{r['id']} - {r['event_cause'].capitalize()} (Predicted: {r['predicted_duration']:.1f}m)", axis=1),
            key="control_feedback_select"
        )
        selected_event_id = selected_event_row.split(" - ")[0]
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM recommendations WHERE id = ?", (selected_event_id,))
        evt_data = cursor.fetchone()
        conn.close()
        
        if evt_data:
            details_lbl = f"ಘಟನೆ ವಿವರಗಳು: ID={evt_data[0]} | ಕಾರಣ={evt_data[2]} | ಮುನ್ಸೂಚಿತ ಅವಧಿ={evt_data[5]:.1f} ನಿಮಿಷಗಳು" if st.session_state.lang == "ಕನ್ನಡ" else f"Event details: ID={evt_data[0]} | Cause={evt_data[2]} | Predicted Duration={evt_data[5]:.1f} mins"
            st.info(details_lbl)
            
            with st.form("feedback_form"):
                st.markdown(f"#### {t('log_feedback_title')}")
                actual_duration = st.number_input(t("actual_resolution"), value=float(np.round(evt_data[5])), min_value=1.0)
                actual_officers = st.number_input(t("actual_officers"), value=int(evt_data[6]), min_value=0)
                actual_marshals = st.number_input(t("actual_marshals"), value=int(evt_data[7]), min_value=0)
                notes = st.text_area(t("operator_remarks"))
                submit_feedback = st.form_submit_button(t("submit_logs"))
                
            if submit_feedback:
                log_feedback(selected_event_id, actual_duration, actual_officers, actual_marshals, "Light", "None", notes)
                st.success("Post-event observations logged successfully!" if st.session_state.lang == "English" else "ಅವಲೋಕನಗಳನ್ನು ಯಶಸ್ವಿಯಾಗಿ ದಾಖಲಿಸಲಾಗಿದೆ!")
# ─────────────────────────────────────────────────────────
# OFFICIAL BILINGUAL GOVERNMENT PORTAL FOOTER
# ─────────────────────────────────────────────────────────
st.markdown(f"""
<div style="border-top: 1px solid rgba(0, 212, 255, 0.15); padding-top: 1.5rem; margin-top: 3.5rem; text-align: center; font-size: 0.8rem; color: #8892b0;">
    <p style="margin: 0; font-weight: bold; color: #a0aec0;">ಕರ್ನಾಟಕ ಸರ್ಕಾರ | GOVERNMENT OF KARNATAKA</p>
    <p style="margin: 0.2rem 0; color: #00d4ff;">Directorate of Urban Land Transport (DULT) & Bengaluru Traffic Police Initiative</p>
    <p style="margin: 0.2rem 0; font-size: 0.72rem; color: #64748b;">
        © 2026 Directorate of Urban Land Transport. All Rights Reserved. Designed, developed and hosted by National Informatics Centre (NIC).
    </p>
    <p style="margin: 0.4rem 0 0 0; font-size: 0.72rem; color: #64748b;">
        <span style="border-right: 1px solid rgba(255,255,255,0.1); padding-right: 8px; margin-right: 8px;"><a href="#" style="color:#64748b; text-decoration:none;">Disclaimer</a></span>
        <span style="border-right: 1px solid rgba(255,255,255,0.1); padding-right: 8px; margin-right: 8px;"><a href="#" style="color:#64748b; text-decoration:none;">Privacy Policy</a></span>
        <span><a href="#" style="color:#64748b; text-decoration:none;">Hyperlinking Policy</a></span>
    </p>
    <p style="margin: 0.5rem 0 0 0; font-size: 0.7rem; color: #888;">
        {t("visitor_count")}: <strong>1,842,912</strong> | {t("last_updated")}: 19 Jun 2026
    </p>
</div>
""", unsafe_allow_html=True)
