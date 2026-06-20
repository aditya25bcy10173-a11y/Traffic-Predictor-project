import sqlite3
import os
import numpy as np

# Database path for logging and feedback
DB_PATH = 'feedback.db'

def init_db():
    """Initializes the feedback database to log recommendations and operator inputs."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Check if table recommendations has latitude column (migration check)
    try:
        cursor.execute("SELECT latitude FROM recommendations LIMIT 1")
        has_latitude = True
    except sqlite3.OperationalError:
        has_latitude = False
        
    if not has_latitude:
        # Drop recommendations and feedback tables to migrate the schema
        cursor.execute("DROP TABLE IF EXISTS feedback")
        cursor.execute("DROP TABLE IF EXISTS recommendations")
        print("Dropped old tables to perform migration for spatial/administrative feature logging.")
        
    # Table to store generated recommendations with input features
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS recommendations (
            id TEXT PRIMARY KEY,
            event_type TEXT,
            event_cause TEXT,
            priority TEXT,
            requires_road_closure BOOLEAN,
            predicted_duration REAL,
            recommended_officers INTEGER,
            recommended_marshals INTEGER,
            recommended_barricading TEXT,
            recommended_diversion TEXT,
            recommended_towing INTEGER,
            latitude REAL,
            longitude REAL,
            police_station TEXT,
            corridor TEXT,
            zone TEXT,
            junction TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Table to store actual post-event operator feedback
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            event_id TEXT PRIMARY KEY,
            actual_duration REAL,
            actual_officers INTEGER,
            actual_marshals INTEGER,
            actual_barricading TEXT,
            actual_diversion TEXT,
            overridden BOOLEAN,
            operator_notes TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(event_id) REFERENCES recommendations(id)
        )
    ''')
    
    conn.commit()
    conn.close()

def get_recommendations(event_type, event_cause, priority, requires_road_closure, predicted_duration_minutes, veh_type=None):
    """
    Expert rule-based logic mapping event metadata and predicted duration to:
    - Officers
    - Marshals
    - Barricading (None, Light, Heavy, Critical)
    - Diversion (None, Minor, Major)
    - Towing Trucks
    """
    event_type = str(event_type).lower()
    event_cause = str(event_cause).lower()
    priority = str(priority).lower()
    requires_road_closure = bool(requires_road_closure)
    
    # 1. Base Manpower
    if event_type == 'planned':
        # Planned events (rallies, sports, festivals) require high baseline crowd control
        base_officers = 6
        base_marshals = 10
    else:
        # Unplanned events (breakdowns, accidents, water logging)
        base_officers = 4 if priority == 'high' else 2
        base_marshals = 6 if priority == 'high' else 3
        
    # Scale based on predicted duration severity
    duration_multiplier = 1.0
    if predicted_duration_minutes > 240: # > 4 hours
        duration_multiplier = 2.0
    elif predicted_duration_minutes > 120: # > 2 hours
        duration_multiplier = 1.5
        
    officers = int(np.ceil(base_officers * duration_multiplier))
    marshals = int(np.ceil(base_marshals * duration_multiplier))
    
    # Additional manpower for major causes
    if event_cause in ['water_logging', 'protest', 'public_event']:
        officers += 2
        marshals += 4
        
    # 2. Barricading
    if requires_road_closure:
        barricading = "Critical (15+ barriers with warning signs)"
    elif event_cause in ['construction', 'water_logging', 'tree_fall']:
        if predicted_duration_minutes > 90:
            barricading = "Heavy (5-15 barriers to block multiple lanes)"
        else:
            barricading = "Light (1-5 barriers to block single lane)"
    elif priority == 'high':
        barricading = "Light (1-5 barriers for lane organization)"
    else:
        barricading = "None"
        
    # 3. Diversion
    if requires_road_closure:
        diversion = "Major (Divert all traffic at preceding intersections)"
    elif predicted_duration_minutes > 180 and priority == 'high':
        diversion = "Major (Divert heavy vehicles and buses)"
    elif predicted_duration_minutes > 60 and priority == 'high':
        diversion = "Minor (Local lane-level diversion, merge lanes early)"
    else:
        diversion = "None"
        
    # 4. Towing Trucks
    towing = 0
    if event_cause in ['vehicle_breakdown', 'accident']:
        if veh_type in ['heavy_vehicle', 'private_bus', 'bmtc_bus', 'bus', 'truck']:
            towing = 2 # Requires heavy duty towing crane
        else:
            towing = 1 # Standard towing vehicle
            
    # Placement details helper text
    placements = []
    if barricading != "None":
        placements.append("Set up barriers 50-100 meters upstream of the incident.")
    if diversion == "Minor":
        placements.append("Merge traffic into the free lanes using cones.")
    elif diversion == "Major":
        placements.append("Deploy officers at nearest intersections to divert inbound vehicles.")
    if towing > 0:
        placements.append("Request towing vehicle dispatch immediately to clear the block.")
        
    action_plan = " ".join(placements) if placements else "Monitor traffic flow closely."

    return {
        'officers': officers,
        'marshals': marshals,
        'barricading': barricading,
        'diversion': diversion,
        'towing': towing,
        'action_plan': action_plan
    }

def log_recommendation(event_id, event_type, event_cause, priority, requires_road_closure, predicted_duration, recs,
                       latitude=0.0, longitude=0.0, police_station="unknown", corridor="unknown", zone="unknown", junction="unknown"):
    """Saves a recommendation record to the database with spatial and administrative features."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT OR REPLACE INTO recommendations 
            (id, event_type, event_cause, priority, requires_road_closure, predicted_duration, 
             recommended_officers, recommended_marshals, recommended_barricading, recommended_diversion, recommended_towing,
             latitude, longitude, police_station, corridor, zone, junction)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            event_id, event_type, event_cause, priority, requires_road_closure, predicted_duration,
            recs['officers'], recs['marshals'], recs['barricading'], recs['diversion'], recs['towing'],
            latitude, longitude, police_station, corridor, zone, junction
        ))
        conn.commit()
    except Exception as e:
        print(f"Error logging recommendation: {e}")
    finally:
        conn.close()

def log_feedback(event_id, actual_duration, actual_officers, actual_marshals, actual_barricading, actual_diversion, notes=""):
    """Saves operator feedback and checks if model retraining is needed."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Check if recommendation exists to determine if it was overridden
    cursor.execute('SELECT recommended_officers, recommended_marshals FROM recommendations WHERE id = ?', (event_id,))
    rec = cursor.fetchone()
    
    overridden = False
    if rec:
        rec_off, rec_mar = rec
        if rec_off != actual_officers or rec_mar != actual_marshals:
            overridden = True
            
    try:
        cursor.execute('''
            INSERT OR REPLACE INTO feedback 
            (event_id, actual_duration, actual_officers, actual_marshals, actual_barricading, actual_diversion, overridden, operator_notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (event_id, actual_duration, actual_officers, actual_marshals, actual_barricading, actual_diversion, overridden, notes))
        conn.commit()
        print(f"Logged post-event feedback for event {event_id}. Overridden={overridden}.")
    except Exception as e:
        print(f"Error logging feedback: {e}")
    finally:
        conn.close()
        
    # Check count of feedback samples to trigger retraining (simulation)
    check_feedback_count_for_retraining()

def check_feedback_count_for_retraining():
    """Counts feedback records and triggers model retraining if limit reached."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM feedback')
    count = cursor.fetchone()[0]
    conn.close()
    
    print(f"Feedback database contains {count} submissions.")
    if count > 0 and count % 5 == 0:
        print(">>> Retraining Threshold Reached! Triggering post-event model retraining loop...")
        try:
            from retrain_engine import run_retraining
            from generate_metrics_plots import generate_plots
            success = run_retraining()
            if success:
                print("Retraining completed successfully from auto-trigger.")
                generate_plots()
                print("Validation plots regenerated after auto-retraining.")
            return True
        except Exception as e:
            print(f"Error in auto-retraining trigger: {e}")
    return False

# Initialize database on module import
init_db()
