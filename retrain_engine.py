import sqlite3
import pandas as pd
import numpy as np
import os
import datetime
from recommender import DB_PATH
from model import train_hybrid_ensemble

def run_retraining():
    print("Starting closed-loop ML retraining engine...")
    conn = sqlite3.connect(DB_PATH)
    
    # Query joined feedback and recommendations
    query = """
        SELECT r.event_type, r.event_cause, r.requires_road_closure, r.priority, 
               r.police_station, r.corridor, r.zone, r.junction, r.latitude, r.longitude, 
               r.timestamp, f.actual_duration
        FROM feedback f
        JOIN recommendations r ON f.event_id = r.id
    """
    try:
        feedback_df = pd.read_sql_query(query, conn)
    except Exception as e:
        print(f"Error reading database: {e}")
        feedback_df = pd.DataFrame()
    finally:
        conn.close()
        
    base_file = r"D:\flip.csv"
    if not os.path.exists(base_file):
        print(f"Base training file {base_file} not found. Cannot proceed with retraining.")
        return False
        
    print(f"Baseline training records: {pd.read_csv(base_file).shape[0]}")
    print(f"New operator feedback records: {len(feedback_df)}")
    
    if len(feedback_df) > 0:
        # Convert feedback rows into baseline D:\flip.csv format
        new_rows = []
        for _, row in feedback_df.iterrows():
            try:
                start_dt = pd.to_datetime(row['timestamp'])
            except:
                start_dt = datetime.datetime.now()
                
            actual_dur = max(1.0, float(row['actual_duration']))
            resolved_dt = start_dt + datetime.timedelta(minutes=actual_dur)
            
            new_rows.append({
                'id': f"FEEDBACK_{np.random.randint(100000, 999999)}",
                'event_type': str(row['event_type']).lower(),
                'event_cause': str(row['event_cause']).lower(),
                'requires_road_closure': str(row['requires_road_closure']).lower(),
                'priority': str(row['priority']).lower(),
                'police_station': str(row['police_station']).lower(),
                'corridor': str(row['corridor']).lower(),
                'zone': str(row['zone']).lower(),
                'junction': str(row['junction']).lower(),
                'latitude': float(row['latitude']),
                'longitude': float(row['longitude']),
                'start_datetime': start_dt.strftime('%Y-%m-%d %H:%M:%S'),
                'resolved_datetime': resolved_dt.strftime('%Y-%m-%d %H:%M:%S'),
                'closed_datetime': "",
                'description': "Operator validated actual feed."
            })
            
        new_df = pd.DataFrame(new_rows)
        # Read base file and append new rows
        base_df = pd.read_csv(base_file)
        merged_df = pd.concat([base_df, new_df], ignore_index=True)
        temp_file = 'temp_merged_training_data.csv'
        merged_df.to_csv(temp_file, index=False)
        print(f"Merged dataset created with {len(merged_df)} total records.")
        
        try:
            train_hybrid_ensemble(temp_file)
            print("Model weights successfully updated with feedback override samples.")
            success = True
        except Exception as e:
            print(f"Retraining failed: {e}")
            success = False
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        return success
    else:
        print("No new feedback samples in database. Re-running baseline model training...")
        try:
            train_hybrid_ensemble(base_file)
            print("Baseline model weights successfully updated.")
            return True
        except Exception as e:
            print(f"Retraining baseline failed: {e}")
            return False

if __name__ == "__main__":
    run_retraining()
