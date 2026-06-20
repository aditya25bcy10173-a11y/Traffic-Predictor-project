import pandas as pd
import numpy as np
import os

def run_eda(file_path):
    print(f"Loading dataset from {file_path}...")
    df = pd.read_csv(file_path)
    
    print("\n--- Basic Information ---")
    print(f"Total Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    
    # Check nulls
    print("\n--- Missing Values ---")
    null_counts = df.isnull().sum()
    for col, count in null_counts.items():
        if count > 0:
            print(f"  {col}: {count} ({count/len(df)*100:.2f}%)")
            
    # Parse Datetimes
    print("\n--- Parsing Timestamps & Calculating Durations ---")
    df['start_dt'] = pd.to_datetime(df['start_datetime'], errors='coerce')
    df['resolved_dt'] = pd.to_datetime(df['resolved_datetime'], errors='coerce')
    df['closed_dt'] = pd.to_datetime(df['closed_datetime'], errors='coerce')
    df['created_dt'] = pd.to_datetime(df['created_date'], errors='coerce')
    
    # We define resolution duration as resolved_dt - start_dt. If resolved_dt is null, we fallback to closed_dt.
    # If both are null, we fallback to resolved_at_address or ignore.
    df['end_dt'] = df['resolved_dt'].fillna(df['closed_dt'])
    df['duration_minutes'] = (df['end_dt'] - df['start_dt']).dt.total_seconds() / 60.0
    
    # Look at durations
    print("\n--- Duration Statistics (Minutes) ---")
    valid_durations = df['duration_minutes'].dropna()
    print(f"Valid durations count: {len(valid_durations)} ({len(valid_durations)/len(df)*100:.2f}%)")
    print(valid_durations.describe(percentiles=[0.25, 0.5, 0.75, 0.90, 0.95, 0.99]))
    
    # Count negative or zero durations
    neg_count = (df['duration_minutes'] <= 0).sum()
    print(f"Negative or zero durations: {neg_count}")
    
    # Categorical distributions
    print("\n--- Event Type Counts ---")
    print(df['event_type'].value_counts(dropna=False))
    
    print("\n--- Event Cause Counts ---")
    print(df['event_cause'].value_counts(dropna=False))
    
    print("\n--- Priority Counts ---")
    print(df['priority'].value_counts(dropna=False))
    
    print("\n--- Road Closure Requirements ---")
    print(df['requires_road_closure'].value_counts(dropna=False))
    
    print("\n--- Police Station Counts (Top 10) ---")
    print(df['police_station'].value_counts().head(10))
    
    print("\n--- Corridor Counts (Top 10) ---")
    print(df['corridor'].value_counts(dropna=False).head(10))
    
    print("\n--- Junction Counts (Top 10) ---")
    print(df['junction'].value_counts(dropna=False).head(10))
    
    # Spatial range
    print("\n--- Spatial Coordinates Range ---")
    print(f"Latitude range: {df['latitude'].min()} to {df['latitude'].max()}")
    print(f"Longitude range: {df['longitude'].min()} to {df['longitude'].max()}")
    
    # Save a clean dataset sample or copy to scratch for model testing
    output_dir = os.path.dirname(file_path)
    # Filter out anomalous durations (e.g. negative or > 7 days/10080 minutes)
    df_clean = df[df['duration_minutes'] > 0].copy()
    # Remove extreme outliers for baseline modeling if necessary
    df_clean = df_clean[df_clean['duration_minutes'] <= 10080] # max 7 days
    print(f"\nCleaned dataset size (0 < duration <= 7 days): {len(df_clean)}")
    
if __name__ == "__main__":
    run_eda(r"D:\flip.csv")
