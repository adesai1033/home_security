#!/usr/bin/env python3
"""
Nest Camera Object Detection and Dwell Time Tracker

This script monitors a Google Nest camera feed to:
1. Connect to Nest camera via RTSP stream
2. Detect people and objects using YOLO
3. Track how long objects remain in view (dwell time)
4. Save snapshots when people dwell longer than a threshold
5. Store and query events with timestamps and snapshots

Requirements:
    pip install ultralytics opencv-python google-auth-oauthlib requests

Configuration:
    - client_secrets.json: Google OAuth credentials file
    - token.json: Will be created/refreshed automatically
    - snapshots/: Directory for saved event images
    - events.json: Event history storage
"""

import time
import json
import os
import cv2
from ultralytics import YOLO
import argparse
import requests
from datetime import datetime
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow

# Nest API Configuration
NEST_PROJECT_ID = "enter project id here"
DEVICE_ID = "enter device id here"
SCOPES = ['https://www.googleapis.com/auth/sdm.service']
CLIENT_SECRETS_FILE = "client_secrets.json"
TOKEN_FILE = "token.json"

def get_nest_auth():
    """
    Handle Google OAuth2 authentication flow for Nest API access.
    
    Returns:
        Credentials: Valid Google OAuth credentials object
    """
    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        
        with open(TOKEN_FILE, 'w') as token:
            token.write(creds.to_json())
    
    return creds

def get_nest_stream_url():
    """
    Request RTSP stream URL from Nest camera via API.
    
    Returns:
        str: RTSP stream URL
        
    Raises:
        Exception: If stream URL cannot be obtained
    """
    creds = get_nest_auth()
    url = f"https://smartdevicemanagement.googleapis.com/v1/enterprises/{NEST_PROJECT_ID}/devices/{DEVICE_ID}:executeCommand"
    
    headers = {
        'Authorization': f'Bearer {creds.token}',
        'Content-Type': 'application/json',
    }
    payload = {
        "command": "sdm.devices.commands.CameraLiveStream.GenerateRtspStream"
    }
    
    print("Requesting RTSP stream...")
    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code == 200:
        stream_data = response.json()
        stream_url = stream_data.get('results', {}).get('streamUrls', {}).get('rtspUrl')
        if stream_url:
            print(f"Got RTSP stream URL: {stream_url}")
            return stream_url
    
    raise Exception(f"Failed to get stream URL. Status: {response.status_code}, Response: {response.text}")

class ObjectDwellTracker:
    """
    Tracks objects' presence and dwell time in video frames.
    
    Attributes:
        dwell_threshold_sec (int): Minimum seconds to consider as significant dwell
        object_states (dict): Tracks presence and timing for each object
        events (list): History of dwell events exceeding threshold
        snapshots_dir (str): Directory to save event snapshots
    """
    
    def __init__(self, dwell_threshold_sec=10):
        self.dwell_threshold_sec = dwell_threshold_sec
        self.object_states = {}  # {object_label: {"present": bool, "start_time": float}}
        self.events = []
        self.snapshots_dir = "snapshots"
        os.makedirs(self.snapshots_dir, exist_ok=True)

    def save_snapshot(self, frame, event):
        """
        Save a snapshot image for a dwell event.
        
        Args:
            frame: OpenCV image frame
            event (dict): Event data including timestamp
            
        Returns:
            str: Path to saved snapshot file
        """
        timestamp = datetime.fromtimestamp(event['end_time']).strftime('%Y%m%d_%H%M%S')
        filename = f"person_dwell_{timestamp}.jpg"
        filepath = os.path.join(self.snapshots_dir, filename)
        cv2.imwrite(filepath, frame)
        print(f"Saved snapshot: {filepath}")
        event['snapshot_path'] = filepath
        return filepath

    def update(self, detected_objects, frame=None):
        """
        Update object presence and track dwell times.
        
        Args:
            detected_objects (set): Labels of objects detected in current frame
            frame: Optional frame image for saving snapshots
        """
        current_time = time.time()
        
        # Check disappeared objects
        for obj in [o for o, data in self.object_states.items() if data["present"]]:
            if obj not in detected_objects:
                start = self.object_states[obj]["start_time"]
                duration = current_time - start
                self.object_states[obj]["present"] = False
                
                if duration >= self.dwell_threshold_sec:
                    event = {
                        "object_label": obj,
                        "start_time": start,
                        "end_time": current_time,
                        "duration_sec": duration
                    }
                    if obj == 'person' and frame is not None:
                        self.save_snapshot(frame, event)
                    self.events.append(event)

        # Update current objects
        for obj in detected_objects:
            if obj not in self.object_states:
                self.object_states[obj] = {"present": True, "start_time": current_time}
            elif not self.object_states[obj]["present"]:
                self.object_states[obj]["present"] = True
                self.object_states[obj]["start_time"] = current_time

    def save_events_to_json(self, filename="events.json"):
        """Save event history to JSON file"""
        with open(filename, 'w') as f:
            json.dump(self.events, f, indent=2)

    def load_events_from_json(self, filename="events.json"):
        """Load event history from JSON file"""
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                self.events = json.load(f)

    def query_events(self, object_label=None, min_duration_sec=10):
        """
        Query events matching criteria.
        
        Args:
            object_label (str): Filter by object type (e.g., 'person')
            min_duration_sec (int): Minimum dwell duration to include
            
        Returns:
            list: Matching events
        """
        return [event for event in self.events 
                if (object_label is None or event["object_label"] == object_label)
                and event["duration_sec"] >= min_duration_sec]

def process_nest_video(dwell_threshold_sec=10, output_json="events.json"):
    """
    Main video processing loop.
    
    Args:
        dwell_threshold_sec (int): Minimum dwell time to record
        output_json (str): Path to save events
    """
    tracker = ObjectDwellTracker(dwell_threshold_sec=dwell_threshold_sec)
    model = YOLO('yolov8n.pt')
    
    while True:
        try:
            stream_url = get_nest_stream_url()
            cap = cv2.VideoCapture(stream_url)
            if not cap.isOpened():
                raise Exception("Failed to open RTSP stream")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Run YOLO detection
                results = model(frame, verbose=False)[0]
                detected_labels = set()
                
                # Process detections
                for box in results.boxes.data:
                    class_id = int(box[5].item())
                    label = model.names[class_id]
                    detected_labels.add(label)
                    
                    # Draw boxes
                    x1, y1, x2, y2 = map(int, box[:4])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                # Update tracking
                tracker.update(detected_labels, frame.copy())
                
                # Display frame
                cv2.imshow('Nest Camera Feed', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                tracker.save_events_to_json(output_json)
                
        except Exception as e:
            print(f"Error: {e}")
            print("Retrying in 5 seconds...")
            time.sleep(5)
            continue
        finally:
            if 'cap' in locals():
                cap.release()
            cv2.destroyAllWindows()

def main():
    """Parse arguments and run appropriate mode"""
    parser = argparse.ArgumentParser(description="Nest Camera Object Dwell Tracker")
    parser.add_argument("--min-dwell-sec", type=int, default=10,
                      help="Minimum dwell time in seconds")
    parser.add_argument("--mode", choices=['monitor', 'query'], default='monitor',
                      help="'monitor' for live monitoring, 'query' for event queries")
    parser.add_argument("--query-object", type=str, default='person',
                      help="Object type to query for")
    args = parser.parse_args()
    
    if args.mode == 'monitor':
        process_nest_video(dwell_threshold_sec=args.min_dwell_sec)
    else:
        tracker = ObjectDwellTracker()
        tracker.load_events_from_json()
        results = tracker.query_events(
            object_label=args.query_object,
            min_duration_sec=args.min_dwell_sec
        )
        
        print(f"\nEvents where {args.query_object} dwelled > {args.min_dwell_sec} seconds:")
        for i, event in enumerate(results, 1):
            start_time = datetime.fromtimestamp(event['start_time'])
            duration = event['duration_sec']
            snapshot = event.get('snapshot_path', 'No snapshot')
            print(f"{i}. {start_time}: {duration:.1f} seconds - Snapshot: {snapshot}")

if __name__ == "__main__":
    main()
