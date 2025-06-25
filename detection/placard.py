import cv2
import numpy as np
import torch
from ultralytics import YOLO
from collections import defaultdict
import time

class PlacardTracker:
    def __init__(self, model_path='placard_model.pt', conf_threshold=0.5, iou_threshold=0.5):
        """
        Initialize the placard detection and tracking system
        
        Args:
            model_path: Path to trained YOLO model
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Tracking variables
        self.track_history = defaultdict(lambda: [])
        self.next_id = 0
        self.active_tracks = {}
        self.max_disappeared = 30  # Max frames before removing a track
        
        # Color mapping for different placard types
        self.colors = {
            'explosive': (0, 0, 255),      # Red
            'flammable': (0, 255, 255),    # Yellow
            'toxic': (0, 255, 0),          # Green
            'corrosive': (255, 0, 0),      # Blue
            'radioactive': (128, 0, 128),  # Purple
            'oxidizer': (0, 165, 255),     # Orange
            'default': (255, 255, 255)     # White
        }
    
    def detect_placards(self, frame):
        """
        Detect placards in a frame using YOLO
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            List of detection dictionaries with bbox, confidence, and class
        """
        results = self.model(frame, conf=self.conf_threshold, iou=self.iou_threshold)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Extract bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Get class name
                    class_name = self.model.names[class_id] if class_id < len(self.model.names) else 'unknown'
                    
                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(confidence),
                        'class': class_name,
                        'class_id': class_id
                    })
        
        return detections
    
    def calculate_centroid(self, bbox):
        """Calculate centroid of bounding box"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)
    
    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def update_tracks(self, detections):
        """
        Update tracking for detected placards using centroid tracking
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            List of tracked objects with IDs
        """
        if len(detections) == 0:
            # Mark all existing tracks as disappeared
            for track_id in list(self.active_tracks.keys()):
                self.active_tracks[track_id]['disappeared'] += 1
                if self.active_tracks[track_id]['disappeared'] > self.max_disappeared:
                    del self.active_tracks[track_id]
            return []
        
        # Calculate centroids for current detections
        input_centroids = []
        for detection in detections:
            centroid = self.calculate_centroid(detection['bbox'])
            input_centroids.append(centroid)
        
        # If no existing tracks, register all detections as new tracks
        if len(self.active_tracks) == 0:
            for i, detection in enumerate(detections):
                self.register_track(detection, input_centroids[i])
        else:
            # Match existing tracks to current detections
            track_ids = list(self.active_tracks.keys())
            track_centroids = [self.active_tracks[tid]['centroid'] for tid in track_ids]
            
            # Compute distance matrix
            distances = np.linalg.norm(np.array(track_centroids)[:, np.newaxis] - 
                                     np.array(input_centroids), axis=2)
            
            # Find minimum distances
            rows = distances.min(axis=1).argsort()
            cols = distances.argmin(axis=1)[rows]
            
            used_row_indices = set()
            used_col_indices = set()
            
            # Update existing tracks
            for (row, col) in zip(rows, cols):
                if row in used_row_indices or col in used_col_indices:
                    continue
                
                # If distance is reasonable, update the track
                if distances[row, col] <= 50:  # Max distance threshold
                    track_id = track_ids[row]
                    self.active_tracks[track_id]['centroid'] = input_centroids[col]
                    self.active_tracks[track_id]['bbox'] = detections[col]['bbox']
                    self.active_tracks[track_id]['confidence'] = detections[col]['confidence']
                    self.active_tracks[track_id]['class'] = detections[col]['class']
                    self.active_tracks[track_id]['disappeared'] = 0
                    
                    used_row_indices.add(row)
                    used_col_indices.add(col)
            
            # Register new tracks for unmatched detections
            unused_col_indices = set(range(0, len(input_centroids))).difference(used_col_indices)
            for col in unused_col_indices:
                self.register_track(detections[col], input_centroids[col])
            
            # Mark unmatched tracks as disappeared
            unused_row_indices = set(range(0, len(track_centroids))).difference(used_row_indices)
            for row in unused_row_indices:
                track_id = track_ids[row]
                self.active_tracks[track_id]['disappeared'] += 1
                if self.active_tracks[track_id]['disappeared'] > self.max_disappeared:
                    del self.active_tracks[track_id]
        
        # Return current tracks
        return list(self.active_tracks.items())
    
    def register_track(self, detection, centroid):
        """Register a new track"""
        self.active_tracks[self.next_id] = {
            'centroid': centroid,
            'bbox': detection['bbox'],
            'confidence': detection['confidence'],
            'class': detection['class'],
            'disappeared': 0
        }
        self.next_id += 1
    
    def draw_tracks(self, frame, tracks):
        """
        Draw bounding boxes and track information on frame
        
        Args:
            frame: Input frame
            tracks: List of tracked objects
            
        Returns:
            Frame with drawn annotations
        """
        annotated_frame = frame.copy()
        
        for track_id, track_info in tracks:
            bbox = track_info['bbox']
            class_name = track_info['class']
            confidence = track_info['confidence']
            centroid = track_info['centroid']
            
            # Get color for this class
            color = self.colors.get(class_name, self.colors['default'])
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, 
                         (bbox[0], bbox[1]), 
                         (bbox[2], bbox[3]), 
                         color, 2)
            
            # Draw centroid
            cv2.circle(annotated_frame, centroid, 4, color, -1)
            
            # Draw track history
            if track_id in self.track_history:
                points = self.track_history[track_id]
                if len(points) > 1:
                    for i in range(1, len(points)):
                        cv2.line(annotated_frame, points[i-1], points[i], color, 2)
            
            # Add track history point
            self.track_history[track_id].append(centroid)
            if len(self.track_history[track_id]) > 30:  # Keep last 30 points
                self.track_history[track_id].pop(0)
            
            # Draw label
            label = f"ID:{track_id} {class_name} {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Draw label background
            cv2.rectangle(annotated_frame,
                         (bbox[0], bbox[1] - label_size[1] - 10),
                         (bbox[0] + label_size[0], bbox[1]),
                         color, -1)
            
            # Draw label text
            cv2.putText(annotated_frame, label,
                       (bbox[0], bbox[1] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return annotated_frame
    
    def process_frame(self, frame):
        """
        Process a single frame: detect and track placards
        
        Args:
            frame: Input frame
            
        Returns:
            Annotated frame with detections and tracks
        """
        # Detect placards
        detections = self.detect_placards(frame)
        
        # Update tracks
        tracks = self.update_tracks(detections)
        
        # Draw annotations
        annotated_frame = self.draw_tracks(frame, tracks)
        
        return annotated_frame, tracks


class MotionEventProcessor:
    """
    Integration class for existing motion detection system
    """
    def __init__(self, placard_model_path='placard_model.pt'):
        self.placard_tracker = PlacardTracker(placard_model_path)
        self.motion_detector = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        
    def detect_motion(self, frame):
        """Basic motion detection"""
        fg_mask = self.motion_detector.apply(frame)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area
        motion_detected = False
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Minimum area threshold
                motion_detected = True
                break
                
        return motion_detected, fg_mask
    
    def process_motion_event(self, frame):
        """
        Process frame when motion is detected
        
        Args:
            frame: Current frame
            
        Returns:
            Tuple of (annotated_frame, placard_tracks, motion_detected)
        """
        motion_detected, motion_mask = self.detect_motion(frame)
        
        if motion_detected:
            # Process placards when motion is detected
            annotated_frame, tracks = self.placard_tracker.process_frame(frame)
            return annotated_frame, tracks, True
        else:
            return frame, [], False


# Example usage and integration
def main():
    """Example of how to integrate with existing motion detection system"""
    
    # Initialize the processor
    processor = MotionEventProcessor('path/to/your/trained/model.pt')
    
    # Open video source (camera or file)
    cap = cv2.VideoCapture(0)  # Use 0 for webcam, or path to video file
    
    # Set up video writer for saving results (optional)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Process the frame
        processed_frame, tracks, motion_detected = processor.process_motion_event(frame)
        
        # Log placard detections
        if motion_detected and tracks:
            print(f"Frame {frame_count}: Detected {len(tracks)} placards")
            for track_id, track_info in tracks:
                print(f"  - Track {track_id}: {track_info['class']} "
                      f"(conf: {track_info['confidence']:.2f})")
        
        # Display the frame
        cv2.imshow('Placard Detection and Tracking', processed_frame)
        
        # Save frame (optional)
        out.write(processed_frame)
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()