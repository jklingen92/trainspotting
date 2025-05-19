import cv2
import numpy as np

def calculate_frame_difference(frame1, frame2, threshold=25, return_visualization=False, sample_rate=1.0, grid_size=None):
    """
    Calculate the difference between two frames, with efficient sampling for large-scale motion detection.
    
    Args:
        frame1: First frame (numpy array)
        frame2: Second frame (numpy array)
        threshold: Pixel difference threshold (0-255) to consider a pixel as "changed"
        return_visualization: Whether to return a visualization image of the differences
        sample_rate: Fraction of pixels to sample (0.0-1.0), where 1.0 uses all pixels
                    For very high-res video, try 0.1-0.3 for significant speedup
        grid_size: If provided, divides the image into an NxN grid and samples from each cell
                  This ensures good coverage across the entire frame
                  
    Returns:
        If return_visualization=False:
            - diff_percent: Percentage of pixels that changed beyond the threshold
            - diff_magnitude: Average magnitude of change across all pixels
            - region_activity: Dictionary with activity levels in different regions (if grid_size is used)
        If return_visualization=True:
            - Returns tuple of (diff_percent, diff_magnitude, region_activity, visualization_image)
    """
    # Make sure frames are the same shape
    if frame1.shape != frame2.shape:
        raise ValueError(f"Frames have different shapes: {frame1.shape} vs {frame2.shape}")
    
    # Convert to grayscale if the frames are color (more efficient for difference calculation)
    if len(frame1.shape) == 3 and frame1.shape[2] == 3:
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    else:
        # Already grayscale
        gray1 = frame1
        gray2 = frame2
    
    # Calculate full frame difference for visualization if needed
    if return_visualization:
        full_frame_diff = cv2.absdiff(gray1, gray2)
    
    # If using grid sampling
    region_activity = {}
    if grid_size is not None:
        h, w = gray1.shape
        cell_h = h // grid_size
        cell_w = w // grid_size
        
        # Initialize mask for sampling
        mask = np.zeros_like(gray1, dtype=bool)
        
        # Process each grid cell
        for y in range(grid_size):
            for x in range(grid_size):
                # Cell boundaries
                y1, y2 = y * cell_h, (y + 1) * cell_h
                x1, x2 = x * cell_w, (x + 1) * cell_w
                
                # If not sampling all pixels in the cell
                if sample_rate < 1.0:
                    # Create random sampling mask for this cell
                    cell_mask = np.zeros((cell_h, cell_w), dtype=bool)
                    sample_count = int(cell_h * cell_w * sample_rate)
                    sample_indices = np.random.choice(cell_h * cell_w, sample_count, replace=False)
                    indices = np.unravel_index(sample_indices, (cell_h, cell_w))
                    cell_mask[indices] = True
                    
                    # Apply to the full mask
                    mask[y1:y2, x1:x2] = cell_mask
                else:
                    # Sample all pixels in this cell
                    mask[y1:y2, x1:x2] = True
                
                # Calculate difference for this cell
                cell_diff = cv2.absdiff(gray1[y1:y2, x1:x2], gray2[y1:y2, x1:x2])
                cell_pixels = cell_diff.size
                changed_pixels = np.sum(cell_diff > threshold)
                cell_diff_percent = (changed_pixels / cell_pixels) * 100
                cell_diff_magnitude = np.mean(cell_diff)
                
                # Store results for this cell
                region_key = f"grid_{y}_{x}"
                region_activity[region_key] = {
                    "diff_percent": cell_diff_percent,
                    "diff_magnitude": cell_diff_magnitude,
                    "coords": (x1, y1, x2, y2)
                }
        
        # Use the mask to sample pixels
        samples1 = gray1[mask]
        samples2 = gray2[mask]
        samples_diff = np.abs(samples1 - samples2)
    
    else:
        # Simple random sampling across the whole frame
        if sample_rate < 1.0:
            # Create sampling mask
            mask = np.zeros_like(gray1, dtype=bool)
            sample_count = int(gray1.size * sample_rate)
            sample_indices = np.random.choice(gray1.size, sample_count, replace=False)
            indices = np.unravel_index(sample_indices, gray1.shape)
            mask[indices] = True
            
            # Use the mask to sample pixels
            samples1 = gray1[mask]
            samples2 = gray2[mask]
            samples_diff = np.abs(samples1 - samples2)
        else:
            # Use all pixels
            samples_diff = cv2.absdiff(gray1, gray2).flatten()
    
    # Calculate statistics on sampled pixels
    total_samples = samples_diff.size
    changed_samples = np.sum(samples_diff > threshold)
    diff_percent = (changed_samples / total_samples) * 100
    diff_magnitude = np.mean(samples_diff)
    
    # Find regions with the most activity
    if grid_size is not None:
        active_regions = sorted(
            region_activity.items(), 
            key=lambda x: x[1]["diff_percent"], 
            reverse=True
        )
        
        # Determine if significant motion is detected
        motion_detected = diff_percent > threshold
        
        # Add overall stats to region_activity
        region_activity["overall"] = {
            "diff_percent": diff_percent,
            "diff_magnitude": diff_magnitude,
            "motion_detected": motion_detected
        }
    
    if not return_visualization:
        if grid_size is not None:
            return diff_percent, diff_magnitude, region_activity
        else:
            return diff_percent, diff_magnitude
    else:
        # Create visualization - apply threshold and colorize
        if not 'full_frame_diff' in locals():
            full_frame_diff = cv2.absdiff(gray1, gray2)
            
        _, thresh_diff = cv2.threshold(full_frame_diff, threshold, 255, cv2.THRESH_BINARY)
        
        # Colorize the difference visualization (heat map)
        diff_color = cv2.applyColorMap(full_frame_diff, cv2.COLORMAP_JET)
        
        # Add text with statistics
        result = diff_color.copy()
        cv2.putText(result, f"Changed: {diff_percent:.2f}%", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(result, f"Magnitude: {diff_magnitude:.2f}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Highlight active regions if using grid
        if grid_size is not None:
            # Get the top active regions
            top_regions = active_regions[:3]  # Top 3 most active regions
            
            for _, region_data in top_regions:
                # Only highlight regions with significant activity
                if region_data["diff_percent"] > threshold:
                    x1, y1, x2, y2 = region_data["coords"]
                    cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Add activity percentage near the region
                    label = f"{region_data['diff_percent']:.1f}%"
                    cv2.putText(result, label, (x1, y1-5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        if grid_size is not None:
            return diff_percent, diff_magnitude, region_activity, result
        else:
            return diff_percent, diff_magnitude, result


def detect_large_motion(frame1, frame2, threshold=10, grid_size=3, sample_rate=0.3):
    """
    Efficiently detect large motion between frames by using a grid-based sampling approach.
    Optimized for scenarios where motion is expected to occupy a significant portion of the frame.
    
    Args:
        frame1: First frame (numpy array)
        frame2: Second frame (numpy array)
        threshold: Percentage threshold to consider significant motion (1-100)
        grid_size: Size of the grid to divide the frame into (e.g., 3 creates a 3x3 grid)
        sample_rate: Fraction of pixels to sample in each grid cell (0.0-1.0)
        
    Returns:
        - motion_detected: Boolean indicating if significant motion was detected
        - active_regions: List of grid cells where significant motion was detected
        - overall_change: Overall percentage of change across the frame
    """
    # Get region-based difference analysis
    _, _, region_activity = calculate_frame_difference(
        frame1, frame2, 
        threshold=25,
        sample_rate=sample_rate,
        grid_size=grid_size
    )
    
    # Check overall motion
    overall_change = region_activity["overall"]["diff_percent"]
    motion_detected = overall_change > threshold
    
    # Find which regions have significant activity
    active_regions = []
    for region_key, data in region_activity.items():
        if region_key != "overall" and data["diff_percent"] > threshold:
            active_regions.append({
                "region": region_key,
                "activity": data["diff_percent"],
                "coords": data["coords"]
            })
    
    # Sort by activity level (highest first)
    active_regions.sort(key=lambda x: x["activity"], reverse=True)
    
    return motion_detected, active_regions, overall_change

# Example usage:
# motion_detected, active_regions, change_percent = detect_large_motion(frame1, frame2, threshold=15)
# 
# # For visualization:
# _, _, region_activity, visualization = calculate_frame_difference(
#     frame1, frame2, 
#     threshold=25,
#     return_visualization=True, 
#     sample_rate=0.3, 
#     grid_size=3
# )
# cv2.imshow("Motion Detection", visualization)
# cv2.waitKey(0)
