import cv2
import numpy as np
import argparse
import os

def correct_rolling_shutter(input_path, direction='vertical', inverted=False, strength=1.0, output_suffix='_corrected'):
    """
    Corrects rolling shutter deformation in images.
    
    Parameters:
    -----------
    input_path : str
        Path to the input image file
    direction : str
        Direction of rolling shutter effect: 'vertical' or 'horizontal'
    inverted : bool
        Whether the rolling shutter direction is inverted
    strength : float
        Strength of the correction (0.0 to 2.0 recommended)
    output_suffix : str
        Suffix to add to the output filename
        
    Returns:
    --------
    str
        Path to the corrected image file
    """
    # Read the input image
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"Could not read image: {input_path}")
    
    # Get image dimensions
    height, width = img.shape[:2]
    
    # Create output image
    corrected = np.zeros_like(img)
    
    # Define the transformation matrix based on direction and inversion
    if direction == 'vertical':
        # Vertical rolling shutter (rows captured at different times)
        for y in range(height):
            # Calculate shift amount based on row position
            progress = y / height  # 0.0 at top, 1.0 at bottom
            if inverted:
                progress = 1.0 - progress
                
            # Apply horizontal shift proportional to row position
            shift = int(strength * (progress - 0.5) * width * 0.1)  # Adjusted by strength
            
            # Apply the shift using affine transformation on this row
            M = np.float32([[1, 0, shift], [0, 1, 0]])
            row = img[y:y+1, :]
            transformed_row = cv2.warpAffine(row, M, (width, 1))
            corrected[y:y+1, :] = transformed_row
            
    elif direction == 'horizontal':
        # Horizontal rolling shutter (columns captured at different times)
        for x in range(width):
            # Calculate shift amount based on column position
            progress = x / width  # 0.0 at left, 1.0 at right
            if inverted:
                progress = 1.0 - progress
                
            # Apply vertical shift proportional to column position
            shift = int(strength * (progress - 0.5) * height * 0.1)  # Adjusted by strength
            
            # Apply the shift using affine transformation on this column
            M = np.float32([[1, 0, 0], [0, 1, shift]])
            col = img[:, x:x+1]
            transformed_col = cv2.warpAffine(col, M, (1, height))
            corrected[:, x:x+1] = transformed_col
    else:
        raise ValueError("Direction must be 'vertical' or 'horizontal'")
    
    # Save the corrected image
    filename, extension = os.path.splitext(input_path)
    output_path = f"{filename}{output_suffix}{extension}"
    cv2.imwrite(output_path, corrected)
    
    return output_path

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='Correct rolling shutter deformation in images')
    parser.add_argument('input_path', help='Path to the input image file')
    parser.add_argument('--direction', choices=['vertical', 'horizontal'], default='vertical',
                        help='Direction of rolling shutter effect')
    parser.add_argument('--inverted', action='store_true', 
                        help='Whether the rolling shutter direction is inverted')
    parser.add_argument('--strength', type=float, default=1.0,
                        help='Strength of the correction (0.0 to 2.0 recommended)')
    parser.add_argument('--output-suffix', default='_corrected',
                        help='Suffix to add to the output filename')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Apply correction
    try:
        output_path = correct_rolling_shutter(
            args.input_path,
            args.direction,
            args.inverted,
            args.strength,
            args.output_suffix
        )
        print(f"Corrected image saved to: {output_path}")
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
