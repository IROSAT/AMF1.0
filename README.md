# AMF1.0
#python code for hyphae detecttion
#important to have 1cm reference line on images


# Import required libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

class ImageLineProcessor:
    def __init__(self):
        self.reference_length_cm = 1.0  # Default reference length is 1 cm
        
    def load_image(self, image_path):
        """Load and return an image"""
        return cv2.imread(image_path)
    
    def preprocess_image(self, image):
        """Preprocess the image for line detection"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Apply threshold to isolate white lines
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        return thresh
    
    def detect_edges(self, preprocessed_image):
        """Detect edges using Canny"""
        return cv2.Canny(preprocessed_image, 50, 150, apertureSize=3)
    
    def find_contours(self, edges):
        """Find contours in the edge-detected image"""
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours
    
    def calculate_lengths(self, contours):
        """Calculate lengths of contours"""
        lengths = [cv2.arcLength(contour, False) for contour in contours]
        return lengths
    
    def draw_contours(self, image, contours):
        """Draw contours on the image"""
        image_with_contours = image.copy()
        cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 2)
        return image_with_contours
    
    def process_image(self, image_path, show_result=True):
        """Process an image and return the total length of white lines"""
        # Load and process image
        image = self.load_image(image_path)
        if image is None:
            print("Error: Could not load image at " + image_path)
            return 0
            
        preprocessed = self.preprocess_image(image)
        edges = self.detect_edges(preprocessed)
        contours = self.find_contours(edges)
        
        if not contours:
            print("No contours found in " + image_path)
            return 0
            
        # Calculate lengths
        lengths = self.calculate_lengths(contours)
        reference_length = max(lengths)  # Assuming longest contour is reference
        scale_factor = self.reference_length_cm / reference_length
        total_length_cm = sum(lengths) * scale_factor
        
        if show_result:
            # Draw and display results
            result_image = self.draw_contours(image, contours)
            plt.figure(figsize=(10, 5))
            plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
            plt.title(f"Total length: {total_length_cm:.2f} cm")
            plt.axis('off')
            plt.show()
            
        return total_length_cm

if __name__ == "__main__":
    # Example usage
    processor = ImageLineProcessor()
    # processor.process_image("your_image.jpg")  # Uncomment and modify path to use
