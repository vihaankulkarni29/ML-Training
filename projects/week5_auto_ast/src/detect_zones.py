"""
Computer Vision Logic for detecting antibiotic susceptibility zones.

This module contains the ASTAnalyzer class for:
- Loading and preprocessing plate images
- Detecting the Petri dish (plate)
- Detecting antibiotic disks
- Measuring zones of inhibition
- Visualizing and saving results
"""

import cv2
import numpy as np
import os


class ASTAnalyzer:
    """Analyzer for Antibiotic Susceptibility Testing (Kirby-Bauer) plates."""
    
    def __init__(self, image_path):
        """
        Initialize the AST Analyzer.
        
        Args:
            image_path: Path to the test plate image
        """
        self.image_path = image_path
        self.original_image = None
        self.gray = None
        self.blurred = None
        self.plate_circle = None
        self.plate_mask = None
        self.disks = []
        self.zones = []
        
    def load_image(self):
        """Load the image from file."""
        print(f"Loading image from {self.image_path}...")
        self.original_image = cv2.imread(self.image_path)
        if self.original_image is None:
            raise FileNotFoundError(f"Image not found at {self.image_path}")
        print(f"✓ Image loaded: {self.original_image.shape}")
        
    def preprocess(self):
        """Preprocess the image for analysis."""
        print("Preprocessing image...")
        # Convert to grayscale
        self.gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        self.blurred = cv2.GaussianBlur(self.gray, (9, 9), 2)
        print("✓ Preprocessing complete")
        
    def detect_plate(self):
        """Detect the Petri dish (largest circle) in the image."""
        print("Detecting Petri dish...")
        
        # Use HoughCircles to find circles
        circles = cv2.HoughCircles(
            self.blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=500,  # Only one large circle expected
            param1=50,
            param2=30,
            minRadius=100,
            maxRadius=800
        )
        
        if circles is None:
            print("⚠ Warning: No plate detected! Using entire image.")
            # Use entire image as ROI
            h, w = self.gray.shape
            self.plate_circle = (w//2, h//2, min(w, h)//2)
        else:
            circles = np.uint16(np.around(circles))
            # Take the largest circle
            largest = max(circles[0], key=lambda c: c[2])
            self.plate_circle = tuple(largest)
            print(f"✓ Plate found: center=({self.plate_circle[0]}, {self.plate_circle[1]}), radius={self.plate_circle[2]}px")
        
        # Create a mask for the plate
        self.plate_mask = np.zeros(self.gray.shape, dtype=np.uint8)
        cv2.circle(self.plate_mask, (self.plate_circle[0], self.plate_circle[1]), 
                   self.plate_circle[2], 255, -1)
        
    def detect_disks(self):
        """Detect antibiotic disks (small white circles) within the plate."""
        print("Detecting antibiotic disks...")
        
        # Apply mask to focus only on the plate
        masked = cv2.bitwise_and(self.blurred, self.blurred, mask=self.plate_mask)
        
        # Detect smaller circles (disks) - they are typically bright white
        circles = cv2.HoughCircles(
            masked,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=50,  # Disks should be at least 50px apart
            param1=50,
            param2=20,
            minRadius=8,
            maxRadius=40
        )
        
        if circles is None:
            print("⚠ Warning: No disks detected!")
            return
        
        circles = np.uint16(np.around(circles))
        self.disks = [tuple(c) for c in circles[0]]
        print(f"✓ Found {len(self.disks)} disks")
        
    def measure_zones(self):
        """Measure the zone of inhibition around each disk."""
        print("Measuring zones of inhibition...")
        
        for i, disk in enumerate(self.disks):
            x, y, r = int(disk[0]), int(disk[1]), int(disk[2])
            
            # Create a region of interest around the disk
            roi_size = 150  # Analyze area around disk
            x1, y1 = max(0, x - roi_size), max(0, y - roi_size)
            x2, y2 = min(self.gray.shape[1], x + roi_size), min(self.gray.shape[0], y + roi_size)
            
            roi = self.gray[y1:y2, x1:x2]
            
            # Skip if ROI is empty or too small
            if roi.size == 0 or roi.shape[0] < 10 or roi.shape[1] < 10:
                continue
            
            # Apply adaptive thresholding to find the zone boundary
            thresh = cv2.adaptiveThreshold(
                roi, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 51, 10
            )
            
            # Find contours in the thresholded ROI
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Find the largest contour that likely represents the zone
            zone_radius = r  # Default to disk radius if no zone found
            
            if contours:
                # Find contours that contain the disk center
                for contour in contours:
                    # Adjust contour coordinates to full image
                    contour_shifted = contour + np.array([x1, y1])
                    
                    # Check if disk center is inside this contour
                    if cv2.pointPolygonTest(contour_shifted, (float(x), float(y)), False) >= 0:
                        # Calculate the radius from this contour
                        moments = cv2.moments(contour)
                        if moments['m00'] != 0:
                            area = cv2.contourArea(contour)
                            contour_radius = int(np.sqrt(area / np.pi))
                            if contour_radius > zone_radius:
                                zone_radius = contour_radius
            
            # Alternative method: radial intensity profile
            max_zone_radius = self._measure_radial_zone(x, y, r)
            zone_radius = max(zone_radius, max_zone_radius)
            
            # Store zone information
            zone_diameter = zone_radius * 2
            self.zones.append({
                'disk_center': (x, y),
                'disk_radius': r,
                'zone_radius': zone_radius,
                'zone_diameter': zone_diameter
            })
            
            print(f"  Disk {i+1}: Zone diameter = {zone_diameter}px")
            
    def _measure_radial_zone(self, cx, cy, disk_radius):
        """
        Measure zone using radial intensity profile.
        
        Args:
            cx, cy: Center coordinates of disk
            disk_radius: Radius of the disk
            
        Returns:
            Estimated zone radius
        """
        max_radius = 100  # Maximum zone size to check
        angles = np.linspace(0, 2*np.pi, 36)  # Check 36 directions
        
        zone_radii = []
        
        for angle in angles:
            # Sample along this angle
            radii = np.arange(disk_radius + 5, max_radius)
            intensities = []
            
            for r in radii:
                x = int(cx + r * np.cos(angle))
                y = int(cy + r * np.sin(angle))
                
                # Check bounds
                if 0 <= x < self.gray.shape[1] and 0 <= y < self.gray.shape[0]:
                    intensities.append(self.gray[y, x])
                else:
                    break
            
            if len(intensities) > 5:
                # Find edge using gradient
                gradient = np.gradient(intensities)
                edge_idx = np.argmax(np.abs(gradient))
                zone_radii.append(radii[edge_idx])
        
        if zone_radii:
            return int(np.median(zone_radii))
        return disk_radius
        
    def visualize_results(self, output_path):
        """
        Create annotated image with detected disks and zones.
        
        Args:
            output_path: Path to save the annotated image
        """
        print("Creating visualization...")
        result = self.original_image.copy()
        
        # Draw the plate boundary (optional, in blue)
        if self.plate_circle:
            cv2.circle(result, (self.plate_circle[0], self.plate_circle[1]), 
                      self.plate_circle[2], (255, 100, 0), 2)
        
        # Draw disks and zones
        for zone_info in self.zones:
            x, y = zone_info['disk_center']
            disk_r = zone_info['disk_radius']
            zone_r = zone_info['zone_radius']
            diameter = zone_info['zone_diameter']
            
            # Draw zone of inhibition (RED circle)
            cv2.circle(result, (x, y), zone_r, (0, 0, 255), 3)
            
            # Draw disk (GREEN circle)
            cv2.circle(result, (x, y), disk_r, (0, 255, 0), 2)
            
            # Add diameter label
            label = f"{diameter}px"
            cv2.putText(result, label, (x + zone_r + 10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(result, label, (x + zone_r + 10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        
        # Save the result
        cv2.imwrite(output_path, result)
        print(f"✓ Annotated image saved to {output_path}")
        
        return result
        
    def analyze(self, output_path=None):
        """
        Run the complete analysis pipeline.
        
        Args:
            output_path: Path to save annotated image (optional)
            
        Returns:
            List of zone measurements
        """
        self.load_image()
        self.preprocess()
        self.detect_plate()
        self.detect_disks()
        self.measure_zones()
        
        if output_path:
            self.visualize_results(output_path)
        
        print(f"\n{'='*50}")
        print(f"Analysis Complete: {len(self.zones)} zones measured")
        print(f"{'='*50}")
        
        return self.zones
