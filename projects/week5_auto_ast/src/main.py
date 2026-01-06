"""
Main application for Automated Antibiotic Susceptibility Testing (Auto AST).

This script loads a bacterial culture plate image and detects antibiotic
susceptibility zones using computer vision techniques.
"""

import os
import cv2
import matplotlib.pyplot as plt
from detect_zones import ASTAnalyzer


def main():
    """Main application entry point."""
    # Define paths
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_image_path = os.path.join(current_dir, 'data', 'test_plate.jpg')
    output_path = os.path.join(current_dir, 'results', 'annotated_plate.jpg')
    
    # Check if test image exists
    if not os.path.exists(test_image_path):
        print(f"❌ Test image not found at {test_image_path}")
        print("Please add your test_plate.jpg to the data/ folder")
        return
    
    print("="*60)
    print("  Automated Antibiotic Susceptibility Testing (AST)")
    print("  Kirby-Bauer Zone Detection System")
    print("="*60)
    print()
    
    # Create analyzer and run analysis
    analyzer = ASTAnalyzer(test_image_path)
    
    try:
        zones = analyzer.analyze(output_path=output_path)
        
        # Display results using matplotlib
        print("\nDisplaying results...")
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Original image
        axes[0].imshow(cv2.cvtColor(analyzer.original_image, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original Plate Image', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # Annotated image
        annotated = cv2.imread(output_path)
        axes[1].imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        axes[1].set_title(f'Detected Zones (n={len(zones)})', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Print summary
        print("\n" + "="*60)
        print("ZONE MEASUREMENTS SUMMARY")
        print("="*60)
        for i, zone in enumerate(zones, 1):
            print(f"Disk {i}: Zone Diameter = {zone['zone_diameter']} pixels")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
