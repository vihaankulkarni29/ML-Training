# Auto AST - Automated Antibiotic Susceptibility Testing 🔬

An intelligent computer vision system for automatically detecting and analyzing antibiotic susceptibility zones on Kirby-Bauer bacterial culture plates using OpenCV and Python.

## 🎯 Overview

This project implements a complete pipeline for automated analysis of antibiotic susceptibility testing (AST) plates:
- **Plate Detection**: Automatically identifies the Petri dish boundary
- **Disk Detection**: Locates all antibiotic disks using circle detection
- **Zone Measurement**: Calculates inhibition zone diameters using radial intensity profiling
- **Visual Analysis**: Generates annotated images with measurements

## 📁 Project Structure

```
week5_auto_ast/
├── data/
│   ├── .gitkeep            # Data directory placeholder
│   └── test_plate.jpg      # Test image of culture plate (not tracked)
├── src/
│   ├── detect_zones.py     # Computer Vision Logic (ASTAnalyzer class)
│   └── main.py             # Main application entry point
├── results/
│   └── annotated_plate.jpg # Output with detected zones (generated)
├── requirements.txt        # Python dependencies
├── .gitignore             # Git ignore rules
└── README.md              # This file
```

## 🚀 Setup

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd week5_auto_ast
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Add your test image**:
   - Place a Kirby-Bauer plate image in `data/test_plate.jpg`
   - Image requirements:
     - Top-down view of Petri dish
     - Clear antibiotic disks (white circles)
     - Visible zones of inhibition
     - Good contrast and lighting

## 💻 Usage

Run the analysis:
```bash
python src/main.py
```

The system will:
1. Load the image from `data/test_plate.jpg`
2. Detect the Petri dish boundary
3. Identify all antibiotic disks
4. Measure inhibition zones
5. Display results with matplotlib
6. Save annotated image to `results/annotated_plate.jpg`

### Output Example
```
============================================================
  Automated Antibiotic Susceptibility Testing (AST)
  Kirby-Bauer Zone Detection System
============================================================

✓ Plate found: center=(248, 256), radius=238px
✓ Found 9 disks
  Disk 1: Zone diameter = 258px
  Disk 2: Zone diameter = 318px
  ...
```

## 🛠️ Technical Details

### Algorithm Pipeline

1. **Preprocessing**
   - Grayscale conversion
   - Gaussian blur (9x9 kernel) for noise reduction

2. **Plate Detection**
   - Hough Circle Transform for circular boundary detection
   - ROI masking to focus analysis on plate area

3. **Disk Detection**
   - Secondary Hough Circle detection with smaller radius parameters
   - Filters for white/bright circular objects

4. **Zone Measurement**
   - Adaptive thresholding in disk vicinity
   - Radial intensity profiling (36 directional samples)
   - Gradient-based edge detection
   - Median zone radius calculation

5. **Visualization**
   - Green circles: Antibiotic disks
   - Red circles: Zones of inhibition
   - White text: Diameter measurements

## 📦 Dependencies

- **opencv-python** (4.x): Industry standard computer vision library
- **numpy**: Numerical computing and array operations
- **matplotlib**: Visualization and result display
- **pandas**: Data manipulation and analysis
- **scipy**: Scientific computing utilities

## 🔬 Scientific Background

**Kirby-Bauer Disk Diffusion Test**: A standardized method to test bacterial susceptibility to antibiotics. Larger inhibition zones indicate greater bacterial susceptibility to the antibiotic.

## 📊 Future Enhancements

- [ ] Calibration for pixel-to-mm conversion
- [ ] CLSI/EUCAST guideline integration for resistance interpretation
- [ ] Batch processing for multiple plates
- [ ] CSV export of measurements
- [ ] Deep learning-based zone detection
- [ ] Support for different plate types

## 📝 License

This project is developed for educational and research purposes.

## 👤 Author

Vihaan - Medical Imaging & Computer Vision

## 🙏 Acknowledgments

- Kirby-Bauer method for antibiotic susceptibility testing
- OpenCV community for robust computer vision tools
