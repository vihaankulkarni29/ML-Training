# week5_auto_ast

Starter scaffold for automated antibiotic susceptibility testing computer vision prototype.

## Layout
- data/: place test images (e.g., test_plate.jpg)
- src/detect_zones.py: CV logic to detect inhibition zones
- src/main.py: entry point wiring detection and CLI

## Quickstart
1) Install deps: `pip install -r requirements.txt`
2) Add your plate image to data/test_plate.jpg
3) Implement detection logic in src/detect_zones.py and run `python src/main.py --image data/test_plate.jpg` once ready.
