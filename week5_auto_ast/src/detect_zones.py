"""Detection stubs for inhibition zone analysis.

Replace stubs with actual OpenCV processing: load image, segment zones, measure diameters.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


def detect_zones(image_path: str | Path) -> List[Tuple[int, int, int]]:
    """Return list of detected zones as (x, y, radius) tuples.

    Currently a stub so tests and imports succeed. Implement with real CV steps:
    - grayscale + blur
    - threshold or edge detection
    - contour finding or Hough circles
    - measure diameters
    """
    _ = cv2.__version__  # keep linting quiet until implemented
    _ = np.zeros((1, 1), dtype=np.uint8)
    return []
