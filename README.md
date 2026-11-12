# South African Bank Note Recognition System

**Course:** COMP702 - Image Processing and Computer Vision

**Authors:** Arjun Ramphal, Darian Robert, Lokadi Naicker, Keolin Naicker

## Overview

This repository contains a robust Computer Vision pipeline for identifying South African bank notes (R10, R20, R50, R100, R200). The system is designed to be invariant to scale, rotation, and facial side (obverse/reverse), and is specifically optimized to perform accurately against high-frequency textured backgrounds.

To fulfill the comparative requirements of the Honours project, this system implements a **Sequential Optimization Study**.

1. **Phase 1: Preprocessing Evaluation:** Tests contrast enhancement and noise reduction techniques (CLAHE, Bilateral, Gaussian, Equalization) to find the optimal structural clarifier.
2. **Phase 2: Segmentation & Geometric Normalization:** Tests boundary isolation methods (Perspective Warping, Exact Masking, Padded Bounding Boxes, Adaptive Thresholding). 
3. **Phase 3: Feature Extraction & Template Matching:** Evaluates the final isolated Regions of Interest (ROI) across four distinct classification architectures:
* **SIFT** (Floating-point geometry matched via FLANN KD-Tree)
* **ORB** (Binary descriptors matched via cross-checked BF Hamming)
* **AKAZE** (Non-linear scale space matched via cross-checked BF Hamming)
* **Color Histograms** (3D HSV color mass matched via Histogram Intersection)


## My contribution

I contributed to developing the team's banknote-recognition model. The pipeline is presented as shared work, with all four project authors credited above.

## Installation & Setup

1. Ensure Python 3.10+ is installed on your system.
2. Clone this repository and enter its directory:

   ```bash
   git clone https://github.com/DarianRobert/SA-Banknote-Recognition.git
   cd SA-Banknote-Recognition
   ```
3. Install the required dependencies:

```bash
python -m pip install -r requirements.txt

```

OpenCV is kept on 4.x because this implementation uses the 4.x AKAZE API.

### Dependencies (`requirements.txt`)

```text
opencv-python>=4.5,<5
numpy
scikit-learn
```

## Dataset Configuration

The system requires a strict directory structure to automatically label and process the images. Ensure your repository is structured as follows before execution:

```text
SA-Banknote-Recognition/
│
├── README.md                 
├── requirements.txt          
├── main.py                   
│
├── reference_templates/      # Directory for baseline white-background template notes
│   ├── R10/
│   ├── R20/
│   ├── R50/
│   ├── R100/
│   └── R200/
│
└── test_images/              # Directory for evaluation notes on textured backgrounds
    ├── R10/
    ├── R20/
    ├── R50/
    ├── R100/
    └── R200/

```

## Execution

Run the system pipeline via the terminal. Default paths are relative to `main.py`, so it can be launched from any directory:

```bash
python main.py
# Optional alternative paths
python main.py --templates path/to/templates --tests path/to/images --snapshots path/to/output
```

Images are loaded in a stable order. Missing or empty datasets report an error, and unreadable images are skipped with a warning. Run the small regression suite with `python -m unittest discover -s tests -v`.

This is a comparative coursework study: the evaluation images are also used to select pipeline stages. The printed accuracy is a selection score on those images, not an independent held-out estimate. Feature matching uses FLANN with a ratio test or cross-checked BF Hamming; geometric RANSAC/MAGSAC verification is not implemented.

### Expected Output

1. **Phase 1 & 2 Diagnostics:** The system will output real-time Scikit-Learn accuracy metrics for every preprocessing and segmentation technique, automatically locking in the mathematical winner for the next stage.
2. **Phase 3 Predictions:** The system will print individual prediction results for every image in the test set across the four distinct extraction architectures (SIFT, ORB, AKAZE, Hist).
3. **Snapshot Generation:** A `report_snapshots/` directory will be automatically generated, saving visual evidence of the preprocessing and segmentation results for use in the final academic report.
4. **Final Pipeline Configuration Report:** Upon completion, a comprehensive summary block will output the absolute best combination of Preprocessing, Segmentation, and Feature Extraction, alongside the final achieved accuracy.


## Saved example

A preprocessing snapshot from the supplied project:

![CLAHE preprocessing of a South African R10 banknote](report_snapshots/Stage1_CLAHE_R10_1.jpg)

[Original team repository](https://github.com/ArjunRamphal/COMP702Project) · [My portfolio](https://darianrobert.github.io/)
