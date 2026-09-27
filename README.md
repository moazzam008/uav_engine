# 🚁 UAV Intelligence Engine (Research Edition)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-ORB%20Stitching-green)
![YOLOv8](https://img.shields.io/badge/YOLOv8-VisDrone%20AI-orange)

A standalone Windows-executable processing engine that takes raw, fragmented video feeds from agricultural or security drones and transforms them into actionable geospatial intelligence. 

## 🌟 Features
- **Seamless Topographical Stitching:** Leverages OpenCV's ORB feature extraction and RANSAC homography to stitch individual video frames into a continuous 2D map. Implements custom alpha-feather blending to eliminate harsh seams.
- **Aerial AI Object Detection:** Integrates a Hugging Face-hosted YOLOv8 neural network trained specifically on the `VisDrone` dataset to detect vehicles (cars, trucks, buses) from a top-down aerial perspective.
- **Spectral Color Tracking (HSV):** Includes a fallback morphological engine for identifying assets (like diseased crops) based on specific HSV color ranges.
- **Actionable Analytics:** Automatically generates and exports `.csv` datasets mapping the exact X/Y coordinate locations and confidence scores of every detected ground asset.
- **Enterprise Dashboard:** Built with CustomTkinter, featuring real-time telemetry, advanced hyperparameter tuning (Confidence, Downscaling), and an integrated system console.

## 📂 Project Structure
```text
uav_engine/
├── src/
│   ├── main.py        # GUI and Pipeline Coordinator
│   ├── stitcher.py    # Alpha-blending ORB Stitching Engine
│   └── analyzer.py    # YOLOv8 VisDrone / HSV Detection Engine
├── output/
│   ├── csv/           # Exported Analytics
│   └── maps/          # Exported Orthomosaics
└── requirements.txt
```

## 🚀 Quick Start
### Running the Source Code
1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/uav-intelligence-engine.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch the engine:
   ```bash
   python src/main.py
   ```

*Note: The AI Engine will automatically connect to Hugging Face to download the `VisDrone` neural weights on its first launch.*

## ⚙️ Compilation (Windows .exe)
To package the app into a standalone Windows executable, use PyInstaller:
```bash
pyinstaller --noconfirm --onefile --windowed --collect-data customtkinter --collect-all torchvision --hidden-import cv2 --hidden-import numpy --hidden-import pandas --hidden-import huggingface_hub src\main.py
```
