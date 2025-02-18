# Automatic Number Plate Recognition (ANPR) Task

## Overview
This project is an **Automatic Number Plate Recognition (ANPR)** system designed to detect vehicles and recognize their license plates from video footage. It utilizes **YOLO (You Only Look Once) models** for object detection and **SORT (Simple Online and Realtime Tracking)** for tracking, ensuring efficient and accurate recognition even in real-time scenarios.

## Project Structure
```
ANPR-Task2/
├── add_missing_data.py
├── data/
│   ├── train/
│   │   ├── ADL-Rundle-6/
│   │   ├── ADL-Rundle-8/
│   │   ├── ETH-Bahnhof/
│   │   ├── ETH-Pedcross2/
│   │   ├── ETH-Sunnyday/
│   │   ├── KITTI-13/
│   │   ├── KITTI-17/
│   │   ├── PETS09-S2L1/
│   │   ├── TUD-Campus/
│   │   ├── TUD-Stadtmitte/
│   │   ├── Venice-2/
│   ├── license_plate_detector.pt
├── main.py
├── README.md
├── requirements.txt
├── sort.py
├── util.py
├── visualize.py
├── yolov8n.pt
```

## Installation
Before running the project, install the required dependencies:
```sh
pip install -r requirements.txt
```

## Usage
### Running the Main Script
To detect and track vehicles and license plates from a video, run:
```sh
python main.py
```
This script processes the input video frame by frame, detects vehicles using YOLO, tracks their movement using SORT, and extracts the license plate information using OCR. The results are saved in a CSV file for further analysis.

### Visualizing the Results
To overlay the detected and tracked information on the video, run:
```sh
python visualize.py
```
This script reads the processed data from the CSV file and overlays bounding boxes, tracking IDs, and license plate numbers on the video frames for visualization.

## File Descriptions
- **main.py** - Detects and tracks vehicles and license plates from a video.
- **visualize.py** - Overlays detected and tracked information onto the video for easy interpretation.
- **sort.py** - Implements the **SORT (Simple Online and Realtime Tracking)** algorithm for tracking detected vehicles across frames.
- **util.py** - Contains utility functions such as reading license plates using OCR, writing results to CSV, and other helper functions.
- **add_missing_data.py** - Script for handling missing or incomplete data.
- **data/** - Directory containing training datasets and pre-trained models used for detection.
- **requirements.txt** - List of Python packages required to run the project smoothly.

## Detailed Workflow
1. **Vehicle Detection** - The YOLO model (**yolov8n.pt**) detects vehicles in each frame with high accuracy.
2. **Vehicle Tracking** - The **SORT algorithm** assigns unique IDs to detected vehicles and tracks them across frames.
3. **License Plate Detection** - A specialized YOLO model (**license_plate_detector.pt**) detects license plates from identified vehicles.
4. **License Plate Recognition** - OCR (Optical Character Recognition) extracts text from detected license plates for identification.
5. **Results Saving** - The detected and recognized data, including vehicle positions, IDs, and license plate numbers, are saved to a CSV file for further analysis.
6. **Visualization** - The processed data is overlayed onto the video for better interpretation, highlighting detected vehicles, tracking IDs, and license plate numbers.

## Example
To execute the full process:
```sh
python main.py
python visualize.py
```
This workflow enables the system to detect and track multiple vehicles simultaneously while maintaining accurate license plate recognition.

