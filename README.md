# 🚘 Number Plate Detection

This project aims to detect and extract vehicle number plates from images using image processing and machine learning techniques. It can be used for traffic monitoring, automated toll systems, and vehicle verification systems.

---

## 📌 Features

- Detects number plates from input images
- Uses image preprocessing (grayscale, thresholding, edge detection)
- Extracts the region of interest (ROI) containing the plate
- Displays and saves the detected plate section

---

## 📁 Folder Structure

```
number_plate_detection/
├── detect_plate.py         # Main detection script
├── utils.py                # Helper functions
├── images/                 # Input images
├── results/                # Output images with detected plates
├── model/ (optional)       # Pretrained model (if used)
├── requirements.txt        # Required packages
└── README.md               # Project documentation
```

##  How It Works

1. Load an input vehicle image
2. Preprocess the image (resize, convert to grayscale, blur, edge detection)
3. Detect contours that resemble number plate shapes
4. Extract and display/save the number plate region

---
