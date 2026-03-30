# 🏃 On-Device Human Activity Recognition (TinyML)

> Real-time human activity recognition running entirely on a microcontroller — no cloud, no latency, no privacy concerns.

[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)](https://python.org)
[![MicroPython](https://img.shields.io/badge/MicroPython-1.20+-green?logo=micropython)](https://micropython.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?logo=scikit-learn)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Recognized Activities](#recognized-activities)
- [System Architecture](#system-architecture)
- [Hardware Requirements](#hardware-requirements)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [1. Model Development (PC / Colab)](#1-model-development-pc--colab)
  - [2. On-Device Deployment (Microcontroller)](#2-on-device-deployment-microcontroller)
- [Feature Engineering](#feature-engineering)
- [Model Details](#model-details)
- [LED Feedback](#led-feedback)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This project implements a complete **TinyML pipeline** for human activity recognition (HAR):

1. **Data collection** — raw IMU data (accelerometer + gyroscope) recorded at 20 Hz
2. **Feature engineering** — statistical time-domain features extracted from sliding windows
3. **Model training** — a memory-efficient Decision Tree trained with scikit-learn
4. **Edge deployment** — the model is transpiled to pure MicroPython using [m2cgen](https://github.com/BayesWitnesses/m2cgen) and runs entirely on-device

The inference pipeline occupies less than **50 KB of RAM**, making it suitable for resource-constrained boards like the Arduino Portenta H7.

---

## Recognized Activities

The classifier distinguishes **13 activity classes** from a 6-axis IMU:

| Index | Activity            | Index | Activity           |
|------:|---------------------|------:|--------------------|
| 0     | Brisk Walking       | 7     | Sit–Stand–Sit      |
| 1     | Cycling             | 8     | Sitting            |
| 2     | Eating with Spoon   | 9     | Stair Descent      |
| 3     | Jogging             | 10    | Stair Ascent       |
| 4     | Jumping             | 11    | Standing           |
| 5     | Phone Interaction   | 12    | Walking            |
| 6     | Pick and Place      |       |                    |

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        DEVELOPMENT (PC)                          │
│                                                                  │
│  Raw CSV Data ──► Feature Engineering ──► Decision Tree ──► .pkl │
│                   (sliding window,          (scikit-learn)        │
│                    10 stats × 6 axes)                            │
│                                                ▼                 │
│                                           m2cgen                 │
│                                                ▼                 │
│                                      model_logic.py              │
│                                    (pure Python/MicroPython)     │
└──────────────────────────────────────────────────────────────────┘
                                │
                          Flash to board
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│                    ON-DEVICE (Microcontroller)                   │
│                                                                  │
│  LSM6DSOX ──► Sliding Window ──► Feature Vector ──► score()     │
│  (IMU @ 20 Hz)  (20 samples)      (66 features)    (inference)  │
│                                                       ▼          │
│                                                 LED Feedback     │
│                                              + Serial Output     │
└──────────────────────────────────────────────────────────────────┘
```

---

## Hardware Requirements

| Component | Description |
|-----------|-------------|
| **Board** | Arduino Portenta H7 (or any STM32-based board with MicroPython support) |
| **IMU** | LSM6DSOX (6-axis: accelerometer + gyroscope) via SPI |
| **LEDs** | Red / Green / Blue (onboard or external) |
| **Interface** | USB Serial for monitoring output |

> **Note:** The `lsm6dsox` MicroPython library is required on the board. See [OpenMV's lsm6dsox driver](https://github.com/openmv/openmv/blob/master/scripts/libraries/lsm6dsox.py).

---

## Project Structure

```
har-tinyml/
├── src/
│   └── model_deployment.py          # MicroPython inference script (flash this to the board)
├── data/
│   └── 13 activity csv                 
├── docs/
│   └── report.pdf             
├── model_development.ipynb    
├── requirements.txt          
├── .gitignore
└── README.md
```

---

## Getting Started

### 1. Model Development (PC / Colab)

#### Install dependencies

```bash
pip install -r requirements.txt
```

#### Prepare your dataset

Place your activity CSV files inside `data/`. Each file should be named after the activity class (e.g. `Walking.csv`) and contain columns:

```
ax, ay, az, gx, gy, gz
```

#### Run the notebook

Open `model_development.ipynb` in Jupyter or Google Colab and run all cells. The notebook will:

- Load and preprocess all activity CSVs
- Extract statistical features using a sliding window
- Train a Decision Tree classifier
- Evaluate performance (accuracy, confusion matrix, classification report)
- Export the model to MicroPython-compatible code using `m2cgen`

After running, copy the exported `score()` function into `src/deployment.py` (it is already pre-populated with a trained version).

---

### 2. On-Device Deployment (Microcontroller)

#### Flash MicroPython

Ensure your board is running a recent MicroPython firmware. For Portenta H7, follow [this guide](https://docs.arduino.cc/tutorials/portenta-h7/micropython-installation).

#### Copy files to the board

Using a tool like [mpremote](https://github.com/micropython/micropython/blob/master/tools/mpremote/README.md) or Thonny IDE:

```bash
mpremote cp src/deployment.py :main.py
```

#### Monitor output

```bash
mpremote connect /dev/ttyUSB0   # adjust port as needed
```

You should see output like:

```
========================================
  HAR TinyML — System Ready
  Target: 20 Hz | Window: 20 samples
========================================
Activity: Walking              | Confidence: 88% | Class: 12
Activity: Standing             | Confidence: 95% | Class: 11
Activity: Jogging              | Confidence: 91% | Class: 3
```

---

## Feature Engineering

For each 20-sample window, **10 time-domain statistics** are computed per sensor axis (6 axes × 10 = 60 features). The 6 instantaneous sensor values from the latest sample are appended, giving a **66-element feature vector**.

| Feature | Description |
|---------|-------------|
| Mean | Average value |
| Min / Max | Range extremes |
| Std | Standard deviation |
| Variance | Spread of values |
| RMS | Root-mean-square — signal power |
| Energy | Sum of squared values |
| Median | 50th percentile |
| IQR | Interquartile range (75th – 25th percentile) |
| ZCR | Zero-crossing rate relative to mean |

---

## Model Details

| Parameter | Value |
|-----------|-------|
| Algorithm | Decision Tree (scikit-learn) |
| Max depth | 12 |
| Min samples per leaf | 5 |
| Feature vector size | 66 |
| Number of classes | 13 |
| Transpiler | m2cgen (→ pure Python) |
| RAM footprint (inference) | < 50 KB |

The depth and leaf-size constraints prevent overfitting while keeping the model small enough for microcontroller deployment.

---

## LED Feedback

| LED Color | Meaning |
|-----------|---------|
| 🟢 Green | Sedentary activity detected (Sitting or Standing) |
| 🔴 Red | Active movement detected (all other classes) |
| 🔵 Blue (blink) | System startup |

---

## Results

See [`docs/report.pdf`](docs/report.pdf) for the full evaluation including:

- Per-class precision, recall and F1-score
- Confusion matrix heatmap
- Feature importance analysis
- Memory and latency benchmarks

---

## Contributing

Contributions are welcome! To add a new activity class:

1. Collect IMU data at 20 Hz and save it as `<ActivityName>.csv` in `data/sample/`
2. Re-run `model_development.ipynb` to retrain
3. Copy the new `score()` function output into `src/deployment.py`
4. Update `ACTIVITY_MAP` in `src/deployment.py`

Please open an issue before submitting large changes.

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
