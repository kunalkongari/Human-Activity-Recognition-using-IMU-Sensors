# On-Device Human Activity Recognition
### TinyML · Wrist-Mounted IMU · Real-Time Inference on Microcontroller

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2%2B-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![MicroPython](https://img.shields.io/badge/MicroPython-TinyML-2B9348?style=flat-square&logo=micropython&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)
![Hardware](https://img.shields.io/badge/Hardware-Arduino%20Nicla%20Vision-00979D?style=flat-square&logo=arduino&logoColor=white)

---

A complete end-to-end **TinyML pipeline** for recognising 13 human activities in real time, running entirely on a microcontroller — no cloud, no connectivity, no latency.

Raw 6-axis IMU data is collected at the wrist using a **LSM6DSOX** sensor, processed through a **sliding-window feature engineering** stage, and classified by a **Decision Tree** model transpiled to pure MicroPython via `m2cgen`. The final system runs continuous inference at 20 Hz on an **Arduino Nicla Vision**.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Hardware & Sensor Setup](#hardware--sensor-setup)
3. [Activities Recognised](#activities-recognised)
4. [Repository Structure](#repository-structure)
5. [Quick Start](#quick-start)
6. [Data Collection](#data-collection)
7. [Feature Engineering Pipeline](#feature-engineering-pipeline)
8. [Model Development & Evaluation](#model-development--evaluation)
9. [On-Device Deployment](#on-device-deployment)
10. [Results](#results)
11. [Key Design Decisions](#key-design-decisions)
12. [Challenges & Observations](#challenges--observations)
13. [Dependencies](#dependencies)
14. [License](#license)

---

## Project Overview

Human Activity Recognition (HAR) is a classical problem in mobile sensing and health monitoring. Most production systems offload inference to the cloud, which introduces latency, privacy risks, and power costs.

This project takes the opposite approach: the entire inference pipeline — feature extraction and classification — runs **on-device** using MicroPython, with no external calls required. This makes the system viable for battery-powered wearables operating in bandwidth-constrained or offline environments.

The core idea is to keep the model architecture intentionally simple. A Decision Tree is an ideal TinyML candidate: it can be fully transpiled to a chain of `if/else` branches in pure Python (via `m2cgen`), uses no floating-point matrix operations, and occupies a tiny memory footprint — all critical constraints on a microcontroller with ~1 MB of RAM.

**Key design priorities:**
- Reproducible and interpretable model (Decision Tree)
- Consistent sliding-window approach across training and deployment
- Clean separation between data collection, feature engineering, and inference code
- Real hardware demonstration — not a simulation

---

## Hardware & Sensor Setup

| Component | Detail |
|-----------|--------|
| **Microcontroller** | Arduino Nicla Vision (STM32H747, Cortex-M7/M4) |
| **IMU Sensor** | LSM6DSOX (6-axis: accelerometer + gyroscope) |
| **Interface** | I2C (onboard, no chip-select pin needed) |
| **Sensor placement** | Wrist-mounted |
| **Sampling rate** | 50 Hz (data collection) / 20 Hz (deployment inference loop) |
| **LED feedback** | Green → sedentary; Red → active motion |

The sensor provides:
- **Accelerometer** axes: `ax`, `ay`, `az` (units: *g*, gravitational acceleration)
- **Gyroscope** axes: `gx`, `gy`, `gz` (units: *dps*, degrees per second)

---

## Activities Recognised

The model classifies 13 distinct daily-life activities:

| Index | Activity | Motion Type |
|-------|----------|-------------|
| 0 | Brisk Walking | Periodic, high-cadence |
| 1 | Cycling | Periodic, rotational |
| 2 | Eating with Spoon | Fine motor, repetitive |
| 3 | Jogging | High-impact, periodic |
| 4 | Jumping | High-amplitude, impulsive |
| 5 | Phone Interaction | Low-motion, finger tap/scroll |
| 6 | Pick and Place | Episodic reach-and-grasp |
| 7 | Sit–Stand–Sit | Postural transition |
| 8 | Sitting | Quasi-static |
| 9 | Stair Descent | Asymmetric periodic, heel-strike |
| 10 | Stair Ascent | Asymmetric periodic, toe-push |
| 11 | Standing | Near-static, balance sway |
| 12 | Walking | Periodic, moderate cadence |

---

## Repository Structure

```
On-Device Human Activity Recognition/
│
├── model_development.ipynb     # End-to-end training notebook
│                                 (data loading → feature engineering → training → evaluation → export)
│
├── src/
│   └── deployment.py           # MicroPython inference script (runs on-device)
│
├── data/
│   ├── Sitting_*.csv
│   ├── Standing_*.csv
│   ├── Walking_*.csv
│   ├── Brisk walking_*.csv
│   ├── Jogging_*.csv
│   ├── Cycling_*.csv
│   ├── Stair-Up_*.csv
│   ├── Stair-Down_*.csv
│   ├── Sit–Stand–Sit_*.csv
│   ├── Phone Interaction_*.csv
│   ├── Eating with Spoon_*.csv
│   ├── Pick and Place_*.csv
│   └── jumping_*.csv
│
├── docs/
│   └── report.pdf              # Full write-up with methodology and analysis
│
├── requirements.txt            # Python dependencies (PC / Colab)
└── LICENSE
```

> **Note on model file:** The trained model (`.pkl`) and the `m2cgen`-exported Python function are not committed to this repository. Running `model_development.ipynb` end-to-end regenerates both. The exported `score()` function must be pasted into `src/deployment.py` before flashing to hardware.
 


---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/on-device-har.git
cd on-device-har
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the training notebook

Open `model_development.ipynb` in Jupyter Lab or VS Code and execute all cells:

```bash
jupyter lab model_development.ipynb
```

This will:
1. Load the 13 raw IMU CSV files from `data/`
2. Apply sliding-window feature extraction
3. Train and evaluate the Decision Tree
4. Export the model as a MicroPython-compatible `score()` function

### 4. Deploy to hardware

1. Copy the generated `score()` function body from the notebook output
2. Paste it into the placeholder in `src/deployment.py`
3. Upload `deployment.py` to the Portenta H7 using OpenMV IDE or `mpremote`
4. Open a serial monitor — inference output will stream at 20 Hz

---

## Data Collection

**Sensor:** LSM6DSOX via SPI  
**Rate:** 50 Hz (one sample every 20 ms)  
**Duration per activity:** ~3 minutes (~9,000 samples)  
**Format:** One CSV file per activity

Each CSV has the following columns:

```
timestamp, ax, ay, az, gx, gy, gz, activity
```

Sample row:
```
1667319, -0.218872, -0.654907, -0.700318, 2.624511, 2.197266, 1.037598, Sitting
```

**Collection discipline matters.** Activities were performed in a controlled and natural manner — maintaining realistic motion patterns rather than exaggerated movements. For periodic activities (walking, jogging), pace was kept consistent throughout the session. Transitional activities (Sit–Stand–Sit) were executed with deliberate timing to ensure the transition was well-represented in the data.

---

## Feature Engineering Pipeline

The feature engineering stage bridges raw sensor streams and a fixed-size model input. It is applied **only during model development** — not during data collection.

### Sliding Window

| Parameter | Value |
|-----------|-------|
| Window length (N) | **20 samples** |
| Step / Overlap | **10 samples** (50% overlap) |
| Resulting window duration | 400 ms at 50 Hz |

A 50% overlap doubles the number of training windows and ensures that activity boundaries are better represented — transitional moments appear in both the window that starts before the transition and the one that starts after it.

### Features per Window

For each of the 6 IMU channels (`ax`, `ay`, `az`, `gx`, `gy`, `gz`), 10 time-domain statistical descriptors are extracted:

| # | Feature | Description |
|---|---------|-------------|
| 1 | Mean | Average signal level |
| 2 | Minimum | Lowest raw value in window |
| 3 | Maximum | Peak raw value in window |
| 4 | Standard Deviation | Spread of signal |
| 5 | Variance | Squared spread |
| 6 | RMS | Root Mean Square (energy proxy) |
| 7 | Energy | Sum of squared values |
| 8 | Median | Robust central tendency |
| 9 | IQR | Interquartile range (robust spread) |
| 10 | Zero-Crossing Rate | Oscillation frequency proxy |

**6 channels × 10 features = 60 statistical features**

Additionally, the 6 raw sensor values from the **most recent sample** in the window (the 20th sample) are appended to provide instantaneous context:

**Total feature vector: 60 + 6 = 66 features**

This matches exactly the input dimensionality expected by `deployment.py` on-device.

---

## Model Development & Evaluation

### Algorithm Choice: Decision Tree

A Decision Tree was selected as the classifier for reasons that are particularly relevant in TinyML contexts:

- **No matrix math at inference time** — the model is purely a sequence of scalar comparisons, which runs efficiently even without an FPU
- **Fully transpilable** — `m2cgen` converts the trained tree to a pure Python `if/else` chain, compatible with MicroPython
- **Interpretable** — feature importances provide direct feedback on which signals are most discriminative
- **Memory efficient** — the on-device representation is compact enough to fit in the Portenta H7's available RAM

### Hyperparameters

```python
DecisionTreeClassifier(
    max_depth=12,          # Limits overfitting; keeps tree compact for deployment
    min_samples_leaf=5,    # Prevents very small leaf nodes; improves generalisation
    random_state=42        # Reproducibility
)
```

`max_depth=12` is a deliberate constraint — deeper trees produce better training accuracy but consume more memory when transpiled. At depth 12, the exported `score()` function remains small enough to fit comfortably on the target hardware.

### Train/Test Split

```python
train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
```

An 80/20 split with stratification ensures that each activity class is proportionally represented in both partitions — important given that some activities (e.g., Sit–Stand–Sit) naturally produce fewer windows than quasi-static activities (e.g., Sitting).

### Evaluation Metrics

- Per-class **precision**, **recall**, and **F1-score** via `classification_report`
- **Confusion matrix** visualised as a heatmap (Seaborn, `plasma` colormap)
- **Top-20 feature importances** bar chart to understand model decisions

### Model Export for Deployment

```python
import m2cgen as m2c
code = m2c.export_to_python(clf_dt)
```

`m2cgen` transpiles the trained scikit-learn tree into a self-contained Python function. This function is the only piece of the ML model that runs on the microcontroller — no NumPy, no scikit-learn, no pickle loading.

---

## On-Device Deployment

`src/deployment.py` is the complete MicroPython script that runs continuously on the Portenta H7.

### Inference Loop

```
┌─────────────────────┐
│ Read IMU (ax,ay,az, │
│  gx,gy,gz) @ 20 Hz  │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Append to rolling  │
│  buffer (N=20)      │
└────────┬────────────┘
         │ buffer full?
         ▼
┌─────────────────────┐
│  Extract 66-feature │
│  vector             │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  score(features)    │  ← m2cgen-exported Decision Tree
│  → probability[13]  │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  argmax → activity  │
│  Set LED + print    │
└─────────────────────┘
```

### Key Configuration

```python
WINDOW_SIZE = 20        # Must match training exactly
SAMPLING_HZ = 20        # On-device inference rate
SLEEP_MS    = 50        # 1000 // SAMPLING_HZ
```

> ⚠️ The `WINDOW_SIZE` and the 10 features per axis **must be identical** between the training notebook and `deployment.py`. Any mismatch will produce silent dimension errors or incorrect predictions.

### LED Activity Feedback

| LED | Meaning |
|-----|---------|
| 🟢 Green | Sedentary activity detected (Sitting or Standing) |
| 🔴 Red | Active movement detected |
| 🔵 Blue (startup) | System initialisation |

---

## Results

The model was trained and evaluated on windowed features from all 13 activity classes.

**Overall test accuracy: see notebook output after running `model_development.ipynb`**

Notable observations from the confusion matrix:

- **High confidence classes:** Jogging, Jumping, and Cycling are well-separated due to their distinctive high-amplitude, periodic accelerometer signatures
- **Frequent confusions:** Sitting and Standing can be confused at the window level because both produce near-zero accelerometer variance; the gyroscope channels help distinguish them but are not always reliable
- **Transitional ambiguity:** Sit–Stand–Sit windows that capture the middle of the transition can be misclassified as either Walking or Standing, since the brief motion resembles the initial phase of both
- **Fine motor activities:** Eating with Spoon and Phone Interaction show moderate confusion with each other — both involve low-amplitude, high-frequency wrist micromovements

**Top predictive features:** Accelerometer `az` (vertical axis) standard deviation and RMS consistently ranked among the top 5 features, followed by gyroscope variance across all three axes. This matches intuition — vertical body loading during gait and rotational wrist dynamics during activities like eating are the most discriminative signals.

---

## Key Design Decisions

**Why a Decision Tree over a neural network?**  
Neural networks require matrix multiplication at inference time, which demands either an FPU or a software math library — both expensive on small microcontrollers. A Decision Tree transpiled by `m2cgen` is a pure conditional chain: fast, predictable, and memory-efficient with zero external dependencies.

**Why 50% window overlap?**  
A 50% overlap doubles the training dataset size without collecting new data, and ensures that activity boundaries — where the signal is most informative — appear in multiple windows and are not systematically missed.

**Why 10 features per axis?**  
The feature set covers four distinct signal properties: central tendency (mean, median), spread (std, variance, IQR), energy (RMS, energy), and temporal dynamics (min, max, ZCR). This provides a reasonably complete description of each window's statistical character without requiring frequency-domain transforms, which would be expensive to compute on-device.

**Why `max_depth=12` and `min_samples_leaf=5`?**  
These constraints trade a small amount of training accuracy for two benefits: reduced overfitting to the specific data collection session, and a smaller on-device footprint when the tree is transpiled. A deeper tree exports to a larger `score()` function, which may exceed available program memory.

---

## Challenges & Observations

**Intra-class variability from wrist placement**  
Small differences in wrist angle during collection sessions introduce sensor offset drift across CSV files for the same activity. This manifests as shifted mean values on the accelerometer axes and was partially mitigated by including both mean and variance features, which capture level and spread separately.

**Phone Interaction and Eating with Spoon overlap**  
Both activities involve low-amplitude repetitive wrist motion. At a 400 ms window size, the signal signatures are sometimes indistinguishable. Longer windows or frequency-domain features (e.g., dominant FFT bin) would help, but both would increase model size and deployment cost.

**Sampling rate discipline**  
Maintaining a consistent 50 Hz during collection required attention — any task that blocked the main loop (e.g., SD card writes during data logging) caused gaps in the time series. These gaps were cleaned in the notebook before feature extraction.

**Transitional activities are sparse**  
A Sit–Stand–Sit cycle takes roughly 10 seconds; 3 minutes of data yields only ~18 complete transitions. After windowing, this class had significantly fewer samples than quasi-static activities like Sitting. Stratified splitting mitigated class imbalance in evaluation, but the model likely underperforms on this class in edge cases.

---

## Dependencies

### PC / Training Environment

```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
joblib>=1.2.0
m2cgen>=0.10.0
matplotlib>=3.6.0
seaborn>=0.12.0
ipykernel>=6.0.0
```

Install with:
```bash
pip install -r requirements.txt
```

### On-Device (MicroPython)

No external packages required. The deployment script uses only:
- `machine` (I2C, Pin, LED — MicroPython standard library)
- `lsm6dsox` (bundled MicroPython driver for the LSM6DSOX sensor)
- `math` (MicroPython built-in)
- `time` (MicroPython built-in)

The `score()` function generated by `m2cgen` is pure Python with no imports.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for full terms.

---

<div align="center">

Built with 🎯 precision, deployed with ⚡ efficiency.

*Wrist-worn · Edge-native · Zero-cloud*

</div>
