# MelanoDetect

MelanoDetect is a deep learning-based skin lesion classification system that predicts whether a skin lesion image is benign or malignant and provides Grad-CAM visual explanations.

## Project Overview
This project uses the HAM10000 dataset and transfer learning with EfficientNetB0 to support skin cancer screening through a web-based interface.

## Features
- Binary classification: benign vs malignant
- Grad-CAM explainability
- Flask web application
- Threshold tuning for improved malignant detection

## Technologies Used
- Python
- TensorFlow / Keras
- EfficientNetB0
- Flask
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

## Dataset
The project uses the HAM10000 dataset. The dataset is not included in this repository because of file size limitations.

## Best Model
## Best Model
- Model: EfficientNetB0 fine-tuned with 260×260 input
- Highest test accuracy: 85.7%
- Final threshold: 0.50

## Repository Structure
- `app/` - Flask application
- `notebooks/` - training and experimentation notebooks
- `results/` - graphs and screenshots
- `docs/` - project notes and summaries

## How to Run
1. Clone the repository
2. Install dependencies with `pip install -r requirements.txt`
3. Add the trained model file locally
4. Run the Flask app

## Note
This project is intended for academic and research purposes only and is not a clinical diagnostic tool.