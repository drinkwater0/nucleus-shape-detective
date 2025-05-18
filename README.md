# Nucleus‑Shape Detective

A deep learning tool for detecting abnormal nuclear blebs in fluorescence microscopy images. This project uses a fine-tuned ResNet-18 model to classify nucleus images as either normal or containing blebs, which are abnormal protrusions of the nuclear membrane.

## Features

- Real-time nucleus classification using a lightweight CNN
- User-friendly web interface with drag-and-drop functionality
- Support for various image formats (PNG, JPG, TIFF)
- Pre-trained model included
- Example dataset included for immediate testing
- Easy training pipeline for custom datasets

## Project Structure

```
nucleus-shape-detective/
├── app/                    # Web application
│   ├── app.py             # Streamlit frontend
│   └── requirements.txt   # App dependencies
├── data/                  # Data directory
│   ├── bleb/              # Images of healthy nucleus
│   ├── normal/            # Images of blebbed nucleus
│   ├── annotations/       # Training labels
│   │   └── labels.csv     # Labeling of the images used for training 
│   ├── zdroje.xlsx        # Excel table with data about training images - source, highlighted protein, quality of the training image
│   ├── normalSource.csv   # CSV file with data only of the healthy nucleus from zdroje.xlsx; needed for labeling images for training of the model
│   └── mutantsSource.csv  # CSV file with data only of the blebbed nucleus from zdroje.xlsx; needed for labeling images for training of the model 
├── src/                   # Source code
│   ├── train.py           # Main training script
│   │                    # Usage: python train.py [--img_dir DATA_DIR] [--csv CSV_PATH] [--epochs N] [--lr LEARNING_RATE]
│   │                    # Trains a ResNet-18 model with data augmentation and early stopping
│   │
│   ├── evaluate.py        # Model evaluation and visualization tool
│   │                    # Usage: python evaluate.py [--model_path MODEL_PATH] [--data_dir DATA_DIR] [--csv_path CSV_PATH]
│   │                    # Generates classification report and confusion matrix visualization
│   │
│   ├── inference.py       # Command-line inference tool
│   │                    # Usage: python inference.py IMAGE_PATH
│   │                    # Classifies a single image and outputs probabilities
│   │
│   ├── prepare_labels.py  # Data preparation script
│   │                    # Usage: python prepare_labels.py
│   │                    # Combines normalSource.csv and mutantsSource.csv into labels.csv
│   │                    # Handles image paths, labels, quality scores, and flags
│   │
│   ├── utils.py           # Core utilities and shared functionality
│   │                    # Contains:
│   │                    # - NucleusDataset: Custom PyTorch dataset for nucleus images
│   │                    # - evaluate: Basic evaluation function for training
│   │                    # - train_loop: Training loop with early stopping
│   │
│   └── __init__.py       # Package initialization file
├── tests/               # Test suite
├── notebooks/          # Jupyter notebooks
├── model.pt           # Pre-trained model
└── requirements.txt   # Project dependencies
```

## Data

The project uses several data files to manage image labels and metadata:

1. `data/annotations/labels.csv`: Contains the training labels with the following columns:
```
filename,label,quality,flags
img_0001.png,0,1,
img_0002.png,1,2,foreign
img_0003.png,0,3,part
```

2. `data/zdroje.xlsx`: Excel table containing comprehensive information about training images:
   - Source of the images
   - Highlighted protein information
   - Quality assessment of training images

3. Additional CSV files for specific nucleus types:
   - `data/normalSource.csv`: Contains data for healthy nucleus images
   - `data/mutantsSource.csv`: Contains data for blebbed nucleus images

The dataset structure is organized as follows:
- `data/normal/`: Directory containing images of healthy nucleus
- `data/bleb/`: Directory containing images of nucleus with blebs
- `data/annotations/`: Directory containing training labels and metadata

Place your microscopy images in the appropriate directories:
- `data/normal/` for normal nucleus images
- `data/bleb/` for nucleus images with blebs

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)

## Installation

### Step 1: Install Python
1. Download Python from [python.org](https://www.python.org/downloads/)
2. During installation, make sure to check "Add Python to PATH"
3. To verify installation, open Command Prompt (Windows) or Terminal (Mac/Linux) and type:
   ```
   python --version
   ```
   You should see something like "Python 3.8.0" or higher

### Step 2: Download the Project
1. Click the green "Code" button on this page
2. Click "Download ZIP"
3. Extract the ZIP file to a location of your choice
4. Open Command Prompt (Windows) or Terminal (Mac/Linux)
5. Navigate to the extracted folder:
   ```
   cd path/to/nucleus-shape-detective
   ```

### Step 3: Set Up the Environment
1. Create a virtual environment (this keeps the project's dependencies separate):
   ```
   python -m venv venv
   ```

2. Activate the environment:
   - On Windows:
     ```
     .\venv\Scripts\activate
     ```
   - On Mac/Linux:
     ```
     source venv/bin/activate
     ```
   You'll know it's activated when you see `(venv)` at the start of your command line

3. Install required packages:
   ```
   pip install -r requirements.txt
   ```
   This might take a few minutes. Wait until it completes.

### Step 4: Run the Application
1. Start the web interface:
   ```
   streamlit run app/app.py
   ```
2. Your default web browser should open automatically with the application
3. If it doesn't open automatically, copy and paste the URL shown in the terminal into your browser

### Troubleshooting
- If you get a "command not found" error, make sure Python is added to your PATH
- If you get permission errors, try running the terminal as administrator
- If the web interface doesn't open, make sure no other application is using port 8501


## Usage

### Running the Web Interface

Launch the Streamlit app:
```bash
streamlit run app/app.py
```

The web interface will open in your default browser. You can:
1. Upload nucleus images using the file uploader
2. View the classification results in real-time
3. See the confidence score for each prediction

### Training Your Own Model

#### Step 1: Prepare Your Data
1. Place your microscopy images in the correct folders:
   - Put normal nucleus images in `data/normal/`
   - Put blebbed nucleus images in `data/bleb/`

2. Prepare your source data:
   - Open `data/zdroje.xlsx` in Excel
   - Fill in the information for each image:
     - Source of the image
     - Highlighted protein
     - Quality score (1-10)
     - Any flags (e.g., "foreign" for foreign objects, "part" for partial nuclei)

3. Create source CSV files:
   - Export data for normal nuclei to `data/normalSource.csv`
   - Export data for blebbed nuclei to `data/mutantsSource.csv`
   - Make sure to use semicolon (;) as separator
   - Include these columns: id, url, article, protein, blank, quality, flags, address

4. Generate training labels:
   ```
   python src/prepare_labels.py
   ```
   This will create `data/annotations/labels.csv` automatically

#### Step 2: Train the Model
1. Make sure your virtual environment is activated (you should see `(venv)` in your terminal)

2. Start the training:
   ```
   python src/train.py
   ```
   This will:
   - Load your images and labels
   - Train a ResNet-18 model
   - Save the trained model as `model.pt`
   - Show training progress in the terminal

3. Optional: Customize training parameters:
   ```
   python src/train.py --epochs 20 --lr 0.0003
   ```
   - `--epochs`: Number of training cycles (default: 20)
   - `--lr`: Learning rate (default: 0.0003)

#### Step 3: Evaluate the Model
1. Check model performance:
   ```
   python src/evaluate.py
   ```
   This will:
   - Generate a classification report
   - Create a confusion matrix visualization (`confusion_matrix.png`)
   - Show accuracy metrics

#### Step 4: Use the Model
1. Start the web interface:
   ```
   streamlit run app/app.py
   ```
2. Upload your images to test the model
3. View the classification results and confidence scores

### Training Tips
- Start with a small dataset to test the pipeline
- Use high-quality images (quality score ≥ 7)
- Exclude images with foreign objects or partial nuclei
- If training is slow, consider using a GPU
- If results are poor, try:
  - Increasing the number of epochs
  - Adjusting the learning rate
  - Adding more training images
  - Improving image quality

### Expected Training Time
- CPU: ~1-2 hours for 1000 images
- GPU: ~15-30 minutes for 1000 images
- Progress is shown in the terminal during training

## Model Details

- Architecture: ResNet-18 (pre-trained on ImageNet)
- Input: 224×224 RGB images
- Output: Binary classification (normal/bleb)
- Preprocessing:
  - Resize to 224×224
  - RGB conversion (if needed)
  - Normalization using ImageNet statistics

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is unlicensed. You may use this code for research purposes only.





## Quick Start

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/app.py

# Mac/Linux
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app/app.py
```

## Model
Baseline: ResNet‑18 (torchvision) fine‑tuned for 2 classes. Images are resized
to 224 × 224 and histogram‑equalised.
