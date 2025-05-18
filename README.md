# Nucleus‑Shape Detective

A deep learning tool for detecting abnormal nuclear blebs in fluorescence microscopy images. This project uses a fine-tuned ResNet-18 model to classify nucleus images as either normal or containing blebs, which are abnormal protrusions of the nuclear membrane.

## Features

- Real-time nucleus classification using a lightweight CNN
- User-friendly web interface with drag-and-drop functionality
- Support for various image formats (PNG, JPG, TIFF)
- Pre-trained model included
- Example dataset included for immediate testing
- Easy training pipeline for custom datasets

## Usage 

You can use latest trained model with web interface. It's hosted online - https://nucleusdetective.streamlit.app 

Or you can download this code and train your own model with your data.

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)

### Step-by-Step Installation
1. Download Python from [python.org](https://www.python.org/downloads/)
2. Download this project (click "Code" → "Download ZIP")
3. Extract the ZIP file
4. Open terminal/command prompt in the extracted folder
5. Run the Quick Start commands above

### Training Your Own Model

#### Step 1: Prepare Your Data
1. Place your images in the correct folders:
   - Normal nuclei → `data/normal/`
   - Blebbed nuclei → `data/bleb/`

2. Prepare your source data in `data/zdroje.xlsx`:
   - Source of the image
   - Highlighted protein
   - Quality score (1-10)
   - Any flags (e.g., "foreign", "part")

3. Generate training labels:
   ```bash
   python src/prepare_labels.py
   ```

#### Step 2: Train the Model
```bash
python src/train.py
```

Optional parameters:
- `--epochs`: Number of training cycles (default: 20)
- `--lr`: Learning rate (default: 0.0003)

#### Step 3: Evaluate the Model
```bash
python src/evaluate.py
```

### Web Interface
1. Start the application using the Quick Start commands
2. Upload nucleus images using the file uploader
3. View classification results in real-time
4. See confidence scores for each prediction


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

This project is unlicensed. You may use this code as you wish.
