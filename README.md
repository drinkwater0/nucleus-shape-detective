# Nucleus‑Shape Detective

A deep learning tool for detecting abnormal nuclear blebs in fluorescence microscopy images. This project uses a fine-tuned ResNet-18 model to classify nucleus images as either normal or containing blebs, which are abnormal protrusions of the nuclear membrane.

## Features

- Real-time nucleus classification using a lightweight CNN
- User-friendly web interface with drag-and-drop functionality
- Support for various image formats (PNG, JPG, TIFF)
- Pre-trained model included
- Easy training pipeline for custom datasets

## Project Structure

```
nucleus-shape-detective/
├── app/                    # Web application
│   ├── app.py             # Streamlit frontend
│   └── requirements.txt   # App dependencies
├── data/                  # Data directory
│   ├── raw/              # Raw microscopy images
│   └── annotations/      # Training labels
├── src/                  # Source code
│   ├── train.py         # Training script
│   └── utils.py         # Utility functions
├── tests/               # Test suite
├── notebooks/          # Jupyter notebooks
├── model.pt           # Pre-trained model
└── requirements.txt   # Project dependencies
```

## Data

The project uses a CSV file to manage image labels and metadata. Create a file at `data/annotations/labels.csv` with the following columns:

```
filename,label,quality,flags
img_0001.png,0,1,
img_0002.png,1,2,foreign
img_0003.png,0,3,part
```

Where:
- `filename`: Path to the image file
- `label`: 0 for normal, 1 for bleb
- `quality`: Optional quality score (1-3)
- `flags`: Optional comma-separated flags (e.g., "foreign" for foreign objects, "part" for partial nuclei)

The dataset will automatically filter out low-quality images and those with specified flags.

Place your microscopy images in the appropriate directories:
- `data/normal/` for normal nucleus images
- `data/bleb/` for nucleus images with blebs

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/nucleus-shape-detective.git
cd nucleus-shape-detective
```

2. Create and activate a virtual environment:

For Windows:
```bash
python -m venv venv
.\venv\Scripts\activate
```

For Linux/Mac:
```bash
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:

For production use:
```bash
pip install -r requirements.txt
```

For development (includes testing and code quality tools):
```bash
pip install -r requirements-dev.txt
```

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

1. Prepare your data:
   - Place your microscopy images in `data/raw/`
   - Create a CSV file at `data/annotations/labels.csv` with columns:
     ```
     filename,label,quality,flags
     img_0001.png,0,1,
     img_0002.png,1,2,foreign
     img_0003.png,0,3,part
     ```

2. Train the model:
```bash
python src/train.py --img_dir data/raw --csv data/annotations/labels.csv --epochs 10
```

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

[Add your license information here]

## Citation

If you use this tool in your research, please cite:
[Add citation information]

## Acknowledgments

- [Add any acknowledgments here]

## Quick start
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python src/train.py  # train baseline model
streamlit run app/app.py  # launch demo
```

## Model
Baseline: ResNet‑18 (torchvision) fine‑tuned for 2 classes. Images are resized
to 224 × 224 and histogram‑equalised.
