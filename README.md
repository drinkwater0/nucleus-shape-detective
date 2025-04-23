
# Nucleus‑Shape Detective

Detects abnormal nuclear blebs in fluorescence‑microscopy images using a
lightweight convolutional neural network.

## Quick start
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python src/train.py  # train baseline model
streamlit run app/app.py  # launch demo
```

## Data
Place raw TIFF/PNG microscopy images in `data/raw/`. Add a file
`data/annotations/labels.csv` with two columns:
```
filename,label
img_0001.png,normal
img_0002.png,bleb
```

## Model
Baseline: ResNet‑18 (torchvision) fine‑tuned for 2 classes. Images are resized
to 224 × 224 and histogram‑equalised.


 ## requirements.txt 
 torch>=2.2
 torchvision
 pandas
 scikit-learn
 matplotlib
 albumentations
 streamlit
 pillow
 pytest
 
