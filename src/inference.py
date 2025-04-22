"""CLI helper to classify a single image."""
import torch, torchvision, sys
from PIL import Image
from torchvision import transforms as T

model = torchvision.models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("model.pt", map_location="cpu"))
model.eval()

preprocess = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

img = Image.open(sys.argv[1])
probs = torch.softmax(model(preprocess(img).unsqueeze(0)), dim=1)[0]
print({"normal": float(probs[0]), "bleb": float(probs[1])})
