from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
from PIL import Image
from torchvision import transforms
import io, base64
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from models_code.resnet50_finetune import get_resnet50
import os
import urllib.request

app = FastAPI()

# ✅ Allow frontend (React) requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Load your trained model
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
MODEL_PATH = "models/checkpoints/resnet50_best.pt"

model = get_resnet50(num_classes=2, pretrained=False)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval().to(device)
if not os.path.exists(MODEL_PATH):
    os.makedirs("models/checkpoints", exist_ok=True)
    print("Downloading model...")
    url = "https://drive.google.com/file/d/1qNCiqMLZ0A36h9m8Fy2TcoTLZQU6b6-h/view?usp=sharing"
    urllib.request.urlretrieve(url, MODEL_PATH)

# ✅ Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

@app.get("/")
def root():
    return {"message": "✅ Breast Cancer XAI API is running!"}


@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        pred = int(np.argmax(probs))

    return JSONResponse({
        "prediction": pred,
        "probabilities": probs.tolist()
    })


@app.post("/explain/image")
async def explain_image(file: UploadFile = File(...)):
    """Grad-CAM explanation endpoint — fixed for detach() issue."""
    print("📥 Received /explain/image request")
    try:
        contents = await file.read()
        if not contents:
            return {"error": "No file data received"}
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)

        model.eval()
        activations = {}
        gradients = {}

        # ✅ Attach forward & backward hooks
        def forward_hook(module, inp, out):
            activations["value"] = out

        def backward_hook(module, grad_in, grad_out):
            gradients["value"] = grad_out[0]

        target_layer = model.layer4[-1].conv3
        handle_fw = target_layer.register_forward_hook(forward_hook)
        handle_bw = target_layer.register_full_backward_hook(backward_hook)  # ✅ full backward hook

        # ✅ Forward + backward
        outputs = model(input_tensor)
        pred_class = outputs.argmax(dim=1)
        score = outputs[0, pred_class]
        model.zero_grad()
        score.backward()

        # ✅ Detach before converting to NumPy
        grads = gradients["value"].detach().cpu().numpy()[0]
        acts = activations["value"].detach().cpu().numpy()[0]

        weights = np.mean(grads, axis=(1, 2))
        cam = np.zeros(acts.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * acts[i, :, :]
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = cam / cam.max() if cam.max() != 0 else cam

        # ✅ Overlay heatmap on image
        heatmap = (plt.cm.jet(cam)[:, :, :3] * 255).astype(np.uint8)
        image_np = np.array(image.resize((224, 224)))
        overlay = cv2.addWeighted(image_np, 0.6, heatmap, 0.4, 0)

        # ✅ Convert overlay to base64
        _, buffer = cv2.imencode(".png", overlay)
        encoded = base64.b64encode(buffer).decode("utf-8")

        handle_fw.remove()
        handle_bw.remove()

        print("✅ Grad-CAM successfully generated.")
        return {"gradcam_image_base64": encoded}

    except Exception as e:
        print("❌ Grad-CAM generation failed:", e)
        return JSONResponse({"error": str(e)}, status_code=500)
