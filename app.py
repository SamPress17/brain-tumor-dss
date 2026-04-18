"""
Brain Tumor DSS — Flask Backend
Endpoints:
    GET  /health
    POST /predict   → classification + confidence + severity + GradCAM heatmap
"""

import io
import os
import json
import base64
import datetime
from pathlib import Path
from functools import wraps

import numpy as np
import torch
import torch.nn as nn
import cv2
from torchvision import models, transforms
from PIL import Image
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle, Image as RLImage

# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────

CHECKPOINT  = "best_model.pth"
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
IMAGE_SIZE  = 224
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

SEVERITY = {
    "glioma": {
        "level":       "Critical",
        "score":       5,
        "color":       "#e74c3c",
        "description": "Malignant brain tumor originating from glial cells. Most aggressive type.",
        "action":      "Immediate neurosurgery and oncology referral required.",
        "urgency":     "URGENT",
    },
    "meningioma": {
        "level":       "Moderate",
        "score":       3,
        "color":       "#f39c12",
        "description": "Usually benign tumor arising from the meninges. Slow-growing.",
        "action":      "Neurology consultation and follow-up MRI advised.",
        "urgency":     "MONITOR",
    },
    "pituitary": {
        "level":       "Moderate",
        "score":       3,
        "color":       "#f39c12",
        "description": "Tumor in the pituitary gland. Often benign but affects hormones.",
        "action":      "Endocrinology and neurosurgery consultation recommended.",
        "urgency":     "MONITOR",
    },
    "notumor": {
        "level":       "Normal",
        "score":       1,
        "color":       "#2ecc71",
        "description": "No tumor detected in the MRI scan.",
        "action":      "No immediate action required. Routine follow-up as indicated.",
        "urgency":     "ROUTINE",
    },
}

AUTH_USERS = {"doctor": "brain123"}
AUTH_TOKEN = "brain-tumor-dss-key-2026"
HISTORY_FILE = Path("scan_history.json")

# ─────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────

class BrainTumorResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        backbone      = models.resnet50(weights=None)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def load_model():
    m = BrainTumorResNet(len(CLASS_NAMES)).to(DEVICE)
    m.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    m.eval()
    print(f"[BACKEND] Model loaded on {DEVICE}")
    return m


model = load_model()

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ─────────────────────────────────────────────────────────
# GRAD-CAM
# ─────────────────────────────────────────────────────────

class GradCAMPlusPlus:
    """
    Grad-CAM++ using the last convolutional layer of ResNet-50 (layer4[-1]).
    """
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None

        target_layer = list(model.features.children())[7][-1].conv3
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, tensor, class_idx):
        self.model.zero_grad()
        output = self.model(tensor)
        score = output[0, class_idx]
        score.backward(retain_graph=True)

        gradients = self.gradients
        activations = self.activations

        grads_pow_2 = gradients.pow(2)
        grads_pow_3 = gradients.pow(3)
        alpha_num = grads_pow_2
        alpha_denom = 2 * grads_pow_2 + (activations * grads_pow_3).sum(dim=(2, 3), keepdim=True)
        alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))
        alpha = alpha_num / (alpha_denom + 1e-8)

        weights = (alpha * torch.relu(gradients)).sum(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activations).sum(dim=1).squeeze())
        cam = cam.cpu().numpy()

        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def apply_heatmap(original_pil: Image.Image, cam: np.ndarray) -> str:
    orig = np.array(original_pil.resize((IMAGE_SIZE, IMAGE_SIZE)))
    cam_resized = cv2.resize(cam, (IMAGE_SIZE, IMAGE_SIZE))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    blended = (0.55 * orig + 0.45 * heatmap).astype(np.uint8)
    return pil_to_b64(Image.fromarray(blended))


def apply_segmentation(original_pil: Image.Image, cam: np.ndarray):
    orig = np.array(original_pil.resize((IMAGE_SIZE, IMAGE_SIZE)))
    cam_resized = cv2.resize(cam, (IMAGE_SIZE, IMAGE_SIZE))
    
    # Lower threshold to 0.15 for better tumor detection
    mask = (cam_resized >= 0.15).astype(np.uint8) * 255
    
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Close small holes
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # Open to remove noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)
    
    # Create red overlay with better visibility (70% red, 30% original)
    overlay = orig.copy()
    overlay[mask == 255] = (overlay[mask == 255] * 0.3 + np.array([255, 0, 0]) * 0.7).astype(np.uint8)
    
    # Add contour outline for better definition
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 100, 100), 2)
    
    seg_mask = Image.fromarray(mask).convert("RGB")
    overlay_img = Image.fromarray(overlay)
    area_pct = round(np.count_nonzero(mask) * 100 / (IMAGE_SIZE * IMAGE_SIZE), 2)
    return pil_to_b64(overlay_img), pil_to_b64(seg_mask), area_pct


gradcam = GradCAMPlusPlus(model)

# ─────────────────────────────────────────────────────────
# FLASK
# ─────────────────────────────────────────────────────────

app = Flask(__name__)
CORS(app)


def require_auth(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        token = request.headers.get("X-Api-Key", "")
        if token != AUTH_TOKEN:
            return jsonify({"error": "Unauthorized"}), 401
        return fn(*args, **kwargs)
    return wrapper


def load_history():
    if HISTORY_FILE.exists():
        try:
            return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def save_history(entries):
    HISTORY_FILE.write_text(json.dumps(entries, indent=2), encoding="utf-8")


def append_history(entry):
    history = load_history()
    history.insert(0, entry)
    save_history(history)


def pil_to_b64(pil_img: Image.Image, fmt="PNG") -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def b64_to_rl_image(b64str, width=220):
    img_data = io.BytesIO(base64.b64decode(b64str))
    return RLImage(img_data, width=width)


def create_pdf_report(data):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )
    styles = getSampleStyleSheet()
    story = []

    title_style = styles["Title"]
    title_style.textColor = colors.HexColor("#0f172a")
    story.append(Paragraph("Brain Tumor DSS — Diagnostic Report", title_style))
    story.append(Spacer(1, 12))

    patient = data.get("patient", {})
    patient_rows = [
        ["Patient ID", patient.get("patient_id", "—")],
        ["Age / Sex", patient.get("age_sex", "—")],
        ["Scan type", patient.get("scan_type", "—")],
        ["Referred by", patient.get("referred_by", "—")],
        ["Timestamp", data.get("timestamp", "")],
    ]
    patient_table = Table(patient_rows, colWidths=[120, 320])
    patient_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E6F1FB")),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#0f172a")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(patient_table)
    story.append(Spacer(1, 16))

    story.append(Paragraph("Prediction summary", styles["Heading2"]))
    summary_rows = [
        ["Prediction", data.get("prediction", "N/A")],
        ["Confidence", f"{data.get('confidence', 0)}%"],
        ["Severity", data.get("severity", {}).get("level", "N/A")],
        ["Urgency", data.get("severity", {}).get("urgency", "N/A")],
        ["Tumor coverage", f"{data.get('segmentation_area_pct', 0)}%"],
    ]
    summary_table = Table(summary_rows, colWidths=[120, 320])
    summary_table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2ff")),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#0f172a")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 16))

    story.append(Paragraph("Probability distribution", styles["Heading2"]))
    prob_rows = [["Class", "Probability"]]
    for cls, pct in data.get("probabilities", {}).items():
        prob_rows.append([cls.capitalize(), f"{pct}%"])
    prob_table = Table(prob_rows, colWidths=[160, 280])
    prob_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f8fafc")),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#0f172a")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(prob_table)
    story.append(Spacer(1, 16))

    if data.get("original_b64"):
        story.append(Paragraph("Original scan", styles["Heading2"]))
        story.append(b64_to_rl_image(data["original_b64"], width=240))
        story.append(Spacer(1, 12))

    if data.get("gradcam_b64"):
        story.append(Paragraph("Grad-CAM++ attention map", styles["Heading2"]))
        story.append(b64_to_rl_image(data["gradcam_b64"], width=240))
        story.append(Spacer(1, 12))

    if data.get("segmentation_overlay_b64"):
        story.append(Paragraph("Tumor segmentation overlay", styles["Heading2"]))
        story.append(b64_to_rl_image(data["segmentation_overlay_b64"], width=240))
        story.append(Spacer(1, 12))

    story.append(Paragraph("Generated by Brain Tumor DSS", styles["Normal"]))
    doc.build(story)
    buffer.seek(0)
    return buffer.read()


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "device": DEVICE})


@app.route("/login", methods=["POST"])
def login():
    payload = request.get_json(force=True, silent=True) or {}
    username = payload.get("username", "")
    password = payload.get("password", "")
    if AUTH_USERS.get(username) == password:
        return jsonify({"token": AUTH_TOKEN, "user": username})
    return jsonify({"error": "Invalid credentials"}), 401


@app.route("/history", methods=["GET"])
@require_auth
def history():
    return jsonify({"history": load_history()})


@app.route("/report", methods=["POST"])
@require_auth
def report():
    payload = request.get_json(force=True, silent=True) or {}
    if not payload:
        return jsonify({"error": "No report payload received."}), 400
    pdf_bytes = create_pdf_report(payload)
    return Response(pdf_bytes, mimetype="application/pdf")


@app.route("/predict", methods=["POST"])
@require_auth
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Use field name 'file'."}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename."}), 400

    metadata_json = request.form.get("metadata", "{}")
    try:
        metadata = json.loads(metadata_json)
    except Exception:
        metadata = {}

    try:
        img_bytes = file.read()
        pil_img   = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception:
        return jsonify({"error": "Cannot read image. Use JPG or PNG."}), 400

    tensor = transform(pil_img).unsqueeze(0).to(DEVICE)
    tensor.requires_grad_(True)

    # ── Forward pass ──
    with torch.enable_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1).squeeze().detach().cpu().tolist()

    top_idx   = int(np.argmax(probs))
    top_class = CLASS_NAMES[top_idx]
    top_prob  = probs[top_idx]

    # ── GradCAM ++ and segmentation ──
    cam                     = gradcam.generate(tensor, top_idx)
    heatmap_b64             = apply_heatmap(pil_img, cam)
    segmentation_overlay_b64, segmentation_mask_b64, segmentation_area_pct = apply_segmentation(pil_img, cam)

    # ── Original image base64 ──
    orig_buf = io.BytesIO()
    pil_img.resize((IMAGE_SIZE, IMAGE_SIZE)).save(orig_buf, format="PNG")
    original_b64 = base64.b64encode(orig_buf.getvalue()).decode("utf-8")
    timestamp = datetime.datetime.utcnow().isoformat() + "Z"

    patient_info = metadata.get("patient", {}) if isinstance(metadata, dict) else {}
    record = {
        "timestamp": timestamp,
        "patient": {
            "patient_id": patient_info.get("patient_id", "—"),
            "age_sex": patient_info.get("age_sex", "—"),
            "scan_type": patient_info.get("scan_type", "—"),
            "referred_by": patient_info.get("referred_by", "—"),
        },
        "prediction": top_class,
        "confidence": round(top_prob * 100, 2),
        "probabilities": {c: round(p * 100, 2) for c, p in zip(CLASS_NAMES, probs)},
        "severity": SEVERITY[top_class],
        "segmentation_area_pct": segmentation_area_pct,
    }
    append_history(record)

    return jsonify({
        "prediction":    top_class,
        "confidence":    round(top_prob * 100, 2),
        "probabilities": {c: round(p * 100, 2) for c, p in zip(CLASS_NAMES, probs)},
        "severity":      SEVERITY[top_class],
        "gradcam":       heatmap_b64,
        "original":      original_b64,
        "segmentation_overlay": segmentation_overlay_b64,
        "segmentation_mask": segmentation_mask_b64,
        "segmentation_area_pct": segmentation_area_pct,
        "timestamp": timestamp,
    })


if __name__ == "__main__":
    print(f"[BACKEND] Starting on http://localhost:5000  [device={DEVICE}]")
    app.run(host="0.0.0.0", port=5000, debug=False)
