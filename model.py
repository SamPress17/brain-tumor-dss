"""
Brain Tumor Decision Support System
Model Training — ResNet-50
Classes: Glioma | Meningioma | Pituitary | No Tumor

Dataset folder structure:
    dataset/
        Training/
            glioma/
            meningioma/
            notumor/
            pituitary/
        Testing/
            glioma/
            meningioma/
            notumor/
            pituitary/

Usage:
    python model.py --mode train
    python model.py --mode evaluate
    python model.py --mode predict --image path/to/mri.jpg
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve, auc)
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
from PIL import Image

# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────

CONFIG = {
    "data_dir":        "dataset/",
    "num_classes":     4,
    "class_names":     ["glioma", "meningioma", "notumor", "pituitary"],
    "batch_size":      32,
    "num_epochs":      30,
    "learning_rate":   1e-4,
    "weight_decay":    1e-4,
    "image_size":      224,
    "dropout1":        0.5,
    "dropout2":        0.3,
    "hidden_units":    256,
    "unfreeze_epoch":  10,
    "early_stop":      7,
    "checkpoint_path": "best_model.pth",
    "device":          "cuda" if torch.cuda.is_available() else "cpu",
}

print(f"[CONFIG] Device  : {CONFIG['device']}")
print(f"[CONFIG] Classes : {CONFIG['class_names']}")

# ─────────────────────────────────────────────────────────
# TRANSFORMS
# ─────────────────────────────────────────────────────────

train_transforms = transforms.Compose([
    transforms.Resize((CONFIG["image_size"] + 20, CONFIG["image_size"] + 20)),
    transforms.RandomCrop(CONFIG["image_size"]),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.1),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_transforms = transforms.Compose([
    transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ─────────────────────────────────────────────────────────
# DATALOADERS
# ─────────────────────────────────────────────────────────

def get_dataloaders():
    train_dir = os.path.join(CONFIG["data_dir"], "Training")
    test_dir  = os.path.join(CONFIG["data_dir"], "Testing")

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset   = datasets.ImageFolder(test_dir,  transform=val_transforms)

    class_counts   = np.bincount(train_dataset.targets)
    class_weights  = 1.0 / class_counts
    sample_weights = [class_weights[t] for t in train_dataset.targets]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"],
                              sampler=sampler, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=CONFIG["batch_size"],
                              shuffle=False,  num_workers=4, pin_memory=True)

    print(f"[DATA] Train: {len(train_dataset)} | Test: {len(val_dataset)}")
    print(f"[DATA] Mapping: {train_dataset.class_to_idx}")
    return train_loader, val_loader, class_counts


# ─────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────

class BrainTumorResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.features   = nn.Sequential(*list(backbone.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Dropout(CONFIG["dropout1"]),
            nn.Linear(2048, CONFIG["hidden_units"]),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(CONFIG["hidden_units"]),
            nn.Dropout(CONFIG["dropout2"]),
            nn.Linear(CONFIG["hidden_units"], num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def freeze_backbone(self):
        for p in self.features.parameters(): p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.features.parameters(): p.requires_grad = True

    def count_params(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[MODEL] Total: {total:,}  Trainable: {trainable:,}")


# ─────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────

def run_epoch(loader, model, criterion, optimizer=None, phase="train"):
    is_train = phase == "train"
    model.train() if is_train else model.eval()
    total_loss = correct = total = 0

    with torch.set_grad_enabled(is_train):
        for images, labels in tqdm(loader, desc=f"  {phase:5s}", leave=False):
            images, labels = images.to(CONFIG["device"]), labels.to(CONFIG["device"])
            outputs = model(images)
            loss    = criterion(outputs, labels)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            correct    += (outputs.argmax(1) == labels).sum().item()
            total      += labels.size(0)

    return total_loss / total, correct / total


def train(model, train_loader, val_loader, class_counts):
    model.freeze_backbone()
    model.count_params()
    cw        = torch.tensor(1.0 / class_counts, dtype=torch.float32).to(CONFIG["device"])
    criterion = nn.CrossEntropyLoss(weight=cw)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG["num_epochs"], eta_min=1e-6)

    history  = {k: [] for k in ("train_loss","val_loss","train_acc","val_acc")}
    best_acc = 0.0
    patience = 0

    print(f"\n{'─'*65}")
    print(f"{'Epoch':>6}  {'TrLoss':>8}  {'TrAcc':>7}  {'VaLoss':>8}  {'VaAcc':>7}  {'LR':>10}")
    print(f"{'─'*65}")

    for epoch in range(1, CONFIG["num_epochs"] + 1):
        if epoch == CONFIG["unfreeze_epoch"]:
            model.unfreeze_backbone()
            optimizer = optim.Adam(model.parameters(),
                                   lr=CONFIG["learning_rate"] * 0.1,
                                   weight_decay=CONFIG["weight_decay"])
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=CONFIG["num_epochs"] - epoch, eta_min=1e-7)
            print("[MODEL] Backbone unfrozen.")

        tl, ta = run_epoch(train_loader, model, criterion, optimizer, "train")
        vl, va = run_epoch(val_loader,   model, criterion, phase="val")
        scheduler.step()

        for k, v in zip(("train_loss","val_loss","train_acc","val_acc"), (tl,vl,ta,va)):
            history[k].append(v)

        lr   = optimizer.param_groups[0]["lr"]
        flag = " ✔" if va > best_acc else ""
        print(f"{epoch:>6}  {tl:>8.4f}  {ta:>7.4f}  {vl:>8.4f}  {va:>7.4f}  {lr:>10.2e}{flag}")

        if va > best_acc:
            best_acc = va
            torch.save(model.state_dict(), CONFIG["checkpoint_path"])
            patience = 0
        else:
            patience += 1
            if patience >= CONFIG["early_stop"]:
                print(f"[TRAIN] Early stop at epoch {epoch}.")
                break

    print(f"{'─'*65}")
    print(f"[TRAIN] Best val acc: {best_acc:.4f}  →  {CONFIG['checkpoint_path']}\n")
    return history


# ─────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────

def evaluate(model, val_loader):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="  Evaluating"):
            probs = torch.softmax(model(images.to(CONFIG["device"])), 1).cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend(probs.argmax(1))
            all_labels.extend(labels.numpy())

    all_probs  = np.array(all_probs)
    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    print("\n── Classification Report " + "─"*40)
    print(classification_report(all_labels, all_preds,
                                 target_names=CONFIG["class_names"], digits=4))
    y_bin = label_binarize(all_labels, classes=list(range(CONFIG["num_classes"])))
    print(f"Macro ROC-AUC: {roc_auc_score(y_bin, all_probs, average='macro', multi_class='ovr'):.4f}\n")

    return all_labels, all_preds, all_probs


# ─────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────

def plot_results(history, all_labels, all_preds, all_probs):
    fig = plt.figure(figsize=(20, 12))
    fig.patch.set_facecolor("#0d1117")
    C = ["#e74c3c","#f39c12","#2ecc71","#3b6fd4"]

    def style(ax, title):
        ax.set_facecolor("#161b22")
        ax.set_title(title, color="#e8eaf0", fontsize=12, pad=8)
        ax.tick_params(colors="#e8eaf0")
        for s in ax.spines.values(): s.set_edgecolor("#30363d")

    ax1 = fig.add_subplot(2,3,1)
    if history["train_loss"]:
        ax1.plot(history["train_loss"], color="#3b82f6", label="Train")
        ax1.plot(history["val_loss"],   color="#f59e0b", label="Val")
        ax1.legend(facecolor="#161b22", labelcolor="#e8eaf0")
    ax1.set_xlabel("Epoch", color="#e8eaf0"); style(ax1, "Loss Curve")

    ax2 = fig.add_subplot(2,3,2)
    if history["train_acc"]:
        ax2.plot(history["train_acc"], color="#3b82f6", label="Train")
        ax2.plot(history["val_acc"],   color="#f59e0b", label="Val")
        ax2.legend(facecolor="#161b22", labelcolor="#e8eaf0")
    ax2.set_xlabel("Epoch", color="#e8eaf0"); style(ax2, "Accuracy Curve")

    ax3 = fig.add_subplot(2,3,3)
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CONFIG["class_names"], yticklabels=CONFIG["class_names"],
                ax=ax3, linewidths=0.5, linecolor="#30363d")
    ax3.set_xlabel("Predicted", color="#e8eaf0"); ax3.set_ylabel("Actual", color="#e8eaf0")
    ax3.tick_params(colors="#e8eaf0", rotation=30); style(ax3, "Confusion Matrix")

    ax4 = fig.add_subplot(2,3,4)
    cm_n = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm_n, annot=True, fmt=".2f", cmap="YlOrRd",
                xticklabels=CONFIG["class_names"], yticklabels=CONFIG["class_names"],
                ax=ax4, vmin=0, vmax=1)
    ax4.set_xlabel("Predicted", color="#e8eaf0"); ax4.set_ylabel("Actual", color="#e8eaf0")
    ax4.tick_params(colors="#e8eaf0", rotation=30); style(ax4, "Normalised Confusion Matrix")

    ax5 = fig.add_subplot(2,3,5)
    y_bin = label_binarize(all_labels, classes=list(range(CONFIG["num_classes"])))
    for i, (cls, col) in enumerate(zip(CONFIG["class_names"], C)):
        fpr, tpr, _ = roc_curve(y_bin[:,i], all_probs[:,i])
        ax5.plot(fpr, tpr, color=col, lw=2, label=f"{cls} AUC={auc(fpr,tpr):.2f}")
    ax5.plot([0,1],[0,1],"w--",lw=1,alpha=0.4)
    ax5.set_xlabel("FPR", color="#e8eaf0"); ax5.set_ylabel("TPR", color="#e8eaf0")
    ax5.legend(facecolor="#161b22", labelcolor="#e8eaf0", fontsize=9); style(ax5, "ROC Curves")

    ax6 = fig.add_subplot(2,3,6)
    unique, counts = np.unique(all_labels, return_counts=True)
    bars = ax6.bar([CONFIG["class_names"][i] for i in unique], counts,
                   color=C, edgecolor="#30363d")
    for bar, c in zip(bars, counts):
        ax6.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                 str(c), ha="center", color="#e8eaf0", fontsize=10)
    ax6.set_ylabel("Samples", color="#e8eaf0"); style(ax6, "Test Set Distribution")

    plt.suptitle("Brain Tumor DSS — ResNet-50 Evaluation",
                 color="#e8eaf0", fontsize=15, y=1.01)
    plt.tight_layout()
    plt.savefig("results.png", dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.show()
    print("[PLOT] Saved → results.png")


# ─────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────

def load_model():
    m = BrainTumorResNet(CONFIG["num_classes"]).to(CONFIG["device"])
    m.load_state_dict(torch.load(CONFIG["checkpoint_path"], map_location=CONFIG["device"]))
    m.eval()
    print(f"[MODEL] Loaded ← {CONFIG['checkpoint_path']}")
    return m


def predict_single(image_path, model):
    img    = Image.open(image_path).convert("RGB")
    tensor = val_transforms(img).unsqueeze(0).to(CONFIG["device"])
    with torch.no_grad():
        probs = torch.softmax(model(tensor), 1).squeeze().cpu().tolist()
    top = int(np.argmax(probs))
    return {
        "prediction":    CONFIG["class_names"][top].capitalize(),
        "confidence":    round(probs[top] * 100, 2),
        "probabilities": {c: round(p*100,2) for c,p in zip(CONFIG["class_names"], probs)},
    }


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",  choices=["train","evaluate","predict"], default="train")
    parser.add_argument("--image", type=str)
    args = parser.parse_args()

    if args.mode == "train":
        train_loader, val_loader, class_counts = get_dataloaders()
        model   = BrainTumorResNet(CONFIG["num_classes"]).to(CONFIG["device"])
        history = train(model, train_loader, val_loader, class_counts)
        model   = load_model()
        _, val_loader, _ = get_dataloaders()
        labels, preds, probs = evaluate(model, val_loader)
        plot_results(history, labels, preds, probs)

    elif args.mode == "evaluate":
        _, val_loader, _ = get_dataloaders()
        model = load_model()
        labels, preds, probs = evaluate(model, val_loader)
        plot_results({k:[] for k in ("train_loss","val_loss","train_acc","val_acc")},
                     labels, preds, probs)

    elif args.mode == "predict":
        if not args.image or not Path(args.image).exists():
            print("ERROR: provide a valid --image path"); exit(1)
        model  = load_model()
        result = predict_single(args.image, model)
        print(f"\n  Prediction : {result['prediction']}")
        print(f"  Confidence : {result['confidence']}%")
        for cls, p in result["probabilities"].items():
            print(f"    {cls:<12} {p:>6.2f}%  {'█'*int(p/5)}")
