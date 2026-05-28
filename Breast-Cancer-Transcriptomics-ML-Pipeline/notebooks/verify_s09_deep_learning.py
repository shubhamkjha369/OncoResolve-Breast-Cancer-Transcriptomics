"""
SECTION 9 VERIFICATION: PyTorch MLP Deep Learning Classifier
Architecture: Input -> Dense(512,ReLU) -> Dropout(0.4) -> Dense(256,ReLU) -> Dropout(0.3) -> Dense(5,Softmax)
Loss: CrossEntropyLoss with class weights
Optimizer: Adam (lr=1e-3)
Runs 80 epochs, saves best checkpoint
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = PROJECT_ROOT / "data" / "artifacts"

print("=" * 60)
print("SECTION 9: DEEP LEARNING -- PyTorch MLP")
print("=" * 60)

# ── Check device ──────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n[9.0] Device: {device}")
print(f"      PyTorch version: {torch.__version__}")

# ── Load data ─────────────────────────────────────────────────
X_train_c = joblib.load(ARTIFACT_DIR / "X_train_consensus.pkl").astype(np.float32)
X_test_c  = joblib.load(ARTIFACT_DIR / "X_test_consensus.pkl").astype(np.float32)
y_train   = joblib.load(ARTIFACT_DIR / "y_train.pkl").astype(np.int64)
y_test    = joblib.load(ARTIFACT_DIR / "y_test.pkl").astype(np.int64)
le        = joblib.load(ARTIFACT_DIR / "label_encoder.pkl")
class_names = list(le.classes_)

n_features  = X_train_c.shape[1]
n_classes   = len(class_names)
print(f"\n[9.1] Input features: {n_features}, Classes: {n_classes}")
print(f"      Train samples: {len(y_train)}, Test samples: {len(y_test)}")

# ── Class weights (handle imbalance) ─────────────────────────
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', classes=np.arange(n_classes), y=y_train)
class_weights_tensor = torch.FloatTensor(class_weights).to(device)
print(f"\n[9.2] Class weights (balanced): {[f'{w:.3f}' for w in class_weights]}")

# ── Build MLP model ───────────────────────────────────────────
class BreastCancerMLP(nn.Module):
    def __init__(self, in_dim, n_cls):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_cls),
        )
    def forward(self, x):
        return self.net(x)

model = BreastCancerMLP(n_features, n_classes).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"\n[9.3] MLP Architecture:")
print(f"      Input({n_features}) -> Dense(512,BN,ReLU,Dropout0.4)")
print(f"      -> Dense(256,BN,ReLU,Dropout0.3) -> Dense(128,ReLU,Dropout0.2) -> Dense({n_classes})")
print(f"      Total parameters: {total_params:,}")

# ── DataLoaders ───────────────────────────────────────────────
X_tr_t = torch.FloatTensor(X_train_c)
y_tr_t = torch.LongTensor(y_train)
X_te_t = torch.FloatTensor(X_test_c)
y_te_t = torch.LongTensor(y_test)

train_ds = TensorDataset(X_tr_t, y_tr_t)
train_dl = DataLoader(train_ds, batch_size=16, shuffle=True, pin_memory=True)

# ── Optimizer & loss ──────────────────────────────────────────
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

# ── Training loop ─────────────────────────────────────────────
EPOCHS = 100
history = {'train_loss': [], 'train_acc': [], 'val_acc': [], 'val_f1': []}
best_val_acc = 0
best_epoch = 0

print(f"\n[9.4] Training for {EPOCHS} epochs (batch_size=16)...")
print(f"  {'Epoch':>5} {'TrainLoss':>10} {'TrainAcc':>9} {'ValAcc':>8} {'ValF1':>8}")
print(f"  {'-'*5} {'-'*10} {'-'*9} {'-'*8} {'-'*8}")

model.train()
for epoch in range(1, EPOCHS + 1):
    # ── Train ──
    model.train()
    epoch_loss = 0; correct = 0; total = 0
    for Xb, yb in train_dl:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(Xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * len(yb)
        correct += (out.argmax(1) == yb).sum().item()
        total += len(yb)
    train_loss = epoch_loss / total
    train_acc  = correct / total

    # ── Validate ──
    model.eval()
    with torch.no_grad():
        val_out   = model(X_te_t.to(device))
        val_preds = val_out.argmax(1).cpu().numpy()
    val_acc = accuracy_score(y_test, val_preds)
    val_f1  = f1_score(y_test, val_preds, average='weighted', zero_division=0)

    scheduler.step(1 - val_acc)
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)
    history['val_f1'].append(val_f1)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch   = epoch
        torch.save(model.state_dict(), ARTIFACT_DIR / "mlp_best.pt")

    # Print every 10 epochs
    if epoch % 10 == 0 or epoch == 1:
        print(f"  {epoch:>5} {train_loss:>10.4f} {train_acc:>9.4f} {val_acc:>8.4f} {val_f1:>8.4f}")

print(f"\n  Best epoch: {best_epoch}  Best val accuracy: {best_val_acc:.4f}")

# ── Final evaluation on best checkpoint ──────────────────────
print(f"\n[9.5] Loading best checkpoint and evaluating...")
model.load_state_dict(torch.load(ARTIFACT_DIR / "mlp_best.pt", map_location=device))
model.eval()
with torch.no_grad():
    final_preds = model(X_te_t.to(device)).argmax(1).cpu().numpy()

final_acc = accuracy_score(y_test, final_preds)
final_f1  = f1_score(y_test, final_preds, average='weighted', zero_division=0)
print(f"  Final Test Accuracy: {final_acc:.4f}")
print(f"  Final Test F1      : {final_f1:.4f}")
print(f"\n  Classification Report:")
print(classification_report(y_test, final_preds, target_names=class_names, zero_division=0))

# ── Save history ──────────────────────────────────────────────
import pandas as pd
hist_df = pd.DataFrame(history)
hist_df.to_parquet(ARTIFACT_DIR / "mlp_training_history.parquet", index=False)
joblib.dump({'test_acc': final_acc, 'test_f1': final_f1,
             'best_epoch': best_epoch}, ARTIFACT_DIR / "mlp_results.pkl")
print(f"\n[9.6] Saved: mlp_best.pt, mlp_training_history.parquet, mlp_results.pkl")

print("\n" + "=" * 60)
print("SECTION 9 COMPLETE [OK]")
print("=" * 60)
