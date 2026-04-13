# Applying EfficientNetV2 Variants for Face Verification


An advanced face verification system built with PyTorch, comparing EfficientNetV2 variants (Small, Medium, Large) using Additive Angular Margin Loss (ArcFace) and Online Hard Pair Mining. Evaluated on the standard LFW 6,000-pair benchmark.

## Team Members (Group 8)
* **Le Viet Cuong** - Project Manager, Eval & Pipeline Setup
* **Hoang Quoc Huy** - Data Engineer (MTCNN & Hard Miner)
* **Dinh Ha Hai** - ML Engineer (Baseline Siamese Network)
* **Nguyen Ngoc Linh** - ML Engineer (ArcFace & Demo)

## Key Features
* **State-of-the-Art Backbone:** Comparative study of EfficientNetV2 S/M/L architectures.
* **Advanced Metric Learning:** Replaced standard contrastive loss with ArcFace ($m=0.5, s=64$) for tighter cluster margins.
* **Online Hard Pair Mining (OHPM):** Custom PyTorch batch sampler that dynamically hunts for extreme lookalikes during the training loop.
* **Rigorous Evaluation:** Automated calculation of FAR, FRR, and EER thresholds mapped to real-world latency metrics.

---

## Environment Setup

**1. Create a Virtual Environment (Python 3.11+)**
```bash
python -m venv .venv

# Activate on Windows:
.venv\Scripts\activate

# Activate on Mac/Linux:
source .venv/bin/activate
```

**2. Install PyTorch (Hardware Specific)**

GPU (CUDA 11.8): 
```bash 
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

CPU Only: 
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**3. Install Project Dependencies**
```bash
pip install -r requirements.txt
```

## How to Run the Pipeline

**Step 1: Data Preprocessing**

Download the LFW dataset to data/raw/. Then run MTCNN to align and crop faces:
```bash
python scripts/mtcnn_preprocess.py
```

**Step 2: Training**

Train the baseline model (Contrastive Loss) and the advanced model (ArcFace):
```bash
python scripts/train_baseline.py --config configs/baseline_s.yaml
python scripts/train_arcface.py --config configs/arcface_s.yaml
```

**Step 3: Evaluation**

Evaluate the checkpoints on the LFW 6,000-pair protocol to generate the EER comparison table:
```bash
python scripts/evaluate_lfw.py
```

**Step 4: Live Demo**

Run the real-time webcam verification system:
```bash
python demo/verify.py
```