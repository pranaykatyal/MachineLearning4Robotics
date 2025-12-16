# HWFin — Monocular Depth Estimation (Summary)
Final Project - Pranay Katyal, Anirudh Ramanathan

Core locations
- Codebase: M4Depth/ (cloned project used for eval & fine-tune)
- Requirements: M4Depth/requirements.txt

Primary outputs (paths inside this repo)
- Phase 1 — Evaluate pretrained MidAir on UseGeo
  - Predictions & visuals: M4Depth/pretrained_weights/midair/eval_outputs/usegeo/
    - predictions/ (pred_*.tiff + gt_*.tiff)
    - visualizations/ (pred_*.png)
    - final_comparisons/ (hand-picked PNGs for reports)

- Phase 2 — Fine-tune on UseGeo
  - Training checkpoints & logs: M4Depth/pretrained_weights/midair/train/
    - checkpoints/ (saved ckpts, best model)
  - Training figures (compact): HWFin/phase2_training_curves.png
  - Comparison metrics/summary: HWFin/phase2_comparison_metrics.png
  - Final best model (example): M4Depth/pretrained_weights/midair/train/checkpoints/best.ckpt

Deliverables (paper / slides)
- Report PDF: HWFin/report.pdf
- Presentation PPTX: HWFin/presentation.pptx

Quick run pointers
- Prepare env:
  - conda activate m4depth_tf
  - cd M4Depth
  - ensure pretrained checkpoint in: pretrained_weights/midair/best/
- Eval (single-machine):
  - python eval_and_save_usegeo.py --config configs/usegeo_eval.yml --ckpt pretrained_weights/midair/best/cp-0071.ckpt --out-dir pretrained_weights/midair/eval_outputs/usegeo/
- Fine-tune:
  - place checkpoint in pretrained_weights/midair/train/ then run training wrapper (see finetune_usegeo.sh or train.py for args)

Exact commands (copy & paste)

```bash
# 1) Create / activate environment and install requirements
conda create -n m4depth_tf python=3.9 -y
conda activate m4depth_tf
pip install -r M4Depth/requirements.txt

# 2) Prepare checkpoint (example)
mkdir -p M4Depth/pretrained_weights/midair/best
cp /path/to/cp-0071.ckpt.* M4Depth/pretrained_weights/midair/best/

# 3) Convert UseGeo poses (required)
python M4Depth/convert_usegeo_poses.py --input /path/to/usegeo/poses_abs.csv --output /path/to/usegeo/poses_relative.csv

# 4) Generate dataset records (if needed)
python M4Depth/create_usegeo_records.py --data-root /path/to/UseGeo/images --poses /path/to/usegeo/poses_relative.csv --out-csv usegeo_records.csv

# 5) Run evaluation (single GPU)
cd M4Depth
# adjust flags as needed; --out-dir will contain predictions & visualizations
python eval_and_save_usegeo.py --config configs/usegeo_eval.yml \
    --ckpt pretrained_weights/midair/best/cp-0071.ckpt \
    --out-dir pretrained_weights/midair/eval_outputs/usegeo/

# 6) Start fine-tuning (interactive)
mkdir -p pretrained_weights/midair/train
cp pretrained_weights/midair/best/cp-0071.ckpt.* pretrained_weights/midair/train/
python train.py --config configs/finetune_usegeo.yml \
    --train-csv /path/to/train_relative.csv --val-csv /path/to/val_relative.csv \
    --ckpt pretrained_weights/midair/train/cp-0071.ckpt

# 7) Submit via SLURM (example)
sbatch finetune_usegeo.sh
sbatch eval_usegeo.sh

# 8) Run TensorBoard to monitor logs (from repo root)
tensorboard --logdir M4Depth/runs --port 6006
# then open http://localhost:6006
```

Notes & gotchas (brief)
- POSE FORMAT: convert absolute UseGeo poses → relative translations via convert_usegeo_poses.py (required).
- TIFF depth maps are 32-bit floats; keep PIL/hw-friendly loaders.
- If running on cluster, use provided SLURM wrappers: eval_usegeo.sh, finetune_usegeo.sh.

