#the code was written in cells in Google Colab, like the Genetic Algorithm project

#cell 1, download the dataset

import os, json
from google.colab import userdata

os.environ['KAGGLE_TOKEN'] = userdata.get('KAGGLE_TOKEN')

!pip install -q kagglehub
import kagglehub, shutil

path = kagglehub.dataset_download(
    "youssefahmed003/junk-food-object-detection-dataset-yolo-format"
)

DATASET_DIR = '/content/junkfood'
shutil.copytree(path, DATASET_DIR, dirs_exist_ok=True)

print(f"Dataset ready → {DATASET_DIR}")
!ls {DATASET_DIR}

# cell 2, filter and augment

import yaml, shutil, random, cv2
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict

random.seed(42)
np.random.seed(42)

JUNKFOOD_ROOT    = Path('/content/junkfood')
OUT_DIR          = Path('/content/junkfood_final')
TARGET_PER_CLASS = 700

CLASS_SOURCES = [
    (0, 'burger',          'yolo_dataset', 1),
    (1, 'icecream', 'yolo_dataset', 27),
    (2, 'cookies',          'yolo_dataset', 25),
    (3, 'French_Fry',      'yolo_dataset', 26),
    (4, 'Pizza',           'yolo_dataset', 28),
    #27 icecream, 25 cookies
]
CLASS_NAMES = [c[1] for c in CLASS_SOURCES]

print("Source mapping:")
for new_id, name, folder, old_id in CLASS_SOURCES:
    print(f"  [{old_id}] {name:<20} in {folder}/ → new id [{new_id}]")

all_samples_by_class = defaultdict(list)

for new_id, name, folder, old_id in CLASS_SOURCES:
    src_root = JUNKFOOD_ROOT / folder
    found = 0
    for split in ['train', 'val', 'test']:
        lbl_dir = src_root / 'labels' / split
        img_dir = src_root / 'images' / split
        if not lbl_dir.exists():
            lbl_dir = src_root / 'labels'
            img_dir = src_root / 'images'
        if not lbl_dir.exists():
            continue
        for lbl_path in lbl_dir.glob('*.txt'):
            lines = [l for l in lbl_path.read_text().splitlines()
                     if l.strip() and int(l.split()[0]) == old_id]
            if not lines:
                continue
            img_path = None
            for ext in ['.jpg', '.jpeg', '.png', '.JPG']:
                candidate = img_dir / (lbl_path.stem + ext)
                if candidate.exists():
                    img_path = candidate
                    break
            if img_path:
                all_samples_by_class[new_id].append((lbl_path, img_path, old_id, new_id, lines))
                found += 1
    print(f"  Found {found:>4} original images for [{new_id}] {name}")

def augment_image(img):
    aug = img.copy()
    alpha = random.uniform(0.7, 1.4)
    beta  = random.randint(-30, 30)
    aug   = cv2.convertScaleAbs(aug, alpha=alpha, beta=beta)
    hsv = cv2.cvtColor(aug, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:,:,0] = (hsv[:,:,0] + random.uniform(-10, 10)) % 180
    hsv[:,:,1] = np.clip(hsv[:,:,1] * random.uniform(0.7, 1.3), 0, 255)
    hsv[:,:,2] = np.clip(hsv[:,:,2] * random.uniform(0.7, 1.3), 0, 255)
    aug = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    flipped = random.random() > 0.5
    if flipped:
        aug = cv2.flip(aug, 1)
    h, w  = aug.shape[:2]
    angle = random.uniform(-15, 15)
    M     = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    aug   = cv2.warpAffine(aug, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    zoom = random.uniform(0.0, 0.15)
    x1 = int(w * zoom / 2); y1 = int(h * zoom / 2)
    x2 = w - x1;            y2 = h - y1
    aug = cv2.resize(aug[y1:y2, x1:x2], (w, h))
    return aug, flipped

def flip_labels(lines):
    flipped = []
    for line in lines:
        parts = line.split()
        cx = 1.0 - float(parts[1])
        flipped.append(f"{parts[0]} {cx:.6f} {parts[2]} {parts[3]} {parts[4]}")
    return flipped

print(f"\nBuilding pool of {TARGET_PER_CLASS} images per class...")
final_pool = []
aug_counter = Counter()

for new_id, name, folder, old_id in CLASS_SOURCES:
    samples = all_samples_by_class[new_id]
    random.shuffle(samples)
    have = len(samples)
    need = max(0, TARGET_PER_CLASS - have)

    for lbl_path, img_path, old_id_, new_id_, lines in samples[:TARGET_PER_CLASS]:
        final_pool.append((img_path, lines, new_id, False, lbl_path))

    aug_added = 0
    attempts  = 0
    while aug_added < need and attempts < need * 3:
        lbl_path, img_path, old_id_, new_id_, lines = random.choice(samples)
        final_pool.append((img_path, lines, new_id, True, lbl_path))
        aug_added += 1
        attempts  += 1

    print(f"  [{new_id}] {name:<20} {min(have, TARGET_PER_CLASS)} original "
          f"+ {aug_added} augmented = {min(have, TARGET_PER_CLASS) + aug_added}")

random.shuffle(final_pool)
n         = len(final_pool)
train_end = int(n * 0.75)
val_end   = int(n * 0.90)

split_data = {
    'train': final_pool[:train_end],
    'val':   final_pool[train_end:val_end],
    'test':  final_pool[val_end:],
}

for split in ['train', 'val', 'test']:
    (OUT_DIR / split / 'images').mkdir(parents=True, exist_ok=True)
    (OUT_DIR / split / 'labels').mkdir(parents=True, exist_ok=True)

stem_counters = Counter()

for split, samples in split_data.items():
    for img_path, lines, new_id, is_augment, orig_lbl in samples:
        base_stem = orig_lbl.stem
        if is_augment:
            stem_counters[base_stem] += 1
            stem = f"{base_stem}_aug{stem_counters[base_stem]}"
        else:
            stem = base_stem

        new_lines = []
        for line in lines:
            parts = line.split()
            new_lines.append(f"{new_id} " + ' '.join(parts[1:]))

        if is_augment:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            aug, flipped = augment_image(img)
            if flipped:
                new_lines = flip_labels(new_lines)
            cv2.imwrite(
                str(OUT_DIR / split / 'images' / (stem + '.jpg')),
                aug, [cv2.IMWRITE_JPEG_QUALITY, 92]
            )
        else:
            shutil.copy(img_path, OUT_DIR / split / 'images' / (stem + img_path.suffix))

        (OUT_DIR / split / 'labels' / (stem + '.txt')).write_text('\n'.join(new_lines))

new_cfg = {
    'path':  str(OUT_DIR),
    'train': str(OUT_DIR / 'train/images'),
    'val':   str(OUT_DIR / 'val/images'),
    'test':  str(OUT_DIR / 'test/images'),
    'nc':    5,
    'names': CLASS_NAMES,
}
with open(OUT_DIR / 'data.yaml', 'w') as f:
    yaml.dump(new_cfg, f, default_flow_style=False)

print(f"\nFinal dataset → {OUT_DIR}")
print(f"{'─'*60}")
for split in ['train', 'val', 'test']:
    counts = Counter()
    for lbl in (OUT_DIR / split / 'labels').glob('*.txt'):
        for line in lbl.read_text().splitlines():
            if line.strip():
                try:
                    counts[int(line.split()[0])] += 1
                except:
                    continue
    total = len(list((OUT_DIR / split / 'images').glob('*')))
    print(f"\n  {split} ({total} images):")
    for idx, name in enumerate(CLASS_NAMES):
        n    = counts.get(idx, 0)
        bar  = '|' * (n // 5)
        flag = '! ' if n < 50 else '$'
        print(f"    {flag} [{idx}] {name:<20} {n:>5}  {bar}")

!pip install ultralytics

# cell 3, yolo training

from ultralytics import YOLO
import shutil
from pathlib import Path

run_dir = Path('/content/runs/junkfood_v1')
if run_dir.exists():
    shutil.rmtree(run_dir)
    print("Cleared previous run")

model = YOLO('yolo11s.pt')

results = model.train(
    data         = '/content/junkfood_final/data.yaml',
    epochs       = 50,
    patience     = 10,
    imgsz        = 640,
    batch        = 16,
    device       = 0,
    optimizer    = 'AdamW',
    lr0          = 0.001,
    lrf          = 0.01,
    warmup_epochs= 3,
    mosaic       = 1.0,
    mixup        = 0.15,
    copy_paste   = 0.15,
    hsv_h        = 0.015,
    hsv_s        = 0.7,
    hsv_v        = 0.4,
    degrees      = 10.0,
    translate    = 0.1,
    scale        = 0.6,
    shear        = 5.0,
    perspective  = 0.0005,
    flipud       = 0.0,
    fliplr       = 0.5,
    cls          = 0.7,
    box          = 8.0,
    dropout      = 0.1,
    weight_decay = 0.0005,
    amp          = True,
    save_period  = 5, #better to save tests later for the testing plot
    project      = '/content/runs',
    name         = 'junkfood_v1',
    exist_ok     = True,
)

print(f"\nTraining complete.")
print(f"   mAP@50    → {results.results_dict.get('metrics/mAP50(B)', 0):.4f}")
print(f"   mAP@50-95 → {results.results_dict.get('metrics/mAP50-95(B)', 0):.4f}")

# cell 4, evaluation

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from IPython.display import Image, display
from ultralytics import YOLO

CLASS_NAMES = ['burger', 'icecream', 'cookies', 'French_Fry', 'Pizza']
RUN_DIR     = Path('/content/runs/junkfood_v1')

df = pd.read_csv(RUN_DIR / 'results.csv')
df.columns = df.columns.str.strip()
EPOCHS = df['epoch']

fig = plt.figure(figsize=(16, 10), dpi=150)
fig.suptitle('YOLOv11s — Junk Food Detection Training Metrics',
             fontsize=14, fontweight='bold')
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

def styled_plot(ax, x, y, label, color, ylabel, title):
    ax.plot(x, y, color=color, linewidth=2, label=label)
    ax.fill_between(x, y, alpha=0.08, color=color)
    ax.set_xlabel('Epoch', fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

ax0 = fig.add_subplot(gs[0, :])
ax0.plot(EPOCHS, df['metrics/mAP50(B)'],    color='#185FA5', lw=2, label='mAP@50')
ax0.plot(EPOCHS, df['metrics/mAP50-95(B)'], color='#639922', lw=2, label='mAP@50-95')
ax0.fill_between(EPOCHS, df['metrics/mAP50(B)'],    alpha=0.07, color='#185FA5')
ax0.fill_between(EPOCHS, df['metrics/mAP50-95(B)'], alpha=0.07, color='#639922')
ax0.axhline(0.90, color='#E24B4A', linestyle='--', lw=1.2, label='90% target')
ax0.set_title('Detection Accuracy (mAP)', fontsize=11, fontweight='bold')
ax0.set_xlabel('Epoch')
ax0.set_ylabel('mAP')
ax0.legend(fontsize=9)
ax0.grid(True, alpha=0.3, linestyle='--')
ax0.set_ylim(0, 1.05)

styled_plot(fig.add_subplot(gs[1, 0]), EPOCHS,
    df['train/box_loss'], 'Box Loss', '#E24B4A', 'Loss', 'Box Loss')
styled_plot(fig.add_subplot(gs[1, 1]), EPOCHS,
    df['train/cls_loss'], 'Cls Loss', '#BA7517', 'Loss', 'Classification Loss')
styled_plot(fig.add_subplot(gs[1, 2]), EPOCHS,
    df['train/dfl_loss'], 'DFL Loss', '#534AB7', 'Loss', 'Distribution Focal Loss')

plt.savefig('/content/training_curves.png', bbox_inches='tight', dpi=150)
plt.show()
print("Saved → /content/training_curves.png")

model_eval  = YOLO(str(RUN_DIR / 'weights/best.pt'))
val_results = model_eval.val(
    data      = '/content/junkfood_final/data.yaml',
    split     = 'test',
    plots     = True,
    save_json = True,
)

print(f"\n📊 Test Set Results:")
print(f"   mAP@50     → {val_results.box.map50:.4f}")
print(f"   mAP@50-95  → {val_results.box.map:.4f}")
print(f"   Precision  → {val_results.box.mp:.4f}")
print(f"   Recall     → {val_results.box.mr:.4f}")

print("\nPer-class AP@50:")
for i, (ap, name) in enumerate(zip(val_results.box.ap50, CLASS_NAMES)):
    bar  = '|' * int(ap * 30)
    flag = '! ' if ap < 0.85 else '$'
    print(f"  {flag} [{i}] {name:<20} AP@50: {ap:.4f}  {bar}")

cm_path = RUN_DIR / 'confusion_matrix_normalized.png'
if cm_path.exists():
    display(Image(str(cm_path), width=600))
else:
    for p in Path('/content/runs').rglob('confusion_matrix_normalized.png'):
        display(Image(str(p), width=600))
        break

# cell 4.1, evaluate saved tests every 5 epochs

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from ultralytics import YOLO

RUN_DIR     = Path('/content/runs/junkfood_v1')
CLASS_NAMES = ['burger', 'Ice_Cream', 'French_Fry', 'Pizza', 'Cookies']

checkpoints = sorted(RUN_DIR.glob('weights/epoch*.pt'))
extra = []
for name in ['best.pt', 'last.pt']:
    p = RUN_DIR / 'weights' / name
    if p.exists():
        extra.append(p)

print(f"Found {len(checkpoints)} epoch checkpoints + {len(extra)} extra")
print("Evaluating each on test set — this takes a few minutes...\n")

records = []

for ckpt in checkpoints + extra:
    model = YOLO(str(ckpt))
    results = model.val(
        data    = '/content/junkfood_final/data.yaml',
        split   = 'test',
        verbose = False,
        plots   = False,
    )

    if 'epoch' in ckpt.stem:
        epoch = int(ckpt.stem.replace('epoch', ''))
    elif ckpt.stem == 'best':
        epoch = 9999
    elif ckpt.stem == 'last':
        epoch = 9998
    else:
        epoch = -1

    record = {
        'epoch':      epoch,
        'label':      ckpt.stem,
        'mAP50':      results.box.map50,
        'mAP50_95':   results.box.map,
        'precision':  results.box.mp,
        'recall':     results.box.mr,
    }

    for i, (ap, name) in enumerate(zip(results.box.ap50, CLASS_NAMES)):
        record[f'ap_{name}'] = ap

    records.append(record)
    print(f"  {ckpt.stem:<15} mAP@50: {results.box.map50:.4f}  "
          f"mAP@50-95: {results.box.map:.4f}")

df = pd.DataFrame(records).sort_values('epoch').reset_index(drop=True)

fig, axes = plt.subplots(1, 2, figsize=(16, 5), dpi=150)
fig.suptitle('Test Set Performance Across Training Checkpoints',
             fontsize=13, fontweight='bold')
df_epochs = df[df['epoch'] < 9000].copy()
df_best   = df[df['label'] == 'best'].iloc[0] if 'best' in df['label'].values else None

ax = axes[0]
ax.plot(df_epochs['epoch'], df_epochs['mAP50'],
        color='#185FA5', lw=2, marker='o', markersize=4, label='mAP@50 (test)')
ax.plot(df_epochs['epoch'], df_epochs['mAP50_95'],
        color='#639922', lw=2, marker='o', markersize=4, label='mAP@50-95 (test)')
ax.fill_between(df_epochs['epoch'], df_epochs['mAP50'],    alpha=0.07, color='#185FA5')
ax.fill_between(df_epochs['epoch'], df_epochs['mAP50_95'], alpha=0.07, color='#639922')

if df_best is not None:
    ax.axhline(df_best['mAP50'], color='#E24B4A', linestyle='--',
               lw=1.5, label=f"best.pt mAP@50: {df_best['mAP50']:.4f}")

ax.set_title('mAP on Test Set', fontweight='bold')
ax.set_xlabel('Epoch')
ax.set_ylabel('mAP')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_ylim(0, 1.05)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax2 = axes[1]
ax2.plot(df_epochs['epoch'], df_epochs['precision'],
         color='#534AB7', lw=2, marker='o', markersize=4, label='Precision')
ax2.plot(df_epochs['epoch'], df_epochs['recall'],
         color='#BA7517', lw=2, marker='o', markersize=4, label='Recall')
ax2.fill_between(df_epochs['epoch'], df_epochs['precision'], alpha=0.07, color='#534AB7')
ax2.fill_between(df_epochs['epoch'], df_epochs['recall'],    alpha=0.07, color='#BA7517')
ax2.set_title('Precision & Recall on Test Set', fontweight='bold')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Score')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_ylim(0, 1.05)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('/content/test_mAP_curve.png', bbox_inches='tight', dpi=150)
plt.show()
print("Saved → /content/test_mAP_curve.png")

fig2, ax3 = plt.subplots(figsize=(10, 5), dpi=150)
fig2.suptitle('Per-Class AP@50 on Test Set (best.pt)',
              fontsize=13, fontweight='bold')

if df_best is not None:
    class_aps   = [df_best.get(f'ap_{name}', 0) for name in CLASS_NAMES]
    bar_colors  = ['#E24B4A','#BA7517','#639922','#185FA5','#534AB7']
    bars = ax3.bar(CLASS_NAMES, class_aps, color=bar_colors, width=0.5)

    for bar, val in zip(bars, class_aps):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{val:.3f}', ha='center', va='bottom',
                 fontsize=11, fontweight='bold')

    ax3.axhline(0.85, color='#E24B4A', linestyle='--',
                lw=1.2, label='0.85 minimum target')
    ax3.set_ylim(0, 1.1)
    ax3.set_ylabel('AP@50')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('/content/test_perclass_AP.png', bbox_inches='tight', dpi=150)
plt.show()
print("Saved → /content/test_perclass_AP.png")

print(f"\n{'─'*65}")
print(f"{'Checkpoint':<15} {'mAP@50':>8} {'mAP50-95':>10} {'Precision':>10} {'Recall':>8}")
print(f"{'─'*65}")
for _, row in df.iterrows():
    label = row['label']
    print(f"{label:<15} {row['mAP50']:>8.4f} {row['mAP50_95']:>10.4f} "
          f"{row['precision']:>10.4f} {row['recall']:>8.4f}")
print(f"{'─'*65}")

# cell 5, kcals and user input

import json, cv2
import numpy as np
from ultralytics import YOLO
from google.colab.patches import cv2_imshow
from google.colab import files

CLASS_NAMES = ['burger', 'icecream', 'cookies', 'French_Fry', 'Pizza']

CALORIE_MAP = {
    'burger':     {'kcal': 550, 'note': 'whole burger with bun (~200g)'},
    'icecream':   {'kcal': 230, 'note': 'per scoop/cone (~100g)'},
    'cookies':    {'kcal': 85,  'note': 'per cookie (~30g)'},
    'French_Fry': {'kcal': 365, 'note': 'medium portion (~117g)'},
    'Pizza':      {'kcal': 285, 'note': 'per slice (~125g)'},
}

with open('/content/calorie_map.json', 'w') as f:
    json.dump(CALORIE_MAP, f, indent=2)

print("calorie_map.json saved")
print("\nCalorie reference (per detection):")
print(f"  {'Food':<15} {'kcal/unit':>10}  Note")
print(f"  {'─'*55}")
for name, info in CALORIE_MAP.items():
    print(f"  {name:<15} {info['kcal']:>10}  {info['note']}")

COLORS = {
    'burger':     (226,  75,  74),
    'icecream':   ( 15, 110, 186),
    'cookies':    (140,  80, 200),
    'French_Fry': ( 15, 110,  86),
    'Pizza':      (212,  90,  48),
}

TRAIN_IMGSZ = 640

def prepare_image(image_path: str):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")
    h, w    = img.shape[:2]
    scale   = TRAIN_IMGSZ / max(h, w)
    new_w   = int(w * scale)
    new_h   = int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas  = np.full((TRAIN_IMGSZ, TRAIN_IMGSZ, 3), 114, dtype=np.uint8)
    canvas[:new_h, :new_w] = resized
    print(f"  Original : {w}x{h}px")
    print(f"  Prepared : {new_w}x{new_h}px + padding -> {TRAIN_IMGSZ}x{TRAIN_IMGSZ}px")
    return canvas, scale, (w, h)

def detect_and_estimate(image_path: str, conf_thresh: float = 0.25):
    """
    Detects all food items in the image and calculates total calories
    based on the actual number of detections per class.

    conf_thresh: minimum confidence to count a detection (default 0.25)
                 lower to 0.15 if food is being missed
                 raise to 0.40 if wrong detections appear
    """
    model = YOLO('/content/runs/junkfood_v1/weights/best.pt')
    img, scale, (orig_w, orig_h) = prepare_image(image_path)

    results = model(img, conf=conf_thresh, iou=0.35, augment=True, verbose=False)

    class_detections = {}

    for r in results:
        for box in r.boxes:
            cls_name = model.names[int(box.cls)]
            conf     = float(box.conf)
            color    = COLORS.get(cls_name, (200, 200, 200))
            info     = CALORIE_MAP.get(cls_name, {})

            if not info:
                continue

            kcal_per_unit = info['kcal']
            x1,y1,x2,y2  = map(int, box.xyxy[0].tolist())

            if cls_name not in class_detections:
                class_detections[cls_name] = []
            class_detections[cls_name].append(conf)

            cv2.rectangle(img, (x1,y1), (x2,y2), color, 3)

            label = f"{cls_name}  {kcal_per_unit} kcal  ({conf:.0%})"
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(img, (x1, y1-th-12), (x1+tw+8, y1), color, -1)
            cv2.putText(img, label, (x1+4, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)

    total_kcal = 0
    summary    = []

    for cls_name, confs in class_detections.items():
        count         = len(confs)
        kcal_per_unit = CALORIE_MAP[cls_name]['kcal']
        kcal_total    = kcal_per_unit * count
        avg_conf      = sum(confs) / len(confs)
        total_kcal   += kcal_total
        summary.append((cls_name, count, kcal_per_unit, kcal_total, avg_conf))

    print("\nDetection Results")
    print("─" * 65)
    if not summary:
        print("  Nothing detected at this confidence level.")
        print("  -> Lower conf_thresh or run debug_detections(path)")
    else:
        print(f"  {'Food':<15} {'Count':>6} {'kcal/unit':>10} {'Total kcal':>11} {'Avg conf':>9}")
        print(f"  {'─'*55}")
        for cls_name, count, kcal_unit, kcal_tot, avg_conf in summary:
            note = CALORIE_MAP[cls_name]['note']
            print(f"  {cls_name:<15} {count:>6} {kcal_unit:>10} {kcal_tot:>11}  ({avg_conf:.0%})")
            print(f"  {'':15} {'':>6}  -> {note}")
    print("─" * 65)
    print(f"  TOTAL ESTIMATED: {total_kcal} kcal")

    cv2_imshow(img)
    cv2.imwrite('/content/detection_result.jpg', img)
    print("\nSaved -> /content/detection_result.jpg")
    return img, summary, total_kcal

def debug_detections(image_path: str):
    """Shows ALL candidates at very low confidence to diagnose failures."""
    model = YOLO('/content/runs/junkfood_v1/weights/best.pt')
    img, scale, _ = prepare_image(image_path)
    results = model(img, conf=0.05, iou=0.3, augment=True, verbose=False)

    print("\nDebug -- all candidates at conf=0.05:")
    print("─" * 55)
    found = 0
    for r in results:
        for box in r.boxes:
            cls_name = model.names[int(box.cls)]
            conf     = float(box.conf)
            verdict  = "shown normally" if conf >= 0.25 else "filtered out"
            print(f"  {cls_name:<15} conf: {conf:.3f}  {verdict}")
            found += 1
    if found == 0:
        print("  Nothing found even at 5% confidence.")
        print("  Image looks too different from training data.")
    print("─" * 55)
    print(f"  Total candidates: {found}")
    cv2_imshow(img)

print("Upload an image to run detection:")
uploaded = files.upload()
for fname in uploaded:
    path = f'/content/{fname}'
    print(f"\n{'─'*55}")
    print(f"Processing: {fname}")

    detect_and_estimate(
        path,
        conf_thresh = 0.15,  #lower for undetected, increase for wrong detection
    )

# cell 6, final values(to see what could be improved)

from ultralytics import YOLO

CLASS_NAMES = ['burger', 'icecream', 'cookies', 'French_Fry', 'Pizza']

model = YOLO('/content/runs/junkfood_v1/weights/best.pt')

print("Running TTA validation on test set...")
tta_results = model.val(
    data    = '/content/junkfood_final/data.yaml',
    split   = 'test',
    augment = True,
    conf    = 0.3,
    iou     = 0.5,
)

print(f"\n📊 TTA Results:")
print(f"   mAP@50    → {tta_results.box.map50:.4f}")
print(f"   mAP@50-95 → {tta_results.box.map:.4f}")
print(f"   Precision → {tta_results.box.mp:.4f}")
print(f"   Recall    → {tta_results.box.mr:.4f}")

print("\nPer-class with TTA:")
for i, (ap, name) in enumerate(zip(tta_results.box.ap50, CLASS_NAMES)):
    flag = '! ' if ap < 0.85 else '$'
    bar  = '|' * int(ap * 30)
    print(f"  {flag} [{i}] {name:<20} {ap:.4f}  {bar}")

print("\n📌 Optimizations in this run:")
print("  1. Augmented to 700/class  — rotation, flip, HSV, zoom, brightness")
print("  2. yolo11s backbone        — 3x params vs nano")
print("  3. TTA at inference        — augment=True")
print("  4. cls=0.7, box=8.0        — stronger class penalty")
print("  5. Early stop patience=15  — no wasted epochs")
