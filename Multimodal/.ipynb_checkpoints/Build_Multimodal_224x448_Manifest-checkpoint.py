# Build_Multimodal_224x448_Manifest.py
#
# 配對邏輯（row-based，最嚴謹）：
#   對每一筆 master_manifest.csv 的 train_valid exam unit：
#     - 無 label_conflict
#     - 同時有 OCTA3
#     - 有 OCT0 或 OCT1（至少一張）
#     → 從該列的 OCT0 / OCT1 隨機取一張（同一 exam_key 內，不跨病人）
#     → 與同列的 OCTA3 上下垂直拼接（OCT 上 / OCTA3 下）
#     → 輸出 224×448 RGB image
#
#   保證：
#     同一 exam_key → 同病人、同眼、同日期、同 label
#     不跨病人配對
#     fold_id 沿用 master_manifest，不影響 train/val patient split
#     只保留 same-state（y_true=0 → inactive, y_true=1 → active）
#
# 執行：
#   python Build_Multimodal_224x448_Manifest.py
#
# 輸出：
#   BASE_OUTPUT/
#       images/
#           class_0_inactive/  *.jpg
#           class_1_active/    *.jpg
#       multimodal_manifest.csv
#       build_audit.json
#
# multimodal_manifest.csv 欄位（對齊 Swin_train_multimodal_oof.py）：
#   exam_key, patient_id, eye, exam_date,
#   split_set, fold_id, y_true, class_name, label_conflict,
#   has_oct0(=1), has_oct1(=0), has_octa3(=0),
#   oct0_image_path,   ← 合併後 multimodal image 路徑（訓練直接用此欄）
#   oct1_image_path,   ← 空
#   octa3_image_path,  ← 空
#   src_oct_path,      ← 實際使用的原始 OCT 路徑（記錄用）
#   src_octa3_path     ← 原始 OCTA3 路徑（記錄用）

import os
import csv
import json
import random
from collections import Counter
from datetime import datetime
from pathlib import Path

from tqdm import tqdm
from PIL import Image

# ===================== Config =====================

MASTER_MANIFEST_CSV = (
    "/data/Irene/SwinTransformer/Swin_Meta/outputs/manifests/"
    "master_split/master_manifest.csv"
)

# 原始影像根目錄（manifest 內若已是絕對路徑請留空 ""）
IMAGE_ROOT = ""

# 輸出根目錄
BASE_OUTPUT = "/data/Irene/Multimodal_224x448"

# 影像尺寸
IMAGE_SIZE = 224           # 單張 resize 目標
COMBINED_W = IMAGE_SIZE    # 224
COMBINED_H = IMAGE_SIZE * 2  # 448

# 只處理 train_valid（test 推論時再另行處理）
TARGET_SPLIT_SET = "train_valid"

# 隨機種子（OCT0/OCT1 二選一）
RANDOM_SEED = 42

# 類別定義
CLASS_INFO = {
    0: {"name": "inactive", "dir": "class_0_inactive"},
    1: {"name": "active",   "dir": "class_1_active"},
}

# ===================== Pillow 相容 =====================
try:
    _RESAMPLE = Image.Resampling.BILINEAR
except AttributeError:
    _RESAMPLE = Image.BILINEAR


# ===================== Utilities =====================

def safe_int(v, default=None):
    try:
        return int(float(v))
    except Exception:
        return default


def resolve_path(p) -> str:
    """相對路徑補上 IMAGE_ROOT；已是絕對路徑直接回傳；空值回傳空字串。"""
    if not p or str(p).strip() in ("", "nan", "None"):
        return ""
    p = str(p).strip()
    if os.path.isabs(p):
        return p
    if IMAGE_ROOT:
        return os.path.join(IMAGE_ROOT, p)
    return p


def read_master_manifest(csv_path: str) -> list:
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def make_combined_image(oct_path: str, octa_path: str):
    """
    讀取 OCT 與 OCTA3，convert("RGB")（保留 Swin pretrained 3-ch 期望），
    各自 resize 至 224×224，垂直堆疊（OCT 上 / OCTA3 下）→ 224×448 RGB。
    失敗回傳 None。
    """
    try:
        img_oct  = Image.open(oct_path).convert("RGB").resize(
            (IMAGE_SIZE, IMAGE_SIZE), _RESAMPLE
        )
        img_octa = Image.open(octa_path).convert("RGB").resize(
            (IMAGE_SIZE, IMAGE_SIZE), _RESAMPLE
        )
    except Exception as e:
        print(f"  [WARN] Failed to read image: {oct_path} | {octa_path} | {e}")
        return None

    combo = Image.new("RGB", (COMBINED_W, COMBINED_H))
    combo.paste(img_oct,  (0, 0))
    combo.paste(img_octa, (0, IMAGE_SIZE))
    return combo


# ===================== Main =====================

def main():
    random.seed(RANDOM_SEED)

    out_root   = Path(BASE_OUTPUT)
    images_dir = out_root / "images"
    for info in CLASS_INFO.values():
        (images_dir / info["dir"]).mkdir(parents=True, exist_ok=True)

    manifest_path = out_root / "multimodal_manifest.csv"
    audit_path    = out_root / "build_audit.json"

    print("=" * 80)
    print("Build Multimodal 224x448 Manifest  [row-based pairing]")
    print(f"  Master manifest : {MASTER_MANIFEST_CSV}")
    print(f"  Output root     : {out_root}")
    print("=" * 80)

    # ── 1. 讀取 master manifest ──────────────────────────────────────────────
    all_rows = read_master_manifest(MASTER_MANIFEST_CSV)
    print(f"[1] Extract master manifest：{len(all_rows)} (num)")

    # ── 2. Row-based 篩選與配對準備 ───────────────────────────────────────────
    # 每列獨立處理，確保同一 exam_key 內配對，無跨病人、跨日期問題
    pairs            = []
    skip_split       = 0   # 非 train_valid
    skip_conflict    = 0   # label_conflict != 0
    skip_no_ytrue    = 0   # y_true 缺失或不在 CLASS_INFO
    skip_no_modality = 0   # 缺 OCTA3 或 OCT

    for r in all_rows:

        # 僅處理 train_valid
        if str(r.get("split_set", "")).strip() != TARGET_SPLIT_SET:
            skip_split += 1
            continue

        # 過濾 label_conflict
        if safe_int(r.get("label_conflict", 0), 0) != 0:
            skip_conflict += 1
            continue

        # y_true 必須合法（0 或 1）
        ytrue_int = safe_int(r.get("y_true", ""), -1)
        if ytrue_int not in CLASS_INFO:
            skip_no_ytrue += 1
            continue

        # 建立 OCT 候選清單（OCT0 / OCT1，取有路徑者）
        oct_candidates = []
        if safe_int(r.get("has_oct0", 0), 0):
            p = resolve_path(r.get("oct0_image_path", ""))
            if p:
                oct_candidates.append(p)
        if safe_int(r.get("has_oct1", 0), 0):
            p = resolve_path(r.get("oct1_image_path", ""))
            if p:
                oct_candidates.append(p)

        # OCTA3 路徑
        octa3_path = ""
        if safe_int(r.get("has_octa3", 0), 0):
            octa3_path = resolve_path(r.get("octa3_image_path", ""))

        # 兩者都需存在才能配對
        if not oct_candidates or not octa3_path:
            skip_no_modality += 1
            continue

        pairs.append({
            "exam_key":         str(r.get("exam_key",   "")),
            "patient_id":       str(r.get("patient_id", "")),
            "eye":              str(r.get("eye",         "")),
            "exam_date":        str(r.get("exam_date",   "")),
            "split_set":        TARGET_SPLIT_SET,
            "fold_id":          str(r.get("fold_id",     "")),
            "y_true":           ytrue_int,
            "class_name":       CLASS_INFO[ytrue_int]["name"],
            "label_conflict":   0,
            "_oct_candidates":  oct_candidates,   # 僅用於配對，不寫入 manifest
            "_octa3_path":      octa3_path,
        })

    print(f"[2] Filter results：")
    print(f"    not train_valid Skip    : {skip_split}")
    print(f"    label_conflict Skip   : {skip_conflict}")
    print(f"    y_true fail Skip       : {skip_no_ytrue}")
    print(f"    Lost OCT or OCTA3 Skip  : {skip_no_modality}")
    print(f"    Pairing            : {len(pairs)}")
    y_cnt = Counter(p["y_true"] for p in pairs)
    print(f"    class 0 (inactive)    : {y_cnt.get(0, 0)}")
    print(f"    class 1 (active)      : {y_cnt.get(1, 0)}")

    # ── 3. 建立合併影像 + 輸出 manifest ──────────────────────────────────────
    print("[3] 合併影像並輸出 manifest ...")

    manifest_fieldnames = [
        "exam_key", "patient_id", "eye", "exam_date",
        "split_set", "fold_id", "y_true", "class_name", "label_conflict",
        "has_oct0", "has_oct1", "has_octa3",
        "oct0_image_path",    # ← 訓練腳本由此欄讀取合併影像
        "oct1_image_path",    # ← 空
        "octa3_image_path",   # ← 空
        "src_oct_path",       # ← 原始 OCT 路徑（記錄用）
        "src_octa3_path",     # ← 原始 OCTA3 路徑（記錄用）
    ]

    saved_count = Counter()
    skipped_io  = 0
    cls_counter = Counter()   # 同類別 filename index

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=manifest_fieldnames)
        wr.writeheader()

        for pair in tqdm(pairs, desc="Saving multimodal images"):
            yi = pair["y_true"]

            # 從同一 exam 的 OCT 候選中隨機取一張
            chosen_oct   = random.choice(pair["_oct_candidates"])
            chosen_octa3 = pair["_octa3_path"]

            cls_counter[yi] += 1
            fname = (
                f"{pair['patient_id']}{pair['eye']}"
                f"_{cls_counter[yi]:06d}"
                f"_{CLASS_INFO[yi]['name']}.jpg"
            )
            out_path = images_dir / CLASS_INFO[yi]["dir"] / fname

            combo = make_combined_image(chosen_oct, chosen_octa3)
            if combo is None:
                skipped_io += 1
                continue

            combo.save(str(out_path), quality=95)
            saved_count[yi] += 1

            wr.writerow({
                "exam_key":         pair["exam_key"],
                "patient_id":       pair["patient_id"],
                "eye":              pair["eye"],
                "exam_date":        pair["exam_date"],
                "split_set":        pair["split_set"],
                "fold_id":          pair["fold_id"],
                "y_true":           pair["y_true"],
                "class_name":       pair["class_name"],
                "label_conflict":   0,
                "has_oct0":         1,
                "has_oct1":         0,
                "has_octa3":        0,
                "oct0_image_path":  str(out_path),  # ← 訓練讀此欄
                "oct1_image_path":  "",
                "octa3_image_path": "",
                "src_oct_path":     chosen_oct,
                "src_octa3_path":   chosen_octa3,
            })

    total_saved = saved_count[0] + saved_count[1]
    print(f"    ✓ Save image：class0={saved_count[0]}, class1={saved_count[1]}, Total={total_saved}")
    print(f"    ✗ IO Failed Skip：{skipped_io}")

    # ── 4. 寫入 audit JSON ────────────────────────────────────────────────────
    audit = {
        "time":             datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "master_manifest":  MASTER_MANIFEST_CSV,
        "output_root":      str(out_root),
        "manifest_path":    str(manifest_path),
        "target_split_set": TARGET_SPLIT_SET,
        "random_seed":      RANDOM_SEED,
        "pairing_strategy": (
            "row-based: per exam_key, random.choice(OCT0/OCT1) "
            "paired with OCTA3 from same exam"
        ),
        "image_geometry": {
            "single_image_size": IMAGE_SIZE,
            "combined_width":    COMBINED_W,
            "combined_height":   COMBINED_H,
            "layout":            "OCT top (0~223) / OCTA3 bottom (224~447)",
            "color_mode":        "RGB (3-channel, compatible with Swin pretrained weights)",
        },
        "input_stats": {
            "all_rows":         len(all_rows),
            "skip_split":       skip_split,
            "skip_conflict":    skip_conflict,
            "skip_no_ytrue":    skip_no_ytrue,
            "skip_no_modality": skip_no_modality,
            "valid_pairs":      len(pairs),
            "class0_pairs":     y_cnt.get(0, 0),
            "class1_pairs":     y_cnt.get(1, 0),
        },
        "output_stats": {
            "saved_class0_inactive": saved_count[0],
            "saved_class1_active":   saved_count[1],
            "total_saved":           total_saved,
            "skipped_io":            skipped_io,
        },
        "class_info": {str(k): v for k, v in CLASS_INFO.items()},
    }

    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)

    print("=" * 80)
    print("✓ Finish!")
    print(f"  manifest : {manifest_path}")
    print(f"  audit    : {audit_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()