# New_Build_Multimodal_224x448_Manifest.py
#
# 功能：
#   從 master_manifest.csv 讀取 train_valid 與 test，
#   各自在 split 內部進行 row-based OCT–OCTA3 上下配對，
#   輸出 224×448 RGB multimodal image，完全禁止跨 split 混用。
#
# 配對邏輯（row-based）：
#   對每一筆 exam unit（同 exam_key = 同病人、同眼、同日期、同 label）：
#     - 無 label_conflict
#     - 同時有 OCTA3（os.path.isfile 驗證）
#     - 有 OCT0 或 OCT1（os.path.isfile 驗證，至少一張）
#     → 從 OCT0 / OCT1 隨機取一張，記錄 chosen_oct_type
#     → 與同 row OCTA3 上下拼接（OCT 上 / OCTA3 下）→ 224×448 RGB
#
# 修正項目（本版新增）：
#   ★  OCT0 / OCT1 完整分開紀錄：
#      manifest 欄位新增 src_oct0_path / src_oct1_path / chosen_oct_type
#   ★  audit 新增 OCT0 / OCT1 分 split 統計：
#      原始有幾個 exam 有 OCT0 / OCT1、pairing 實際選用 OCT0 / OCT1 各幾張
#      涉及的病人數全部分開統計
#   其餘修正延續前版：
#      CLEAR_OUTPUT / Path.relative_to() / written_rows 統計 /
#      檔名格式 / exam_key fallback / patient overlap 檢查
#
# 執行：
#   python New_Build_Multimodal_224x448_Manifest.py
#
# 輸出結構：
#   BASE_OUTPUT/
#       train_valid/class_0_inactive/  {exam_key}_{counter:06d}_inactive.jpg
#       train_valid/class_1_active/    {exam_key}_{counter:06d}_active.jpg
#       test/class_0_inactive/         ...
#       test/class_1_active/           ...
#       multimodal_manifest.csv
#       build_audit.json

import os
import csv
import json
import random
import shutil
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

IMAGE_ROOT  = ""
BASE_OUTPUT = "/data/Irene/Multimodal_224x448"

IMAGE_SIZE  = 224
COMBINED_W  = IMAGE_SIZE
COMBINED_H  = IMAGE_SIZE * 2   # 448

TARGET_SPLITS = ["train_valid", "test"]
SPLIT_DIR_MAP = {"train_valid": "train_valid", "test": "test"}

RANDOM_SEED  = 42
CLEAR_OUTPUT = True

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
    if not p or str(p).strip() in ("", "nan", "None"):
        return ""
    p = str(p).strip()
    if os.path.isabs(p):
        return p
    if IMAGE_ROOT:
        return os.path.join(IMAGE_ROOT, p)
    return p


def safe_for_filename(s: str) -> str:
    for ch in ("/", "\\", " ", ":", "*", "?", '"', "<", ">", "|"):
        s = s.replace(ch, "_")
    return s


def build_filename_key(pair: dict) -> str:
    """
    檔名格式：{exam_key}_{counter:06d}_{class_name}.jpg
    exam_key 為空時 fallback 為 {patient_id}{eye}_{exam_date}。
    """
    safe_ek = safe_for_filename(str(pair["exam_key"]).strip())
    if not safe_ek:
        safe_ek = safe_for_filename(
            f"{pair['patient_id']}{pair['eye']}_{pair['exam_date']}"
        )
    return safe_ek


def read_master_manifest(csv_path: str) -> list:
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def make_combined_image(oct_path: str, octa_path: str):
    """224×224 OCT + 224×224 OCTA3 → 224×448 RGB。失敗回傳 None。"""
    try:
        img_oct  = Image.open(oct_path).convert("RGB").resize(
            (IMAGE_SIZE, IMAGE_SIZE), _RESAMPLE
        )
        img_octa = Image.open(octa_path).convert("RGB").resize(
            (IMAGE_SIZE, IMAGE_SIZE), _RESAMPLE
        )
    except Exception as e:
        print(f"  [WARN] 讀圖失敗: {oct_path} | {octa_path} | {e}")
        return None

    combo = Image.new("RGB", (COMBINED_W, COMBINED_H))
    combo.paste(img_oct,  (0, 0))
    combo.paste(img_octa, (0, IMAGE_SIZE))
    return combo


def filter_and_pair_rows(all_rows: list, target_split: str):
    """
    對指定 split 逐列篩選並建立配對清單。
    ★ OCT0 / OCT1 分開記錄：
       _oct_candidates 元素為 ("OCT0"|"OCT1", path) tuple。
       src_oct0_path / src_oct1_path 分別存原始路徑。
    回傳 (pairs, skip_stats, modality_raw_stats)。
    """
    skip_split              = 0
    skip_conflict           = 0
    skip_no_ytrue           = 0
    skip_no_oct             = 0
    skip_no_octa3           = 0
    skip_missing_oct_file   = 0
    skip_missing_octa3_file = 0

    # ★ 原始影像存在性統計（在 os.path.isfile 驗證後）
    raw_oct0_exist_count    = 0   # 有 OCT0 flag 且檔案存在的 row 數
    raw_oct1_exist_count    = 0   # 有 OCT1 flag 且檔案存在的 row 數
    raw_octa3_exist_count   = 0   # 有 OCTA3 flag 且檔案存在的 row 數
    raw_oct0_patients       = set()
    raw_oct1_patients       = set()
    raw_octa3_patients      = set()

    pairs = []

    for r in all_rows:
        if str(r.get("split_set", "")).strip() != target_split:
            skip_split += 1
            continue

        if safe_int(r.get("label_conflict", 0), 0) != 0:
            skip_conflict += 1
            continue

        ytrue_int = safe_int(r.get("y_true", ""), -1)
        if ytrue_int not in CLASS_INFO:
            skip_no_ytrue += 1
            continue

        pid = str(r.get("patient_id", ""))

        # ── OCT0 ─────────────────────────────────────────────────────────────
        oct_candidates   = []   # list of ("OCT0"|"OCT1", path)
        has_any_oct_flag = False
        src_oct0_path    = ""
        src_oct1_path    = ""

        if safe_int(r.get("has_oct0", 0), 0):
            has_any_oct_flag = True
            p = resolve_path(r.get("oct0_image_path", ""))
            if p and os.path.isfile(p):
                src_oct0_path = p
                oct_candidates.append(("OCT0", p))
                raw_oct0_exist_count += 1
                raw_oct0_patients.add(pid)
            else:
                skip_missing_oct_file += 1

        # ── OCT1 ─────────────────────────────────────────────────────────────
        if safe_int(r.get("has_oct1", 0), 0):
            has_any_oct_flag = True
            p = resolve_path(r.get("oct1_image_path", ""))
            if p and os.path.isfile(p):
                src_oct1_path = p
                oct_candidates.append(("OCT1", p))
                raw_oct1_exist_count += 1
                raw_oct1_patients.add(pid)
            else:
                skip_missing_oct_file += 1

        if not has_any_oct_flag:
            skip_no_oct += 1

        # ── OCTA3 ─────────────────────────────────────────────────────────────
        octa3_path = ""
        if safe_int(r.get("has_octa3", 0), 0):
            p = resolve_path(r.get("octa3_image_path", ""))
            if p and os.path.isfile(p):
                octa3_path = p
                raw_octa3_exist_count += 1
                raw_octa3_patients.add(pid)
            else:
                skip_missing_octa3_file += 1
        else:
            skip_no_octa3 += 1

        if not oct_candidates or not octa3_path:
            continue

        pairs.append({
            "exam_key":        str(r.get("exam_key",   "")),
            "patient_id":      pid,
            "eye":             str(r.get("eye",         "")),
            "exam_date":       str(r.get("exam_date",   "")),
            "split_set":       target_split,
            "fold_id":         str(r.get("fold_id",     "")),
            "y_true":          ytrue_int,
            "class_name":      CLASS_INFO[ytrue_int]["name"],
            "label_conflict":  0,
            "src_oct0_path":   src_oct0_path,   # ★ 空 = 該 exam 無有效 OCT0
            "src_oct1_path":   src_oct1_path,   # ★ 空 = 該 exam 無有效 OCT1
            "_oct_candidates": oct_candidates,  # ★ [("OCT0"|"OCT1", path), ...]
            "_octa3_path":     octa3_path,
        })

    skip_stats = {
        "skip_other_split":              skip_split,
        "skip_label_conflict":           skip_conflict,
        "skip_ytrue_invalid":            skip_no_ytrue,
        "skip_no_oct_flag":              skip_no_oct,
        "skip_no_octa3_flag":            skip_no_octa3,
        "skip_missing_oct_file_count":   skip_missing_oct_file,
        "skip_missing_octa3_file_count": skip_missing_octa3_file,
    }

    # ★ 原始影像統計（掃描所有 target_split rows 後得到）
    modality_raw_stats = {
        "rows_with_valid_oct0_file":     raw_oct0_exist_count,
        "rows_with_valid_oct1_file":     raw_oct1_exist_count,
        "rows_with_valid_octa3_file":    raw_octa3_exist_count,
        "patients_with_valid_oct0_file": len(raw_oct0_patients),
        "patients_with_valid_oct1_file": len(raw_oct1_patients),
        "patients_with_valid_octa3_file": len(raw_octa3_patients),
    }

    return pairs, skip_stats, modality_raw_stats


def compute_split_stats(rows_for_stats: list, split: str) -> dict:
    """
    根據 written_rows（實際寫入 manifest 的 row）計算統計，
    確保 audit 與 multimodal_manifest.csv 完全一致。
    """
    y_cnt  = Counter(int(r["y_true"]) for r in rows_for_stats)
    all_p  = set(r["patient_id"] for r in rows_for_stats)
    cls0_p = set(r["patient_id"] for r in rows_for_stats if int(r["y_true"]) == 0)
    cls1_p = set(r["patient_id"] for r in rows_for_stats if int(r["y_true"]) == 1)

    # ★ OCT0 / OCT1 pairing 使用統計（基於 written_rows）
    used_oct0_p = set(r["patient_id"] for r in rows_for_stats
                      if r.get("chosen_oct_type") == "OCT0")
    used_oct1_p = set(r["patient_id"] for r in rows_for_stats
                      if r.get("chosen_oct_type") == "OCT1")
    chose_oct0  = sum(1 for r in rows_for_stats if r.get("chosen_oct_type") == "OCT0")
    chose_oct1  = sum(1 for r in rows_for_stats if r.get("chosen_oct_type") == "OCT1")

    # ★ 有 OCT0 或 OCT1 原始影像的病人（不管選了哪個）
    has_oct0_p  = set(r["patient_id"] for r in rows_for_stats
                      if r.get("src_oct0_path", ""))
    has_oct1_p  = set(r["patient_id"] for r in rows_for_stats
                      if r.get("src_oct1_path", ""))
    both_oct_p  = has_oct0_p & has_oct1_p   # 同時有 OCT0 & OCT1 的病人

    fold_stats = {}
    if split == "train_valid":
        for fold_id in sorted(set(r["fold_id"] for r in rows_for_stats)):
            fp = [r for r in rows_for_stats if r["fold_id"] == fold_id]
            fold_stats[fold_id] = {
                "num_samples":     len(fp),
                "num_patients":    len(set(r["patient_id"] for r in fp)),
                "class0_samples":  sum(1 for r in fp if int(r["y_true"]) == 0),
                "class1_samples":  sum(1 for r in fp if int(r["y_true"]) == 1),
                "class0_patients": len(set(r["patient_id"] for r in fp
                                          if int(r["y_true"]) == 0)),
                "class1_patients": len(set(r["patient_id"] for r in fp
                                          if int(r["y_true"]) == 1)),
                "chosen_oct0":     sum(1 for r in fp
                                       if r.get("chosen_oct_type") == "OCT0"),
                "chosen_oct1":     sum(1 for r in fp
                                       if r.get("chosen_oct_type") == "OCT1"),
            }

    return {
        "patient_stats": {
            "num_patients_total":           len(all_p),
            "num_patients_class0_inactive": len(cls0_p),
            "num_patients_class1_active":   len(cls1_p),
        },
        "oct_pairing_stats": {
            # ── 原始影像有 OCT0 / OCT1 的情況（基於 written_rows）
            "pairs_with_oct0_available":    sum(1 for r in rows_for_stats
                                               if r.get("src_oct0_path", "")),
            "pairs_with_oct1_available":    sum(1 for r in rows_for_stats
                                               if r.get("src_oct1_path", "")),
            "pairs_with_both_oct_available": sum(
                1 for r in rows_for_stats
                if r.get("src_oct0_path", "") and r.get("src_oct1_path", "")
            ),
            "patients_with_oct0_available": len(has_oct0_p),
            "patients_with_oct1_available": len(has_oct1_p),
            "patients_with_both_oct_available": len(both_oct_p),
            # ── 實際被選入配對的情況
            "chosen_oct0_count":            chose_oct0,
            "chosen_oct1_count":            chose_oct1,
            "patients_chosen_oct0":         len(used_oct0_p),
            "patients_chosen_oct1":         len(used_oct1_p),
        },
        "fold_patient_stats": fold_stats,
    }


# ===================== Main =====================

def main():
    random.seed(RANDOM_SEED)
    out_root = Path(BASE_OUTPUT)

    # ── CLEAR_OUTPUT ─────────────────────────────────────────────────────────
    if CLEAR_OUTPUT and out_root.exists():
        print("[0] CLEAR_OUTPUT=True：清空舊輸出 ...")
        for split in TARGET_SPLITS:
            split_dir = out_root / SPLIT_DIR_MAP[split]
            if split_dir.exists():
                shutil.rmtree(split_dir)
                print(f"    removed: {split_dir}")
        for fname in ["multimodal_manifest.csv", "build_audit.json"]:
            fp = out_root / fname
            if fp.exists():
                fp.unlink()
                print(f"    removed: {fp}")

    for split in TARGET_SPLITS:
        for info in CLASS_INFO.values():
            (out_root / SPLIT_DIR_MAP[split] / info["dir"]).mkdir(
                parents=True, exist_ok=True
            )

    manifest_path = out_root / "multimodal_manifest.csv"
    audit_path    = out_root / "build_audit.json"

    print("=" * 80)
    print("New_Build_Multimodal_224x448_Manifest  [row-based, train_valid + test]")
    print(f"  Master manifest : {MASTER_MANIFEST_CSV}")
    print(f"  Output root     : {out_root}")
    print(f"  CLEAR_OUTPUT    : {CLEAR_OUTPUT}")
    print("=" * 80)

    # ── 1. 讀取 master manifest ──────────────────────────────────────────────
    all_rows = read_master_manifest(MASTER_MANIFEST_CSV)
    print(f"[1] 讀取 master manifest：{len(all_rows)} 筆")

    # ── 2. 各 split 獨立篩選 ─────────────────────────────────────────────────
    split_pairs          = {}
    split_skip_stats     = {}
    split_modality_raw   = {}   # ★ 各 split 的原始影像統計

    for split in TARGET_SPLITS:
        print(f"\n[2] 篩選 [{split}] ...")
        pairs, skip_stats, raw_stats = filter_and_pair_rows(all_rows, split)
        split_pairs[split]        = pairs
        split_skip_stats[split]   = skip_stats
        split_modality_raw[split] = raw_stats

        y_cnt = Counter(p["y_true"] for p in pairs)
        print(f"    有效配對數（檔案已驗證）          : {len(pairs)}")
        print(f"    class 0 (inactive)               : {y_cnt.get(0, 0)}")
        print(f"    class 1 (active)                 : {y_cnt.get(1, 0)}")
        print(f"    ── skip 統計 ──")
        for k, v in skip_stats.items():
            print(f"    {k:<44}: {v}")
        print(f"    ── 原始影像（os.path.isfile 驗證後）──")
        for k, v in raw_stats.items():
            print(f"    {k:<44}: {v}")

    # ── patient overlap 檢查 ─────────────────────────────────────────────────
    print("\n[3] 檢查 train_valid / test patient overlap ...")
    train_pids   = set(p["patient_id"] for p in split_pairs["train_valid"])
    test_pids    = set(p["patient_id"] for p in split_pairs["test"])
    overlap_pids = sorted(train_pids & test_pids)
    if overlap_pids:
        raise RuntimeError(
            f"[ERROR] train_valid 與 test 有 {len(overlap_pids)} 位病人重疊！"
            f" 前 10 位：{overlap_pids[:10]}"
        )
    print(f"    ✓ 無病人重疊（train_valid={len(train_pids)} 人，"
          f"test={len(test_pids)} 人）")

    # ── 4. 合併影像 + 輸出 manifest ──────────────────────────────────────────
    print("\n[4] 合併影像並輸出 manifest ...")

    manifest_fieldnames = [
        "exam_key", "patient_id", "eye", "exam_date",
        "split_set", "fold_id", "y_true", "class_name", "label_conflict",
        "has_oct0", "has_oct1", "has_octa3",
        "oct0_image_path",    # ← 合併後 multimodal 影像路徑（訓練讀此欄）
        "oct1_image_path",    # ← 空
        "octa3_image_path",   # ← 空
        # ★ OCT 追蹤欄位
        "src_oct0_path",      # ← 原始 OCT0 路徑（空 = 無有效 OCT0）
        "src_oct1_path",      # ← 原始 OCT1 路徑（空 = 無有效 OCT1）
        "chosen_oct_type",    # ← "OCT0" 或 "OCT1"（實際被選入配對的）
        "src_oct_path",       # ← 實際使用的 OCT 路徑（OOF 分析用）
        "src_octa3_path",     # ← 原始 OCTA3 路徑（OOF 分析用）
    ]

    saved_count            = {s: Counter() for s in TARGET_SPLITS}
    skipped_io             = {s: 0 for s in TARGET_SPLITS}
    written_split_counter  = Counter()
    written_rows           = []   # 供 written_rows 統計 & path check

    # ★ per-split OCT type counter（audit 用）
    chosen_oct_type_counter = {s: Counter() for s in TARGET_SPLITS}

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=manifest_fieldnames)
        wr.writeheader()

        for split in TARGET_SPLITS:
            pairs       = split_pairs[split]
            split_dir   = SPLIT_DIR_MAP[split]
            cls_counter = Counter()

            for pair in tqdm(pairs, desc=f"  [{split}] Saving"):
                yi = pair["y_true"]

                # ★ 選 OCT，同時取出 type
                chosen_oct_type, chosen_oct = random.choice(pair["_oct_candidates"])
                chosen_octa3                = pair["_octa3_path"]

                cls_counter[yi] += 1
                safe_ek = build_filename_key(pair)
                fname   = (
                    f"{safe_ek}"
                    f"_{cls_counter[yi]:06d}"
                    f"_{CLASS_INFO[yi]['name']}.jpg"
                )
                out_path = out_root / split_dir / CLASS_INFO[yi]["dir"] / fname

                combo = make_combined_image(chosen_oct, chosen_octa3)
                if combo is None:
                    skipped_io[split] += 1
                    continue

                combo.save(str(out_path), quality=95)
                saved_count[split][yi] += 1
                written_split_counter[split] += 1
                chosen_oct_type_counter[split][chosen_oct_type] += 1   # ★

                row = {
                    "exam_key":         pair["exam_key"],
                    "patient_id":       pair["patient_id"],
                    "eye":              pair["eye"],
                    "exam_date":        pair["exam_date"],
                    "split_set":        split,
                    "fold_id":          pair["fold_id"],
                    "y_true":           pair["y_true"],
                    "class_name":       pair["class_name"],
                    "label_conflict":   0,
                    "has_oct0":         1,
                    "has_oct1":         0,
                    "has_octa3":        0,
                    "oct0_image_path":  str(out_path),
                    "oct1_image_path":  "",
                    "octa3_image_path": "",
                    # ★ OCT 完整追蹤欄位
                    "src_oct0_path":   pair["src_oct0_path"],
                    "src_oct1_path":   pair["src_oct1_path"],
                    "chosen_oct_type": chosen_oct_type,
                    "src_oct_path":    chosen_oct,
                    "src_octa3_path":  chosen_octa3,
                }
                wr.writerow(row)
                written_rows.append(row)

    # ── Path.relative_to() 嚴格路徑驗證 ─────────────────────────────────────
    print("\n[5] 驗證 oct0_image_path 路徑與 split_set 一致性 ...")
    cross_contamination_examples = []
    for row in written_rows:
        split        = row["split_set"]
        path         = row["oct0_image_path"]
        expected_dir = Path(out_root / SPLIT_DIR_MAP[split]).resolve()
        try:
            Path(path).resolve().relative_to(expected_dir)
        except ValueError:
            cross_contamination_examples.append({
                "split_set":       split,
                "oct0_image_path": path,
                "expected_dir":    str(expected_dir),
            })

    cross_contamination_found = len(cross_contamination_examples) > 0
    if cross_contamination_found:
        raise RuntimeError(
            f"[ERROR] 偵測到路徑與 split 不一致！"
            f" 共 {len(cross_contamination_examples)} 筆。"
            f" 前 3 筆：{cross_contamination_examples[:3]}"
        )
    print(f"    ✓ 所有 {len(written_rows)} 筆路徑均位於正確 split 目錄")

    # ── written_rows 統計 ─────────────────────────────────────────────────────
    print("\n[6] 根據 written_rows 計算 patient / fold / OCT 統計 ...")
    audit_split_stats = {}
    for split in TARGET_SPLITS:
        rows_for_stats = [r for r in written_rows if r["split_set"] == split]
        stats          = compute_split_stats(rows_for_stats, split)

        audit_split_stats[split] = {
            "input_stats": {
                **split_skip_stats[split],
                "valid_pairs_both_files_verified": len(split_pairs[split]),
            },
            # ★ 原始影像存在性統計（掃描所有 target_split rows）
            "modality_raw_stats": split_modality_raw[split],
            "output_stats": {
                "saved_class0_inactive": saved_count[split][0],
                "saved_class1_active":   saved_count[split][1],
                "total_saved":           saved_count[split][0] + saved_count[split][1],
                "manifest_rows_written": written_split_counter[split],
                "skipped_io":            skipped_io[split],
                "image_dir":             str(out_root / SPLIT_DIR_MAP[split]),
            },
            "patient_stats":      stats["patient_stats"],
            # ★ OCT0 / OCT1 完整配對統計（基於 written_rows）
            "oct_pairing_stats":  stats["oct_pairing_stats"],
            # ★ per-split OCT type 選用次數
            "chosen_oct_type_counts": {
                "OCT0": chosen_oct_type_counter[split].get("OCT0", 0),
                "OCT1": chosen_oct_type_counter[split].get("OCT1", 0),
            },
            "fold_patient_stats": stats["fold_patient_stats"],
        }

        # terminal 顯示
        ps  = stats["patient_stats"]
        ocs = stats["oct_pairing_stats"]
        print(f"\n  [{split}]")
        print(f"    patients total={ps['num_patients_total']}, "
              f"cls0={ps['num_patients_class0_inactive']}, "
              f"cls1={ps['num_patients_class1_active']}")
        print(f"    OCT0 available in paired rows : "
              f"{ocs['pairs_with_oct0_available']} pairs / "
              f"{ocs['patients_with_oct0_available']} patients")
        print(f"    OCT1 available in paired rows : "
              f"{ocs['pairs_with_oct1_available']} pairs / "
              f"{ocs['patients_with_oct1_available']} patients")
        print(f"    both OCT0+OCT1 available      : "
              f"{ocs['pairs_with_both_oct_available']} pairs / "
              f"{ocs['patients_with_both_oct_available']} patients")
        print(f"    chosen OCT0 : {ocs['chosen_oct0_count']} pairs / "
              f"{ocs['patients_chosen_oct0']} patients")
        print(f"    chosen OCT1 : {ocs['chosen_oct1_count']} pairs / "
              f"{ocs['patients_chosen_oct1']} patients")
        if split == "train_valid":
            for fid, fs in stats["fold_patient_stats"].items():
                print(f"    fold {fid}: samples={fs['num_samples']}, "
                      f"patients={fs['num_patients']}, "
                      f"cls0={fs['class0_samples']}(n={fs['class0_patients']}), "
                      f"cls1={fs['class1_samples']}(n={fs['class1_patients']}), "
                      f"oct0={fs['chosen_oct0']}, oct1={fs['chosen_oct1']}")

    # ── 寫入 audit JSON ───────────────────────────────────────────────────────
    total_all = sum(saved_count[s][0] + saved_count[s][1] for s in TARGET_SPLITS)

    audit = {
        "time":             datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "master_manifest":  MASTER_MANIFEST_CSV,
        "output_root":      str(out_root),
        "manifest_path":    str(manifest_path),
        "target_splits":    TARGET_SPLITS,
        "random_seed":      RANDOM_SEED,
        "clear_output":     CLEAR_OUTPUT,

        "pairing_strategy": (
            "row-based: per exam_key, "
            "random.choice(OCT0/OCT1 where os.path.isfile) → chosen_oct_type recorded; "
            "paired with OCTA3 from same exam row; "
            "train_valid and test processed and stored separately"
        ),

        "image_geometry": {
            "single_image_size": IMAGE_SIZE,
            "combined_width":    COMBINED_W,
            "combined_height":   COMBINED_H,
            "layout":            "OCT top (rows 0-223) / OCTA3 bottom (rows 224-447)",
            "color_mode":        "RGB (3-ch, compatible with Swin pretrained weights)",
        },

        "filename_format": (
            "{split_dir}/{class_dir}/"
            "{exam_key}_{counter:06d}_{class_name}.jpg"
        ),
        "filename_note": (
            "exam_key fallback: {patient_id}{eye}_{exam_date} if exam_key is empty"
        ),

        "manifest_new_columns_note": {
            "src_oct0_path":   "original OCT0 path if valid file exists, else empty",
            "src_oct1_path":   "original OCT1 path if valid file exists, else empty",
            "chosen_oct_type": "OCT0 or OCT1 — which was randomly selected for pairing",
            "src_oct_path":    "path of the actually chosen OCT (same as src_oct0 or src_oct1)",
            "src_octa3_path":  "original OCTA3 path",
        },

        "total_images_all_splits": total_all,
        "per_split":               audit_split_stats,

        "split_check": {
            "written_split_counts":          dict(written_split_counter),
            "train_valid_image_dir":         str(out_root / SPLIT_DIR_MAP["train_valid"]),
            "test_image_dir":                str(out_root / SPLIT_DIR_MAP["test"]),
            "dirs_are_distinct":             (
                SPLIT_DIR_MAP["train_valid"] != SPLIT_DIR_MAP["test"]
            ),
            "path_prefix_check_passed":      not cross_contamination_found,
            "cross_contamination_found":     cross_contamination_found,
            "cross_contamination_examples":  cross_contamination_examples[:20],
            "train_test_patient_overlap": {
                "num_overlap_patients": len(overlap_pids),
                "overlap_patient_ids":  overlap_pids[:20],
            },
        },

        "class_info":       {str(k): v for k, v in CLASS_INFO.items()},
        "manifest_columns": manifest_fieldnames,
    }

    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)

    # ── 最終摘要 ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("✓ 完成")
    print(f"  manifest : {manifest_path}")
    print(f"  audit    : {audit_path}")
    print()
    print(f"  {'split':<13} {'saved':>7}  {'inactive':>9}  {'active':>7}  "
          f"{'patients':>9}  {'oct0':>6}  {'oct1':>6}  {'IO_fail':>8}")
    print(f"  {'-' * 75}")
    for split in TARGET_SPLITS:
        s0  = saved_count[split][0]
        s1  = saved_count[split][1]
        np_ = audit_split_stats[split]["patient_stats"]["num_patients_total"]
        io_ = skipped_io[split]
        o0  = chosen_oct_type_counter[split].get("OCT0", 0)
        o1  = chosen_oct_type_counter[split].get("OCT1", 0)
        print(f"  {split:<13} {s0+s1:>7}  {s0:>9}  {s1:>7}  "
              f"{np_:>9}  {o0:>6}  {o1:>6}  {io_:>8}")
    print(f"  {'-' * 75}")
    print(f"  {'TOTAL':<13} {total_all:>7}")
    print()
    print(f"  path_prefix_check_passed   : {not cross_contamination_found}  ← 應為 True")
    print(f"  cross_contamination_found  : {cross_contamination_found}  ← 應為 False")
    print(f"  train_test_patient_overlap : {len(overlap_pids)} 人  ← 應為 0")
    print("=" * 80)


if __name__ == "__main__":
    main()