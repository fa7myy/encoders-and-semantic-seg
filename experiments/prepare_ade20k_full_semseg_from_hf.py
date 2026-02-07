#!/usr/bin/env python3
"""
Export the HuggingFace ADE20K dataset (e.g. "CSAILVision/ADE20K" or a local clone)
into the on-disk layout expected by Mask2Former/Detectron2 for ADE20K *full*
semantic segmentation.

Output layout (root = --out-root or $DETECTRON2_DATASETS):
  ADE20K_2021_17_01/
    images_detectron2/
      training/*.jpg
      validation/*.jpg
    annotations_detectron2/
      training/*.tif
      validation/*.tif

The GT files are single-channel uint16 TIFFs with:
  - class IDs in [0, 846] (847 classes)
  - ignore label = 65535

This matches Mask2Former dataset names:
  - ade20k_full_sem_seg_train  (dir: training)
  - ade20k_full_sem_seg_val    (dir: validation)
"""

import argparse
import io
import os
import shutil
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
from PIL import Image


IGNORE_LABEL = 65535
ADE20K_FULL_DIR = "ADE20K_2021_17_01"
REPO_ROOT = Path(__file__).resolve().parents[1]
FALLBACK_NAME_NDX_MAPPING = REPO_ROOT / "configs" / "ade20k_full_name_ndx_to_train_id.json"


def _maybe_tqdm(it: Iterable[Any], total: Optional[int] = None, desc: str = "") -> Iterable[Any]:
    try:
        from tqdm import tqdm  # type: ignore

        return tqdm(it, total=total, desc=desc)
    except Exception:
        return it


def _to_pil_image(x: Any) -> Image.Image:
    if isinstance(x, Image.Image):
        return x
    if isinstance(x, dict):
        if x.get("bytes") is not None:
            return Image.open(io.BytesIO(x["bytes"]))
        if x.get("path") is not None:
            return Image.open(x["path"])
    raise TypeError(f"Unsupported image payload type: {type(x)}")


def _write_jpeg(image_payload: Any, out_path: Path, quality: int) -> None:
    """
    Prefer preserving source bytes/paths (faster, no re-encode) when possible.
    Falls back to decoding and saving.
    """
    if isinstance(image_payload, dict):
        src_path = image_payload.get("path")
        src_bytes = image_payload.get("bytes")

        if src_path:
            src_path = str(src_path)
            if src_path.lower().endswith((".jpg", ".jpeg")) and out_path.suffix.lower() in {".jpg", ".jpeg"}:
                shutil.copyfile(src_path, str(out_path))
                return
        if src_bytes is not None and out_path.suffix.lower() in {".jpg", ".jpeg"}:
            with open(out_path, "wb") as handle:
                handle.write(src_bytes)
            return

    image = _to_pil_image(image_payload).convert("RGB")
    image.save(str(out_path), format="JPEG", quality=int(quality))


def _load_hf_dataset(hf_root_or_id: str):
    try:
        import datasets  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency: datasets. Install with `pip install datasets pyarrow`."
        ) from exc

    hf_path = Path(hf_root_or_id)
    if hf_path.exists() and hf_path.is_dir():
        data_dir = hf_path / "data"
        if data_dir.is_dir():
            data_files = {
                "train": str(data_dir / "train-*.parquet"),
                "validation": str(data_dir / "validation-*.parquet"),
            }
            return datasets.load_dataset("parquet", data_files=data_files)

    # HF Hub ID, or a directory without prebuilt parquet files.
    return datasets.load_dataset(hf_root_or_id)


def _load_ade20k_full_name_ndx_to_train_id() -> Dict[int, int]:
    # Reuse the exact list Mask2Former uses for "ade20k_full_sem_seg_{train,val}".
    try:
        from mask2former.data.datasets.register_ade20k_full import (  # type: ignore
            ADE20K_SEM_SEG_FULL_CATEGORIES,
        )
    except Exception as exc:  # pragma: no cover
        if FALLBACK_NAME_NDX_MAPPING.is_file():
            mapping_raw = json.loads(FALLBACK_NAME_NDX_MAPPING.read_text(encoding="utf-8"))
            mapping = {int(k): int(v) for k, v in mapping_raw.items()}
            if len(mapping) != 847:  # pragma: no cover
                raise RuntimeError(
                    f"Unexpected fallback mapping size: {len(mapping)} (expected 847) "
                    f"from {FALLBACK_NAME_NDX_MAPPING}"
                )
            return mapping

        raise RuntimeError(
            "Could not import Mask2Former’s ADE20K-full mapping and no fallback mapping file exists.\n"
            f"Tried fallback: {FALLBACK_NAME_NDX_MAPPING}\n"
            "Fix options:\n"
            "  1) Put Mask2Former (and detectron2) on PYTHONPATH, or\n"
            "  2) Add the mapping json to this repo."
        ) from exc

    mapping = {int(k["id"]): int(k["trainId"]) for k in ADE20K_SEM_SEG_FULL_CATEGORIES}
    if len(mapping) != 847:  # pragma: no cover
        raise RuntimeError(f"Unexpected ADE20K-full mapping size: {len(mapping)} (expected 847)")
    return mapping


def _seg_rgb_to_uint16(seg_rgb: Image.Image) -> np.ndarray:
    """
    Convert an RGB-encoded segmentation to a uint16 ID map using the first two channels.
    We later infer what this ID represents (object-id vs category-id) using the sample metadata.
    """
    arr = np.asarray(seg_rgb.convert("RGB"), dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:  # pragma: no cover
        raise ValueError(f"Expected RGB mask (H,W,3). Got shape={arr.shape}")
    return arr[..., 0].astype(np.uint16) | (arr[..., 1].astype(np.uint16) << 8)


def _auto_mask16_source(seg16: np.ndarray, objects: List[Mapping[str, Any]]) -> str:
    if not objects:
        return "name_ndx"

    obj_ids = set()
    name_ndx_ids = set()
    for obj in objects:
        if obj.get("id") is not None:
            obj_ids.add(int(obj["id"]))
        if obj.get("name_ndx") is not None:
            name_ndx_ids.add(int(obj["name_ndx"]))

    # seg16 is typically very low-cardinality (object regions + background), so np.unique is cheap.
    uniq = np.unique(seg16)
    hit_obj = sum(1 for v in uniq if int(v) in obj_ids)
    hit_name = sum(1 for v in uniq if int(v) in name_ndx_ids)
    if hit_obj > hit_name and hit_obj > 0:
        return "obj_id"
    return "name_ndx"


def _name_ndx_map_from_obj_id_map(
    obj_id_map: np.ndarray, objects: List[Mapping[str, Any]]
) -> np.ndarray:
    # Build a small LUT (uint16) from obj_id -> name_ndx.
    lut = np.zeros((1 << 16,), dtype=np.uint16)
    for obj in objects:
        if obj.get("id") is None or obj.get("name_ndx") is None:
            continue
        lut[int(obj["id"])] = int(obj["name_ndx"])
    return lut[obj_id_map]


def _train_id_map_from_name_ndx_map(
    name_ndx_map: np.ndarray, name_ndx_to_train_id: Mapping[int, int]
) -> np.ndarray:
    # Map ADE20K name_ndx IDs to contiguous train IDs [0..846]. Anything unknown becomes ignore.
    lut = np.full((1 << 16,), IGNORE_LABEL, dtype=np.uint16)
    for name_ndx, train_id in name_ndx_to_train_id.items():
        if 0 <= name_ndx < (1 << 16):
            lut[name_ndx] = np.uint16(train_id)
    return lut[name_ndx_map]


def _split_to_dirname(split: str) -> str:
    split = split.strip().lower()
    if split in {"train", "training"}:
        return "training"
    if split in {"val", "valid", "validation"}:
        return "validation"
    raise ValueError(f"Unsupported split: {split}")


def _split_to_hf_key(split: str) -> str:
    split = split.strip().lower()
    if split in {"train", "training"}:
        return "train"
    if split in {"val", "valid", "validation"}:
        return "validation"
    raise ValueError(f"Unsupported split: {split}")


def _resolve_filename(sample: Mapping[str, Any]) -> str:
    filename = sample.get("filename") or sample.get("source", {}).get("filename")
    if not filename:
        raise KeyError("Sample is missing 'filename'.")
    filename = os.path.basename(str(filename))
    if not filename.lower().endswith(".jpg"):
        # Detectron2's loader expects jpgs for ade20k_full.
        filename = str(Path(filename).with_suffix(".jpg"))
    return filename


def _inspect_example(
    sample: Mapping[str, Any], name_ndx_to_train_id: Mapping[int, int], mask16_source: str
) -> None:
    filename = _resolve_filename(sample)
    seg0 = _to_pil_image(sample["segmentations"][0])
    seg16 = _seg_rgb_to_uint16(seg0)
    objects = list(sample.get("objects") or [])

    auto_source = _auto_mask16_source(seg16, objects)
    source = auto_source if mask16_source == "auto" else mask16_source
    if source == "obj_id":
        name_ndx = _name_ndx_map_from_obj_id_map(seg16, objects)
    else:
        name_ndx = seg16
    train_ids = _train_id_map_from_name_ndx_map(name_ndx, name_ndx_to_train_id)

    uniq_seg16 = np.unique(seg16)
    uniq_name_ndx = np.unique(name_ndx)
    uniq_train = np.unique(train_ids)
    mapped_px = int(np.count_nonzero(train_ids != IGNORE_LABEL))
    total_px = int(train_ids.size)

    print("---- ADE20K example inspection ----")
    print(f"filename: {filename}")
    print(f"mask16_source: requested={mask16_source} auto={auto_source} used={source}")
    print(f"objects: {len(objects)}")
    print(
        "seg16: unique_count={} min={} max={}".format(
            uniq_seg16.size, int(uniq_seg16.min()), int(uniq_seg16.max())
        )
    )
    print(
        "name_ndx: unique_count={} min={} max={}".format(
            uniq_name_ndx.size, int(uniq_name_ndx.min()), int(uniq_name_ndx.max())
        )
    )
    print(
        "train_id: unique_count={} min={} max={} mapped_px={} ({:.2f}%)".format(
            uniq_train.size,
            int(uniq_train.min()),
            int(uniq_train.max()),
            mapped_px,
            100.0 * mapped_px / max(1, total_px),
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hf-root",
        required=True,
        help="Local path to the ADE20K HF dataset repo, or a HF Hub ID like 'CSAILVision/ADE20K'.",
    )
    parser.add_argument(
        "--out-root",
        default=os.getenv("DETECTRON2_DATASETS", "datasets"),
        help="Detectron2 dataset root. Defaults to $DETECTRON2_DATASETS or ./datasets.",
    )
    parser.add_argument(
        "--split",
        nargs="+",
        default=["train", "validation"],
        help="Which splits to export. Options: train, validation.",
    )
    parser.add_argument("--max-samples", type=int, default=0, help="Limit samples per split (0 = all).")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")
    parser.add_argument(
        "--mask16-source",
        choices=["auto", "obj_id", "name_ndx"],
        default="auto",
        help="How to interpret the uint16 IDs decoded from seg RGB (first 2 channels).",
    )
    parser.add_argument(
        "--tiff-compression",
        choices=["tiff_deflate", "tiff_lzw", "none"],
        default="tiff_deflate",
        help="Compression for output 16-bit TIFF labels (smaller extracted dataset).",
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Print a quick sanity check for the first sample of each split and exit.",
    )
    parser.add_argument("--jpeg-quality", type=int, default=95)
    args = parser.parse_args()

    name_ndx_to_train_id = _load_ade20k_full_name_ndx_to_train_id()
    ds = _load_hf_dataset(args.hf_root)

    out_root = Path(args.out_root).expanduser().resolve()
    out_base = out_root / ADE20K_FULL_DIR

    for split in args.split:
        hf_key = _split_to_hf_key(split)
        dirname = _split_to_dirname(split)
        if hf_key not in ds:
            raise KeyError(
                f"Split '{hf_key}' not found. Available splits: {list(ds.keys())}"
            )
        split_ds = ds[hf_key]

        if len(split_ds) == 0:
            print(f"[prepare_ade20k_full] Split '{split}' is empty, skipping.")
            continue

        if args.inspect_only:
            _inspect_example(split_ds[0], name_ndx_to_train_id, args.mask16_source)
            continue

        out_img_dir = out_base / "images_detectron2" / dirname
        out_gt_dir = out_base / "annotations_detectron2" / dirname
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_gt_dir.mkdir(parents=True, exist_ok=True)

        max_samples = int(args.max_samples or 0)
        n = min(len(split_ds), max_samples) if max_samples > 0 else len(split_ds)
        print(f"[prepare_ade20k_full] Exporting split={hf_key} -> {dirname} (n={n})")

        mapped_px_total = 0
        px_total = 0
        for idx in _maybe_tqdm(range(n), total=n, desc=f"export:{hf_key}"):
            sample = split_ds[int(idx)]

            filename = _resolve_filename(sample)
            img_out = out_img_dir / filename
            gt_out = out_gt_dir / (Path(filename).stem + ".tif")

            if not args.overwrite and img_out.exists() and gt_out.exists():
                continue

            _write_jpeg(sample["image"], img_out, quality=int(args.jpeg_quality))

            seg0 = _to_pil_image(sample["segmentations"][0])
            seg16 = _seg_rgb_to_uint16(seg0)
            objects = list(sample.get("objects") or [])

            source = args.mask16_source
            if source == "auto":
                source = _auto_mask16_source(seg16, objects)
            if source == "obj_id":
                name_ndx_map = _name_ndx_map_from_obj_id_map(seg16, objects)
            else:
                name_ndx_map = seg16

            train_id_map = _train_id_map_from_name_ndx_map(name_ndx_map, name_ndx_to_train_id)
            gt_img = Image.fromarray(train_id_map.astype(np.uint16))
            if args.tiff_compression != "none":
                gt_img.save(str(gt_out), compression=str(args.tiff_compression))
            else:
                gt_img.save(str(gt_out))

            mapped_px_total += int(np.count_nonzero(train_id_map != IGNORE_LABEL))
            px_total += int(train_id_map.size)

        pct = 100.0 * mapped_px_total / max(1, px_total)
        print(f"[prepare_ade20k_full] Done split={hf_key} mapped_px={mapped_px_total}/{px_total} ({pct:.2f}%)")

    if args.inspect_only:
        print("[prepare_ade20k_full] inspect-only complete (no files written).")
    else:
        print(f"[prepare_ade20k_full] Output root: {out_base}")
        print("[prepare_ade20k_full] Next: export DETECTRON2_DATASETS and train on ade20k_full_sem_seg_{train,val}.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
