import os
from typing import List

from detectron2.data import DatasetCatalog, MetadataCatalog

VOC_CLASSES: List[str] = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


def _load_voc_sem_seg(split_file: str, image_dir: str, mask_dir: str):
    with open(split_file, "r", encoding="utf-8") as handle:
        ids = [line.strip() for line in handle if line.strip()]
    return [
        {
            "file_name": os.path.join(image_dir, f"{image_id}.jpg"),
            "sem_seg_file_name": os.path.join(mask_dir, f"{image_id}.png"),
        }
        for image_id in ids
    ]


def register_voc2012(vocdevkit_root: str) -> None:
    voc_root = os.path.join(vocdevkit_root, "VOC2012")
    image_dir = os.path.join(voc_root, "JPEGImages")
    mask_dir = os.path.join(voc_root, "SegmentationClass")
    split_dir = os.path.join(voc_root, "ImageSets", "Segmentation")

    for split in ["train", "val"]:
        name = f"voc_2012_sem_seg_{split}"
        if name in DatasetCatalog.list():
            continue
        split_file = os.path.join(split_dir, f"{split}.txt")
        DatasetCatalog.register(
            name,
            lambda split_file=split_file: _load_voc_sem_seg(split_file, image_dir, mask_dir),
        )
        MetadataCatalog.get(name).set(
            stuff_classes=VOC_CLASSES,
            evaluator_type="sem_seg",
            ignore_label=255,
            image_root=image_dir,
            sem_seg_root=mask_dir,
        )


def maybe_register_voc2012(vocdevkit_root: str = None) -> None:
    root = vocdevkit_root or os.environ.get("VOC_ROOT", "VOCdevkit")
    if not os.path.isdir(os.path.join(root, "VOC2012")):
        return
    register_voc2012(root)
