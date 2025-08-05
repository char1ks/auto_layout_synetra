#!/usr/bin/env python3
import os
import argparse
import numpy as np
from PIL import Image

from mask_withsearch import (
    initialize_models,
    get_vector,
    adjust_embedding,
    annotate_image,
    annotate_image_v2,
)


def load_images_from_folder(folder: str):
    """Load all JPG/PNG images from a folder into a list of PIL.Image."""
    imgs = []
    for fn in sorted(os.listdir(folder)):
        if fn.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(folder, fn)
            imgs.append(Image.open(path).convert("RGB"))
    return imgs


def main():
    parser = argparse.ArgumentParser(
        description="Run SearchDet on a single image using local positive/negative exemplars"
    )
    parser.add_argument(
        "--example", "-i",
        required=True,
        help="Path to the target image you want annotated",
    )
    parser.add_argument(
        "--positive-dir", "-p",
        required=True,
        help="Folder of positive support images",
    )
    parser.add_argument(
        "--negative-dir", "-n",
        required=True,
        help="Folder of negative support images",
    )
    parser.add_argument(
        "--output", "-o",
        default="annotated_output.jpg",
        help="Where to save the annotated result",
    )
    parser.add_argument(
        "--conf-threshold", "-c",
        type=float,
        default=0.5,
        help="Minimum confidence score to keep a detection (default: 0.3)",
    )
    parser.add_argument(
        "--iou-threshold", "-u",
        type=float,
        default=0.5,
        help="IoU threshold for non-maximum suppression (default: 0.5)",
    )
    args = parser.parse_args()

    # 1) Init all models (ResNet, transforms, SAM)
    resnet_model, pooling_layer, transform, sam = initialize_models()

    # 2) Load images
    example_img = Image.open(args.example).convert("RGB")
    positive_imgs = load_images_from_folder(args.positive_dir)
    negative_imgs = load_images_from_folder(args.negative_dir)

    if len(positive_imgs) == 0 or len(negative_imgs) == 0:
        raise ValueError("Positive and negative folders must each contain ≥1 image.")

    # 3) Embed all support images
    pos_embs = np.stack([
        get_vector(img, resnet_model, pooling_layer, transform).numpy()
        for img in positive_imgs
    ], axis=0).astype(np.float32)

    neg_embs = np.stack([
        get_vector(img, resnet_model, pooling_layer, transform).numpy()
        for img in negative_imgs
    ], axis=0).astype(np.float32)

    # 4) Compute adjusted query vectors
    adjusted_queries = np.stack([
        adjust_embedding(q, pos_embs, neg_embs)
        for q in pos_embs
    ], axis=0).astype(np.float32)

    # 5) Annotate with thresholds
    annotate_image_v2(
        example_img,
        adjusted_queries,
        resnet_model,
        pooling_layer,
        transform,
        sam,
        args.output,
        iou_threshold=args.iou_threshold,
        min_area_threshold=6000,
        max_area_threshold=15000,
        similarity_thrsh=args.conf_threshold,
    )
    print(f"✅ Annotated image written to {args.output}")


if __name__ == "__main__":
    main()
