import argparse
import base64
import importlib.util
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import openai
from langchain.schema import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

# --------------------------- Agent Prompt ---------------------------
AGENT_PROMPT_TEMPLATE = """
You are CV‑Coder‑GPT, an expert classical‑computer‑vision engineer.
Goal: given reference information and task description, WRITE a
Python module (pure OpenCV + NumPy, no deep learning) that exposes

    def segment_and_detect(image_path: str) -> list[dict[str, Any]]

Each returned dict MUST contain:
    "mask"      : np.ndarray (binary 0/1, uint8, same H×W as image)
    "bbox"      : List[int]  [x, y, w, h]  (axis‑aligned)
    "polygon"   : List[List[int]]  [[x1,y1], ...]
    "class_name": str

* Use any classical CV techniques: color‑space conversion (HSV/LAB),
  histogram equalization, smoothing, morphological ops, edge detection,
  contour analysis, or a combination.
* Parameterize thresholds so the script is robust to lighting changes.
* NO external ML/DL libraries.
* Do **not** include evaluation code, file IO beyond reading image with
  cv2.imread, or CLI code – just the algorithm.
* Output ONLY the code, enclosed in a single Markdown block like:

```python
# code here ...
```

Nothing else.
"""

# ---------------------------------------------------------------------


def encode_image_to_base64(path: Path) -> str:
    """Read an image file and return a base64 string."""
    with path.open("rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def call_llm(
    chat: ChatOpenAI,
    reference_img_b64: str,
    instructions: str,
    feedback: str = "",
) -> str:
    """Send prompt + optional feedback to the LLM and return raw response."""
    messages = [
        SystemMessage(content=AGENT_PROMPT_TEMPLATE),
        HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": f"Task description:\n{instructions}\n\n{feedback}\n"
                            f"Here is the reference image (use it for reasoning):",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{reference_img_b64}"},
                },
            ]
        ),
    ]
    response = chat.invoke(messages)
    return response.content


CODE_BLOCK_RE = re.compile(r"```python(.*?)```", re.S)


def extract_code(raw_llm_output: str) -> str:
    """Extract the python code block from the LLM output."""
    m = CODE_BLOCK_RE.search(raw_llm_output)
    if not m:
        raise ValueError("No python code block found in LLM output.")
    return m.group(1).strip() + "\n"


def write_module(code: str, path: Path) -> None:
    """Write code to file and ensure it is syntactically valid."""
    path.write_text(code, encoding="utf-8")
    compile(code, filename=str(path), mode="exec")


def import_segmenter(module_path: Path):
    """Dynamically import the generated algorithm module."""
    spec = importlib.util.spec_from_file_location("generated_algo", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["generated_algo"] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "segment_and_detect"):
        raise AttributeError("Generated module lacks 'segment_and_detect'.")
    return module.segment_and_detect


# ------------------------ Metrics utilities -------------------------

def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Pixel IoU between two binary masks (0/1 uint8)."""
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return 0.0 if union == 0 else intersection / union


def simple_ap(
    image_paths: List[Path],
    pred_dict: Dict[str, List[Dict[str, Any]]],
    gt_dir: Path,
    iou_thresh: float,
) -> float:
    """
    Minimal AP@IoU implementation assuming ONE class and at most ONE
    ground‑truth object per image (extend as needed).
    """
    tp, fp, fn = 0, 0, 0
    for img_path in image_paths:
        gt_mask_path = gt_dir / (img_path.stem + "_mask.png")
        if not gt_mask_path.exists():
            continue
        gt_mask = cv2.imread(str(gt_mask_path), cv2.IMREAD_GRAYSCALE)
        gt_mask = (gt_mask > 127).astype(np.uint8)
        preds = pred_dict.get(img_path.name, [])
        if len(preds) == 0:
            fn += 1
            continue
        best_iou = max(compute_iou(p["mask"], gt_mask) for p in preds)
        if best_iou >= iou_thresh:
            tp += 1
            fp += len(preds) - 1  # other preds are FP
        else:
            fp += len(preds)
            fn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    # A very rough “AP” proxy (F‑measure)
    return precision * recall


# --------------------------- Main loop ------------------------------

def run_agent(
    reference_img: Path,
    images_dir: Path,
    gt_dir: Path,
    instructions: str,
    output_dir: Path,
    max_iters: int,
    iou_thresh: float,
    ap_target: float,
    model_name: str = "gpt-4o-mini",
):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if openai.api_key is None:
        raise EnvironmentError("Set OPENAI_API_KEY environment variable.")

    chat = ChatOpenAI(model=model_name, temperature=0.2)
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_b64 = encode_image_to_base64(reference_img)
    feedback = ""
    best_ap = 0.0

    for iter_id in range(1, max_iters + 1):
        print(f"\n=== Iteration {iter_id} ===")
        llm_response = call_llm(chat, reference_b64, instructions, feedback)
        try:
            code = extract_code(llm_response)
        except ValueError:
            print("LLM failed to produce code – retrying with feedback.")
            feedback = "⚠️  No valid code block detected. Please output *only* code."
            continue

        module_path = output_dir / f"generated_algo_iter{iter_id}.py"
        try:
            write_module(code, module_path)
            segment_and_detect = import_segmenter(module_path)
        except Exception as e:
            feedback = (
                f"⚠️  Generated code raised an error on import/compile:\n{e}\n"
                "Please fix and resend."
            )
            print(feedback)
            continue

        # Run generated algorithm on all scene images
        image_paths = sorted(
            [p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".png"}]
        )
        predictions: Dict[str, List[Dict[str, Any]]] = {}

        for img_path in image_paths:
            try:
                preds = segment_and_detect(str(img_path))
                for p in preds:
                    p["mask"] = (p["mask"] > 0).astype(np.uint8)
                predictions[img_path.name] = preds
            except Exception as e:
                feedback = (
                    f"⚠️  segment_and_detect crashed on {img_path.name}: {e}\n"
                    "Please correct the code."
                )
                print(feedback)
                break
        else:
            # All images processed successfully
            if gt_dir.exists():
                ap = simple_ap(image_paths, predictions, gt_dir, iou_thresh)
                best_ap = max(best_ap, ap)
                print(f"➡️  AP@{iou_thresh:.2f} = {ap:.3f}")
                if ap >= ap_target:
                    print("✅  Target achieved. Stopping iterations.")
                    break
                feedback = (
                    f"Current AP = {ap:.3f} (target {ap_target}).\n"
                    "Suggestions:\n"
                    "- Increase recall by broadening thresholds.\n"
                    "- Remove small FP blobs via area filtering.\n"
                    "- Combine edge & color masks."
                )
            else:
                print("⚠️  Ground‑truth directory not found – skipping metrics.")
                break

    print(f"\nBest AP achieved: {best_ap:.3f}")
    print(f"Final algorithm saved as: {module_path}")


# ---------------------------- CLI ------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LangChain CV Agent – classical segmentation/detection generator"
    )
    parser.add_argument("--reference", type=Path, required=True, help="Reference image")
    parser.add_argument(
        "--images_dir", type=Path, required=True, help="Directory with scene images"
    )
    parser.add_argument(
        "--ground_truth_dir",
        type=Path,
        required=False,
        default=None,
        help="Directory with GT masks (_mask.png). Optional.",
    )
    parser.add_argument(
        "--instructions",
        type=str,
        required=True,
        help='Natural-language task description (wrap in quotes if it contains spaces)',
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Where generated code and temp files will be stored",
    )
    parser.add_argument("--max_iters", type=int, default=5, help="Max refinement steps")
    parser.add_argument(
        "--iou_thresh", type=float, default=0.5, help="IoU threshold for AP"
    )
    parser.add_argument(
        "--ap_target",
        type=float,
        default=0.5,
        help="Stop when AP@IoU ≥ this value (0‑1)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model name (vision-capable)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_agent(
        reference_img=args.reference,
        images_dir=args.images_dir,
        gt_dir=args.ground_truth_dir if args.ground_truth_dir else Path("."),
        instructions=args.instructions,
        output_dir=args.output_dir,
        max_iters=args.max_iters,
        iou_thresh=args.iou_thresh,
        ap_target=args.ap_target,
        model_name=args.model,
    )
