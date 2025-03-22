import fiftyone as fo
import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset("coco-2017", split="validation", max_samples=100)

results = dataset.evaluate_detections(
    "predictions", gt_field="ground_truth", method="coco"
)

print("mAP Score:", results.mAP())
