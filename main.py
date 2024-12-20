import numpy as np
import cv2
import rasterio
from rasterio.windows import Window
from model_configuration import Predictor
import json
from pixel_to_mm import pixel_to_mm
import os

image_name = "A_P000002"
tif_image_path = f"../monkey-model/images/pas-cpg/{image_name}_PAS_CPG.tif"
mask_image_path = f"../monkey-model/images/tissue-masks/{image_name}_mask.tif"

output_path = f"/home/can/Desktop/Python/monkey-froc/input/{image_name}/output"
os.makedirs(output_path, exist_ok=True)
patch_size = 1024

monocytes = {
    "name": "monocytes",
    "type": "Multiple points",
    "points": [],
    "version": {"major": 1, "minor": 0}
}
lymphocytes = {
    "name": "lymphocytes",
    "type": "Multiple points",
    "points": [],
    "version": {"major": 1, "minor": 0}
}
inflammatory = {
    "name": "inflammatory-cells",
    "type": "Multiple points",
    "points": [],
    "version": {"major": 1, "minor": 0}
}
monocyte_id = 1
lymphocyte_id = 1
with rasterio.open(tif_image_path) as img, rasterio.open(mask_image_path) as mask:
    width, height = img.width, img.height
    print(width, height)
    results = []
    predictor = Predictor()

    for y in range(0, height, patch_size):
        for x in range(0, width, patch_size):
            patch_x = min(patch_size, width - x)
            patch_y = min(patch_size, height - y)
            window = Window(x, y, patch_x, patch_y)
            mask_patch = mask.read(window=window)
            if np.all(mask_patch == 0):
                continue
            img_patch = img.read(window=window)
            masked_patch = mask_patch * img_patch
            image_hwc = np.transpose(masked_patch, (1, 2, 0))
            image_hwc = cv2.cvtColor(image_hwc, cv2.COLOR_RGB2BGR)
            cv2.imwrite("patch.png", image_hwc)
            predictions = predictor(image_hwc)
            # print(predictions)
            instances = predictions["instances"].get_fields()
            for i, pred_box in enumerate(instances["pred_boxes"]):
                if instances["pred_classes"][i] == 0:
                    lymphocytes["points"].append({"name": f"Point {lymphocyte_id}", "point": [
                        round(pixel_to_mm(float(pred_box[2] + pred_box[0]) / 2 + x), 4), round(pixel_to_mm(float(pred_box[3] + pred_box[1]) / 2 + y), 4), 0.25], "probability": round(float(instances["scores"][i]), 2)})
                    lymphocyte_id += 1
                else:
                    monocytes["points"].append({"name": f"Point {monocyte_id}", "point": [
                        round(pixel_to_mm(float(pred_box[2] + pred_box[0]) / 2 + x), 4), round(pixel_to_mm(float(pred_box[3] + pred_box[1]) / 2 + y), 4), 0.25], "probability": round(float(instances["scores"][i]), 2)})
                    monocyte_id += 1
    with open(f"{output_path}/detected-lymphocytes.json", "w") as f:
        json.dump(lymphocytes, f, indent=2)
    with open(f"{output_path}/detected-monocytes.json", "w") as f:
        json.dump(monocytes, f, indent=2)
    
    for monocyte in monocytes["points"]:
        inflammatory["points"].append(monocyte)
    for lymphocyte in lymphocytes["points"]:
        lymphocyte["name"] = f"Point {monocyte_id}"
        monocyte_id += 1
        inflammatory["points"].append(lymphocyte)
    with open(f"{output_path}/detected-inflammatory-cells.json", "w") as f:
        json.dump(inflammatory, f, indent=2)