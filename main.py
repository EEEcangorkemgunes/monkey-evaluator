import numpy as np
import cv2
import rasterio
from rasterio.windows import Window
from model_configuration import Predictor


tif_image_path = "../monkey-model/images/pas-cpg/A_P000001_PAS_CPG.tif"
mask_image_path = "../monkey-model/images/tissue-masks/A_P000001_mask.tif"


patch_size = 1024
# with rasterio.open(mask_image_path) as img:
#     window = Window(0, 0, patch_size, patch_size)
#     img_patch = img.read(window=window)
#     print(img_patch)


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
            print(predictions)
            instances = predictions["instances"].get_fields()
            for i, pred_box in enumerate(instances["pred_boxes"]):
                results.append([int((pred_box[2] + pred_box[0]) / 2), int((pred_box[3] + pred_box[1]) / 2)])
            # for instance in instances:
            #     print(instance)
            # for prediction in predictions["instances"].pred_boxes.tensor.detach().cpu().numpy():
            #     results.append([int((prediction[2] + prediction[0]) / 2), int((prediction[3] + prediction[1]) / 2)])
            # print(predictions["instances"].scores)
            exit()
