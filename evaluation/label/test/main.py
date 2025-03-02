import os
import time
from glob import glob
import cv2
import numpy as np

from EdgeFenceSeg import EdgeFenceSeg

if __name__ == "__main__":
    imgs_path = "test_img"
    output_path = "output_cv"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    edg = EdgeFenceSeg()
    image_files = sorted(glob(os.path.join(imgs_path, f'*.{"png"}')))
    for img_file in image_files:
        t0 = time.time()
        img = cv2.imread(img_file)

        # inference
        t1 = time.time()
        mask_img, prediction, flag = edg.predict(img)
        t2 = time.time()
        print("{0:50}: Inference time: {1},   Is edge_fence: {2}".format(img_file, round(t2 - t1, 4), flag))

        # save masked img
        if mask_img is not None:
            image_file = os.path.basename(img_file).split('.')[0]
            # mask_img_cv = cv2.cvtColor(np.asarray(mask_img), cv2.COLOR_RGB2BGR)
            mask_img_cv = np.asarray(mask_img)
            cv2.imwrite(os.path.join(output_path, image_file + '.png'), mask_img_cv)