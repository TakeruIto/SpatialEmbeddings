"""
Author: Davy Neven
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import glob
import os
from multiprocessing import Pool

import numpy as np
from PIL import Image
from tqdm import tqdm


def process(tup,i):
    im, inst = tup

    image_path = os.path.splitext(os.path.relpath(im, os.path.join(IMAGE_DIR, 'train')))[0]
    image_path = os.path.join(IMAGE_DIR, 'crops_all1024', image_path)
    instance_path = os.path.splitext(os.path.relpath(inst, os.path.join(INSTANCE_DIR, 'train')))[0]
    instance_path = os.path.join(INSTANCE_DIR, 'crops_all1024', instance_path)

    try:  # can't use 'exists' because of threads
        os.makedirs(os.path.dirname(image_path))
        os.makedirs(os.path.dirname(instance_path))
    except FileExistsError:
        pass

    image = Image.open(im)
    instance = Image.open(inst)
    w, h = image.size

    instance_np = np.array(instance, copy=False)
    object_mask = np.logical_and(instance_np >= i * 1000, instance_np < (i + 1) * 1000)
    if np.sum(object_mask)>0:
        ids = np.unique(instance_np[object_mask])
        ids = ids[ids!= 0]

        # loop over instances
        for j, id in enumerate(ids):
        
            y, x = np.where(instance_np == id)
            ym, xm = np.mean(y), np.mean(x)
        
            ii = int(np.clip(ym-CROP_SIZE/2, 0, h-CROP_SIZE))
            jj = int(np.clip(xm-CROP_SIZE/2, 0, w-CROP_SIZE))

            im_crop = image.crop((jj, ii, jj + CROP_SIZE, ii + CROP_SIZE))
            instance_crop = instance.crop((jj, ii, jj + CROP_SIZE, ii + CROP_SIZE))

            im_crop.save(image_path + "_{:02d}_{:03d}.png".format(i,j))
            instance_crop.save(instance_path + "_{:02d}_{:03d}.png".format(i,j))
        return len(ids)
    else:
        return 0


if __name__ == '__main__':
    # cityscapes dataset
    CITYSCAPES_DIR="/gs/hs0/tga-shinoda/16B01730/data/gtFine_trainvaltest"

    IMAGE_DIR=os.path.join(CITYSCAPES_DIR, 'leftImg8bit')
    INSTANCE_DIR=os.path.join(CITYSCAPES_DIR, 'gtFine')
    OBJ_ID = 24
    CROP_SIZE=1024

    # load images/instances
    images = glob.glob(os.path.join(IMAGE_DIR, 'train', '*/*.png'))
    images.sort()
    instances = glob.glob(os.path.join(INSTANCE_DIR, 'train', '*/*instanceIds.png'))
    instances.sort()

#    with Pool(8) as p:
#        r = list(tqdm(p.imap(process, zip(images,instances)), total=len(images)))
    for i in [24,25,26,27,28,31,32,33]:
        zipped = list(zip(images, instances))
        np.random.shuffle(zipped)
        X_result, y_result = zip(*zipped)
        num = 0
        for x,y in zip(X_result,y_result):
            num += process((x,y),i)
            print(num)
            if num>=len(images)//8:
                break
