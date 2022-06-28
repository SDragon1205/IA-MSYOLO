import sys, cv2
from absl import app, flags, logging
from absl.flags import FLAGS
import tensorflow as tf
from core.dataset_tiny import Dataset, tfDataset, tfAdversailDataset
import numpy as np

flags.DEFINE_string('model', 'yolov4', 'yolov4, yolov3')
flags.DEFINE_string('weights', None, 'pretrained weights')
flags.DEFINE_boolean('tiny', True, 'yolo or yolo-tiny')
flags.DEFINE_boolean('qat', False, 'train w/ or w/o quatize aware')
flags.DEFINE_string('save_dir', 'checkpoints/yolov4_tiny', 'save model dir')
tf.config.optimizer.set_jit(True)

def main(_argv):
    # TENSORFLOW HW 2
    trainset_target = tfAdversailDataset(FLAGS, is_training=True, filter_area=123, use_imgaug=True).dataset_gen()
    for i, a in enumerate(trainset_target):
        imgs=a['images']
        print(imgs.shape)
        imgs=(imgs.numpy()*255).astype(np.uint8)
        output1=np.concatenate([imgs[0], imgs[1]], axis=1)
        output2=np.concatenate([imgs[2], imgs[3]], axis=1)
        output =np.concatenate([output1, output2], axis=0)

        cv2.imwrite(f'scripts/test_{i}.png', output)
        if i == 4: break
    exit()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
