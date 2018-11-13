import tensorflow as tf
import h5py
import cv2
import numpy as np
from utils import utils
from models import cpm
from config import CONFIG
import os
import pck
import data_process
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

#  0 - r ankle, 1 - r knee, 2 - r hip, 3 - l hip, 4 - l knee, 5 - l ankle, 6 - pelvis, 7 - thorax, 8 - upper neck,
#  9 - head top, 10 - r wrist, 11 - r elbow, 12 - r shoulder, 13 - l shoulder, 14 - l elbow, 15 - l wrist

# 0-头顶，1-脖子，2-右肩，3-右肘，4-右腕，5-左肩，6-左肘，7-左腕，8-右胯，9-右膝，10-右踝，11-左胯，12-左膝，13-左踝

"""
'center', 'imgname','scale'
"""
IMG_DIR = "/home/administrator/diskb/PengXiao/code/convolutional-pose-machines-release/dataset/MPI/images/"
VALID_OUTPUT_DIR = "/home/administrator/diskb/PengXiao/code/convolutional-pose-machines-release/valid_output/"
VALID_CROP_DIR = "/home/administrator/diskb/PengXiao/code/convolutional-pose-machines-release/valid_crop/"
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_file', default_value='./dataset/valid.h5', docstring="data file path")
tf.app.flags.DEFINE_integer('input_size', default_value=368, docstring="input image size")
tf.app.flags.DEFINE_integer('hmap_size', default_value=46, docstring='heatmap size')
tf.app.flags.DEFINE_integer('cmap_variance', default_value=50, docstring='center map gaussian variance')
tf.app.flags.DEFINE_integer('joints', default_value=16, docstring='number of joints')
tf.app.flags.DEFINE_integer('stages', default_value=6, docstring="cpm stages number")
tf.app.flags.DEFINE_integer('batch_size', default_value=32, docstring="batch size")


def eval():
    input_data = tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.input_size, FLAGS.input_size, 3], name='input_image')
    center_map = tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.input_size, FLAGS.input_size, 1], name='center_map')  # center map的大小也是368x368

    model = cpm.Model(FLAGS.stages, FLAGS.joints)  # 这里的stages和joints是传进来的参数，跟训练集的关节点个数应该相等
    model.generate_model(input_data, center_map, FLAGS.batch_size)

    center_map = utils.generate_gaussian_map(FLAGS.input_size, FLAGS.input_size, FLAGS.input_size / 2,
                                             FLAGS.input_size / 2, FLAGS.cmap_variance)  # 预测的时候，需要把人放图像中间，这里构造center map是以图片中心来构造的，如果预测人没有在中间，就会出现问题

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # if FLAGS.model_path.endswith('pkl'):
        #     model.load_weights_from_file(FLAGS.model_path, sess, False)
        # else:
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(CONFIG.model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        #构造输入数据
        data = h5py.File(FLAGS.data_file, 'r')  # 打开h5文件
        # centers = data['center'], imgnames = data['imgname'], scales = data['scale'], gt_joints = data['part']
        total_size = len(data['index'])        # 验证集的大小
        mesures_rsts = []       #存储每个人的pck预测结果
        n = 0
        while n*FLAGS.batch_size < total_size:      #一次读一个batch
            i = n*FLAGS.batch_size
            j = min((n+1)*FLAGS.batch_size, total_size)
            n += 1
            centers = data['center'][i:j]
            img_names = data['imgname'][i:j]
            scales = data['scale'][i:j]
            batch_gt_joints = data['part'][i:j]

            #将人体放中心，crop到368x368，调整坐标
            imgs, joints_list = data_process.crop_to_center(img_names, FLAGS.input_size, scales, centers, batch_gt_joints, False)  # 输入图片，368x368，在中心，这里的joints_list是更新过的ground truth
            imgs_input = np.array(imgs) / 255.0 - 0.5                       #归一化到[-0.5,0.5]
            # imgs_input = np.expand_dims(imgs_input, axis=0)               #增加了batch的维数为1，此刻test_img_input： 1x368x368x3

            center_maps = np.array(list(center_map)*len(imgs_input))
            center_maps = np.reshape(center_maps, [len(imgs_input), FLAGS.input_size, FLAGS.input_size, 1])

            #inference
            pred_heatmaps = sess.run([model.stage_heatmaps[FLAGS.stages - 1]],        #最后一个stage的heatmap，列表长度为batch_size的值
                                        feed_dict={'input_image:0': imgs_input, 'center_map:0': center_maps})    #stage_heatmap_np[0]的shape为 [1, 46, 46, 15]
            pred_heatmaps = pred_heatmaps[0]    #得到的heatmap只有个stage的，长度为1，[0]之后的长度是32
            batch_pred_joints = []
            for i in range(len(pred_heatmaps)):
                pred_heatmap = pred_heatmaps[i, :, :, 0:FLAGS.joints].reshape((FLAGS.hmap_size, FLAGS.hmap_size, FLAGS.joints))  # 将有效的heatmap切出来，把背景排除，结果：46x46x14，每张图只有一个人
                pred_heatmap = cv2.resize(pred_heatmap, (FLAGS.input_size, FLAGS.input_size))  # heatmap变成了368x368x14
                preds_joint = np.zeros((FLAGS.joints, 2))  # 预测出的关节坐标
                for joint_idx in range(FLAGS.joints):
                    joint_coord = np.unravel_index(np.argmax(pred_heatmap[:, :, joint_idx]),(FLAGS.input_size, FLAGS.input_size))  # 求出关节点相对于368,368的坐标值
                    preds_joint[joint_idx, :] = joint_coord
                # 画出关节和肢体
                cv2.imwrite(os.path.join(VALID_CROP_DIR, img_names[i].decode('utf-8')), imgs[i].astype(np.uint8))   #画关节和肢体前
                utils.visualize(imgs[i], preds_joint, joints_list[i])       #image, 预测坐标，gt坐标
                cv2.imwrite(os.path.join(VALID_OUTPUT_DIR, img_names[i].decode('utf-8')), imgs[i].astype(np.uint8))
                batch_pred_joints.append(preds_joint)

            #pck
            mesures_rsts.extend(pck.compute_pck(12, 3, joints_list, batch_pred_joints, 0.2))   #2：右肩  11：左胯，对应的mpii是12和3
        acc_joints, acc_ave = pck.compute_pck_accuracy(mesures_rsts)

        print(" head top acc:    %.2f" % acc_joints[0] + "\n neck acc:    %.2f" % acc_joints[1] + "\n right shoulder acc:   %.2f" % acc_joints[2]+ \
              "\n right elbow acc:    %.2f" % acc_joints[3] + "\n right wrist acc:   %.2f" % acc_joints[4] + "\n left shoulder acc:   %.2f" % acc_joints[5] + \
              "\n left elbow acc:   %.2f" % acc_joints[6] + "\n left wrist acc:   %.2f" % acc_joints[7] + "\n right hip acc:   %.2f" % acc_joints[8]+ \
              "\n right knee acc:   %.2f" % acc_joints[9] + "\n right ankle acc:   %.2f" % acc_joints[10] + "\n left hip acc:   %.2f" % acc_joints[11] + \
              "\n left knee acc:   %.2f" % acc_joints[12] + "\n left ankle acc:   %.2f" % acc_joints[13] + "\n average acc:   %.2f" % acc_ave)
    return acc_ave



def main(argv):
    eval()

if __name__ == '__main__':
    tf.app.run()