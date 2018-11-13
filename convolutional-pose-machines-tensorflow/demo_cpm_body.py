import numpy as np
from utils import cpm_utils
import cv2
import time
import math
import sys
import os
import imageio
import tensorflow as tf
from models.nets import cpm_body

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

"""Parameters
"""
base_path="/home/administrator/diskc/pengxiao/convolutional-pose-machines-tensorflow"
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('DEMO_TYPE',
                           default_value='HM',
                           # default_value='test_imgs/single_gym.mp4',
                           # default_value='SINGLE',
                           docstring='MULTI: show multiple stage,'
                                     'SINGLE: only last stage,'
                                     'HM: show last stage heatmap,')
tf.app.flags.DEFINE_string('img_path',
                           default_value='test_imgs/golf.jpg',
                           docstring="Image to test")
tf.app.flags.DEFINE_string('model_path',
                           default_value='models/weights/cpm_body.pkl',
                           docstring='Your model')
tf.app.flags.DEFINE_integer('input_size',
                            default_value=368,
                            docstring='Input image size')
tf.app.flags.DEFINE_integer('hmap_size',
                            default_value=46,
                            docstring='Output heatmap size')
tf.app.flags.DEFINE_integer('cmap_radius',
                            default_value=40,
                            docstring='Center map gaussian variance')
tf.app.flags.DEFINE_integer('joints',
                            # default_value=14,
                            default_value=15,
                            docstring='Number of joints')
tf.app.flags.DEFINE_integer('stages',
                            default_value=6,
                            docstring='How many CPM stages')
# Set color for each finger
joint_color_code = [[139, 53, 255],
                    [0, 56, 255],
                    [43, 140, 237],
                    [37, 168, 36],
                    [147, 147, 0],
                    [70, 17, 145]]

limbs = [[0, 1],
         [2, 3],
         [3, 4],
         [5, 6],
         [6, 7],
         [8, 9],
         [9, 10],
         [11, 12],
         [12, 13]]

def mgray(test_img_resize, test_img):
    test_img_resize = np.dot(test_img_resize[..., :3], [0.299, 0.587, 0.114]).reshape(
                    (FLAGS.input_size, FLAGS.input_size, 1))
    # cv2.imshow('color', test_img.astype(np.uint8))
    # cv2.imshow('gray', test_img_resize.astype(np.uint8))
    cv2.imwrite(os.path.join(base_path, 'demo',  "color_.jpg"), test_img.astype(np.uint8))
    cv2.imwrite(os.path.join(base_path, 'demo',  "gray_.jpg"), test_img_resize.astype(np.uint8))
    cv2.waitKey(1)
    return test_img_resize


def main(argv):
    # tf_device = '/gpu:0'
    tf_device = '/cpu:0'
    with tf.device(tf_device):
        """Build graph
        """
        input_data = tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.input_size, FLAGS.input_size, 3], name='input_image')
        center_map = tf.placeholder(dtype=tf.float32,
                                    shape=[None, FLAGS.input_size, FLAGS.input_size, 1],    #center map的大小也是368x368
                                    name='center_map')

        # model = cpm_body.CPM_Model(FLAGS.stages, FLAGS.joints + 1)
        model = cpm_body.CPM_Model(FLAGS.stages, FLAGS.joints)      #这里的stages和joints是传进来的参数，跟训练集的关节点个数应该相等
        model.build_model(input_data, center_map, 1)

    saver = tf.train.Saver()

    """Create session and restore weights
    """
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess = tf.Session()

    sess.run(tf.global_variables_initializer())
    if FLAGS.model_path.endswith('pkl'):
        model.load_weights_from_file(FLAGS.model_path, sess, False)
    else:
        saver.restore(sess, FLAGS.model_path)

    test_center_map = cpm_utils.gaussian_img(FLAGS.input_size, FLAGS.input_size, FLAGS.input_size/2,  #预测的时候，需要把人放图像中间，这里构造center map是以图片中心来构造的
                                             FLAGS.input_size/2, FLAGS.cmap_radius)                   #如果预测人没有在中间，就会出现问题
    test_center_map = np.reshape(test_center_map, [1, FLAGS.input_size, FLAGS.input_size, 1])       #增加batch和通道的维数都为1，此刻center map：1x368x368x1

    # Check weights
    for variable in tf.trainable_variables():
        with tf.variable_scope('', reuse=True):
            var = tf.get_variable(variable.name.split(':0')[0])
            print(variable.name, np.mean(sess.run(var)))            #这个for循环输出了模型中的参数，包括卷积核值和偏置值

    with tf.device(tf_device):

        # while True:
        test_img_t = time.time()

        test_img = cpm_utils.read_image(FLAGS.img_path, [], FLAGS.input_size, 'IMAGE')          #构造预测输入，[]是视频输出，这里没有，test_img为368x368x3
        test_img_resize = cv2.resize(test_img, (FLAGS.input_size, FLAGS.input_size))        #删掉，没用
        print('img read time %f' % (time.time() - test_img_t))

        test_img_input = test_img_resize / 256.0 - 0.5          #归一化到[-0.5,0.5]
        test_img_input = np.expand_dims(test_img_input, axis=0)     #增加了batch的维数为1，此刻test_img_input： 1x368x368x3

        if FLAGS.DEMO_TYPE == 'MULTI':

            # Inference
            fps_t = time.time()
            predict_heatmap, stage_heatmap_np = sess.run([model.current_heatmap,
                                                          model.stage_heatmap,
                                                          ],
                                                         feed_dict={'input_image:0': test_img_input,
                                                                    'center_map:0': test_center_map})

            # Show visualized image
            demo_img = visualize_result(test_img, FLAGS, stage_heatmap_np, None)
            # cv2.imshow('demo_img', demo_img.astype(np.uint8))
            cv2.imwrite(os.path.join(base_path, 'demo', str(time.time())+"_.jpg"), demo_img.astype(np.uint8))
            # if cv2.waitKey(1) == ord('q'): break
            print('fps: %.2f' % (1 / (time.time() - fps_t)))

        elif FLAGS.DEMO_TYPE == 'SINGLE':

            # Inference
            fps_t = time.time()
            stage_heatmap_np = sess.run([model.stage_heatmap[5]],
                                        feed_dict={'input_image:0': test_img_input,
                                                   'center_map:0': test_center_map})

            # Show visualized image
            demo_img = visualize_result(test_img, FLAGS, stage_heatmap_np, None)
            # cv2.imshow('current heatmap', (demo_img).astype(np.uint8))
            cv2.imwrite(os.path.join(base_path, 'demo', "current_heatmap.jpg"), (demo_img).astype(np.uint8))
            # if cv2.waitKey(1) == ord('q'): break
            print('fps: %.2f' % (1 / (time.time() - fps_t)))

        elif FLAGS.DEMO_TYPE == 'HM':

            # Inference
            fps_t = time.time()
            stage_heatmap_np = sess.run([model.stage_heatmap[FLAGS.stages - 1]],        #最后一个stage的heatmap，列表长度为1，因为只有一个人
                                        feed_dict={'input_image:0': test_img_input,     #stage_heatmap_np[0]的shape为 [1, 46, 46, 15]
                                                   'center_map:0': test_center_map})
            print('fps: %.2f' % (1 / (time.time() - fps_t)))

            # demo_stage_heatmap = stage_heatmap_np[len(stage_heatmap_np) - 1][0, :, :, 0:FLAGS.joints].reshape(
            #     (FLAGS.hmap_size, FLAGS.hmap_size, FLAGS.joints))
            demo_stage_heatmap = stage_heatmap_np[-1][0, :, :, 0:FLAGS.joints].reshape(     #将有效的heatmap切出来，把背景排除，结果：46x46x14
                (FLAGS.hmap_size, FLAGS.hmap_size, FLAGS.joints))
            demo_stage_heatmap = cv2.resize(demo_stage_heatmap, (FLAGS.input_size, FLAGS.input_size))  #heatmap变成了368x368x14

            vertical_imgs = []          #下面在拼接heatmap的展示图片
            tmp_img = None
            joint_coord_set = np.zeros((FLAGS.joints, 2))

            for joint_num in range(FLAGS.joints):
                # Concat until 4 img
                if (joint_num % 4) == 0 and joint_num != 0:
                    vertical_imgs.append(tmp_img)
                    tmp_img = None

                demo_stage_heatmap[:, :, joint_num] *= (255 / np.max(demo_stage_heatmap[:, :, joint_num]))      #将heatmap每层的最大值变成255，共有14层

                # Plot color joints
                if np.min(demo_stage_heatmap[:, :, joint_num]) > -50:
                    joint_coord = np.unravel_index(np.argmax(demo_stage_heatmap[:, :, joint_num]),
                                                   (FLAGS.input_size, FLAGS.input_size))
                    joint_coord_set[joint_num, :] = joint_coord
                    color_code_num = (joint_num // 4)

                    if joint_num in [0, 4, 8, 12, 16]:
                        joint_color = list(map(lambda x: x + 35 * (joint_num % 4), joint_color_code[color_code_num]))

                        cv2.circle(test_img, center=(joint_coord[1], joint_coord[0]), radius=3, color=joint_color,
                                   thickness=-1)
                    else:
                        joint_color = list(map(lambda x: x + 35 * (joint_num % 4), joint_color_code[color_code_num]))

                        cv2.circle(test_img, center=(joint_coord[1], joint_coord[0]), radius=3, color=joint_color,
                                   thickness=-1)

                # Put text
                tmp = demo_stage_heatmap[:, :, joint_num].astype(np.uint8)
                tmp = cv2.putText(tmp, 'Min:' + str(np.min(demo_stage_heatmap[:, :, joint_num])),
                                  org=(5, 20), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.3, color=150)
                tmp = cv2.putText(tmp, 'Mean:' + str(np.mean(demo_stage_heatmap[:, :, joint_num])),
                                  org=(5, 30), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.3, color=150)
                tmp_img = np.concatenate((tmp_img, tmp), axis=0) \
                    if tmp_img is not None else tmp

            # Plot limbs
            for limb_num in range(len(limbs)):
                if np.min(demo_stage_heatmap[:, :, limbs[limb_num][0]]) > -2000 and np.min(
                        demo_stage_heatmap[:, :, limbs[limb_num][1]]) > -2000:
                    x1 = joint_coord_set[limbs[limb_num][0], 0]         #limbs中存储了关节之间的连接关系，两个节点
                    y1 = joint_coord_set[limbs[limb_num][0], 1]
                    x2 = joint_coord_set[limbs[limb_num][1], 0]
                    y2 = joint_coord_set[limbs[limb_num][1], 1]
                    length = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5   #两个关节的距离，肢体的距离
                    if length < 10000 and length > 5:
                        deg = math.degrees(math.atan2(x1 - x2, y1 - y2))    #肢体的角度
                        polygon = cv2.ellipse2Poly((int((y1 + y2) / 2), int((x1 + x2) / 2)), (int(length / 2), 3),
                                                   int(deg), 0, 360, 1)     #画出肢体的椭圆形状，以两关节的中点为中心，椭圆的a值为肢体长度一半，b值自己定，还要给出椭圆的倾斜角度，也就是上面求出的角度
                        color_code_num = limb_num // 4      #给当前肢体选颜色
                        limb_color = list(map(lambda x: x + 35 * (limb_num % 4), joint_color_code[color_code_num]))

                        cv2.fillConvexPoly(test_img, polygon, color=limb_color)     #test_img是我们的测试输入图片 (368,368,3)

            if tmp_img is not None:     #上面已经将肢体的椭圆连线画在了测试输入图片上
                tmp_img = np.lib.pad(tmp_img, ((0, vertical_imgs[0].shape[0] - tmp_img.shape[0]), (0, 0)),
                                     'constant', constant_values=(0, 0))    #在test_img的上面(为0)和下面做padding，padding的值为0，这段可以删掉
                vertical_imgs.append(tmp_img)

            # Concat horizontally
            output_img = None       #输出的图片
            for col in range(len(vertical_imgs)):
                output_img = np.concatenate((output_img, vertical_imgs[col]), axis=1) if output_img is not None else \
                    vertical_imgs[col]

            output_img = output_img.astype(np.uint8)
            output_img = cv2.applyColorMap(output_img, cv2.COLORMAP_JET)
            test_img = cv2.resize(test_img, (300, 300), cv2.INTER_LANCZOS4)
            # cv2.imshow('hm', output_img)
            cv2.imwrite(os.path.join(base_path, 'demo', str(time.time())+"_.jpg"), output_img.astype(np.uint8))
            # cv2.moveWindow('hm', 2000, 200)
            # cv2.imshow('rgb', test_img)
            cv2.imwrite(os.path.join(base_path, 'demo', str(time.time())+"_.jpg"), test_img.astype(np.uint8))
            # cv2.moveWindow('rgb', 2000, 750)
            # if cv2.waitKey(1) == ord('q'): break


def visualize_result(test_img, FLAGS, stage_heatmap_np, kalman_filter_array):
    hm_t = time.time()
    demo_stage_heatmaps = []
    if FLAGS.DEMO_TYPE == 'MULTI':
        for stage in range(len(stage_heatmap_np)):
            demo_stage_heatmap = stage_heatmap_np[stage][0, :, :, 0:FLAGS.joints].reshape(
                (FLAGS.hmap_size, FLAGS.hmap_size, FLAGS.joints))
            demo_stage_heatmap = cv2.resize(demo_stage_heatmap, (test_img.shape[1], test_img.shape[0]))
            demo_stage_heatmap = np.amax(demo_stage_heatmap, axis=2)
            demo_stage_heatmap = np.reshape(demo_stage_heatmap, (test_img.shape[1], test_img.shape[0], 1))
            demo_stage_heatmap = np.repeat(demo_stage_heatmap, 3, axis=2)
            demo_stage_heatmap *= 255
            demo_stage_heatmaps.append(demo_stage_heatmap)

        # last_heatmap = stage_heatmap_np[len(stage_heatmap_np) - 1][0, :, :, 0:FLAGS.joints].reshape(
        #     (FLAGS.hmap_size, FLAGS.hmap_size, FLAGS.joints))
        last_heatmap = stage_heatmap_np[-1][0, :, :, 0:FLAGS.joints].reshape(
            (FLAGS.hmap_size, FLAGS.hmap_size, FLAGS.joints))
        last_heatmap = cv2.resize(last_heatmap, (test_img.shape[1], test_img.shape[0]))
    else:
        # last_heatmap = stage_heatmap_np[len(stage_heatmap_np) - 1][0, :, :, 0:FLAGS.joints].reshape(
        #     (FLAGS.hmap_size, FLAGS.hmap_size, FLAGS.joints))
        last_heatmap = stage_heatmap_np[-1][0, :, :, 0:FLAGS.joints].reshape(
            (FLAGS.hmap_size, FLAGS.hmap_size, FLAGS.joints))
        last_heatmap = cv2.resize(last_heatmap, (test_img.shape[1], test_img.shape[0]))
    print('hm resize time %f' % (time.time() - hm_t))

    joint_t = time.time()
    joint_coord_set = np.zeros((FLAGS.joints, 2))

    for joint_num in range(FLAGS.joints):
        joint_coord = np.unravel_index(np.argmax(last_heatmap[:, :, joint_num]),
                                       (test_img.shape[0], test_img.shape[1]))
        joint_coord_set[joint_num, :] = [joint_coord[0], joint_coord[1]]

        color_code_num = (joint_num // 4)
        if joint_num in [0, 4, 8, 12, 16]:
            joint_color = list(map(lambda x: x + 35 * (joint_num % 4), joint_color_code[color_code_num]))
            cv2.circle(test_img, center=(joint_coord[1], joint_coord[0]), radius=3, color=joint_color, thickness=-1)
        else:
            joint_color = list(map(lambda x: x + 35 * (joint_num % 4), joint_color_code[color_code_num]))
            cv2.circle(test_img, center=(joint_coord[1], joint_coord[0]), radius=3, color=joint_color, thickness=-1)
    print('plot joint time %f' % (time.time() - joint_t))

    limb_t = time.time()

    # Plot limb colors
    for limb_num in range(len(limbs)):
        x1 = joint_coord_set[limbs[limb_num][0], 0]
        y1 = joint_coord_set[limbs[limb_num][0], 1]
        x2 = joint_coord_set[limbs[limb_num][1], 0]
        y2 = joint_coord_set[limbs[limb_num][1], 1]
        length = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        if length < 200 and length > 5:
            deg = math.degrees(math.atan2(x1 - x2, y1 - y2))
            polygon = cv2.ellipse2Poly((int((y1 + y2) / 2), int((x1 + x2) / 2)),
                                       (int(length / 2), 6),
                                       int(deg),
                                       0, 360, 1)
            color_code_num = limb_num // 4
            limb_color = list(map(lambda x: x + 35 * (limb_num % 4), joint_color_code[color_code_num]))
            cv2.fillConvexPoly(test_img, polygon, color=limb_color)
    print('plot limb time %f' % (time.time() - limb_t))

    if FLAGS.DEMO_TYPE == 'MULTI':
        upper_img = np.concatenate((demo_stage_heatmaps[0], demo_stage_heatmaps[1], demo_stage_heatmaps[2]), axis=1)
        lower_img = np.concatenate((demo_stage_heatmaps[3], demo_stage_heatmaps[len(stage_heatmap_np) - 1], test_img),
                                   axis=1)
        demo_img = np.concatenate((upper_img, lower_img), axis=0)
        return demo_img
    else:
        return test_img


if __name__ == '__main__':
    tf.app.run()
