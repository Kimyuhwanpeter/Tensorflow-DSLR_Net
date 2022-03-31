# -*- coding:utf-8 -*-
from DSLR_model import *
from random import random

import matplotlib.pyplot as plt
import numpy as np
import easydict
import os

FLAGS = easydict.EasyDict({"img_size": 256,

                           "tr_img_path": "D:/[1]DB/[5]4th_paper_DB/crop_weed/CropWeed Field Image Dataset (CWFID)/dataset-1.0/low_light/",

                           "tr_lab_path": "D:/[1]DB/[5]4th_paper_DB/crop_weed/CropWeed Field Image Dataset (CWFID)/dataset-1.0/aug_train_images/",
                           
                           "tr_txt_path": "D:/[1]DB/[5]4th_paper_DB/crop_weed/CropWeed Field Image Dataset (CWFID)/dataset-1.0/train_fix.txt",
                           
                           "batch_size": 4,
                           
                           "epochs": 50,
                           
                           "lr": 0.0002,

                           "train": False,

                           "sample_images": "C:/Users/Yuhwan/Downloads/sample_images",
                           
                           "save_checkpoint": "C:/Users/Yuhwan/Downloads/checkpoint",

                           "pre_checkpoint": True,

                           "pre_checkpoint_path": "C:/Users/Yuhwan/Downloads/checkpoint",
                           
                           "te_img_path": "D:/[1]DB/[5]4th_paper_DB/crop_weed/CropWeed Field Image Dataset (CWFID)/dataset-1.0/low_light/",
                           
                           "test_images": "D:/[1]DB/[5]4th_paper_DB/crop_weed/CropWeed Field Image Dataset (CWFID)/dataset-1.0/restored_low_light_DSLR"})

scale1_loss = tf.keras.losses.MeanSquaredError()
scale2_loss = tf.keras.losses.MeanSquaredError()
scale3_loss = tf.keras.losses.MeanSquaredError()
laplace_loss2 = tf.keras.losses.MeanAbsoluteError()
laplace_loss3 = tf.keras.losses.MeanAbsoluteError()
scale1_color = tf.keras.losses.CosineSimilarity()
scale2_color = tf.keras.losses.CosineSimilarity()
scale3_color = tf.keras.losses.CosineSimilarity()

optim1 = tf.keras.optimizers.Adam(FLAGS.lr, beta_1=0.5)
optim2 = tf.keras.optimizers.Adam(FLAGS.lr, beta_1=0.5)
optim3 = tf.keras.optimizers.Adam(FLAGS.lr, beta_1=0.5)

def tr_func(img_data, lab_data):

    img = tf.io.read_file(img_data)
    img = tf.image.decode_png(img, 3)
    img = tf.image.resize(img, [FLAGS.img_size, FLAGS.img_size])
    #img = tf.image.per_image_standardization(img)
    img = img / 127.5 - 1.

    lab = tf.io.read_file(lab_data)
    lab = tf.image.decode_png(lab, 3)
    lab = tf.image.resize(lab, [FLAGS.img_size, FLAGS.img_size])
    #lab = tf.image.per_image_standardization(lab)
    lab = lab / 127.5 - 1.

    if random() > 0.5:
        img = tf.image.flip_left_right(img)
        lab = tf.image.flip_left_right(lab)

    return img, lab

def te_func(img_data):

    img = tf.io.read_file(img_data)
    img = tf.image.decode_png(img, 3)
    img = tf.image.resize(img, [FLAGS.img_size, FLAGS.img_size])
    #img = tf.image.per_image_standardization(img)
    img = img / 127.5 - 1.

    return img

def cal_loss(Stage1, Stage2, Stage3, images, labels):

    with tf.GradientTape() as stage1, tf.GradientTape() as stage2, tf.GradientTape() as stage3:

        x_down2 = tf.image.resize(images, [int(images.shape[1] / 2), int(images.shape[2] / 2)])
        x_down4 = tf.image.resize(x_down2, [int(x_down2.shape[1] / 2), int(x_down2.shape[2] / 2)])

        x_reup2 = tf.image.resize(x_down4, [x_down4.shape[1]*2, x_down4.shape[2]*2])
        x_reup = tf.image.resize(x_down2, [x_down2.shape[1]*2, x_down2.shape[2]*2])

        Laplace_2 = x_down2 - x_reup2
        Laplace_1 = images - x_reup

        Scale1 = Stage1(x_down4, True)
        Scale2 = Stage2(Laplace_2, True)
        Scale3 = Stage3(Laplace_1, True)

        output1 = Scale1
        output2 = tf.image.resize(Scale1, [Scale1.shape[1]*2, Scale1.shape[2]*2]) + Scale2
        output3 = tf.image.resize(output2, [output2.shape[1]*2, output2.shape[2]*2]) + Scale3

        gt_down2 = tf.image.resize(labels, [int(labels.shape[1] / 2), int(labels.shape[2] / 2)])
        gt_down4 = tf.image.resize(gt_down2, [int(gt_down2.shape[1] / 2), int(gt_down2.shape[2] / 2)])
        in_down2 = tf.image.resize(images, [int(images.shape[1] / 2), int(images.shape[2] / 2)])
        in_down4 = tf.image.resize(in_down2, [int(in_down2.shape[1] / 2), int(in_down2.shape[2] / 2)])

        reup2 = tf.image.resize(gt_down4, [gt_down4.shape[1]*2, gt_down4.shape[2]*2])
        reup3 = tf.image.resize(gt_down2, [gt_down2.shape[1]*2, gt_down2.shape[2]*2])

        laplace2 = gt_down2 - reup2
        laplace3 = labels - reup3

        scale3loss = scale3_loss(labels, output3)   # same as paper
        scale2loss = scale2_loss(gt_down2, output2) # same as paper
        scale1loss = scale1_loss(gt_down4, output1) # same as paper
        scale1color = 1 + (1 * scale1_color(gt_down4, output1)) # same as paper
        scale2color = 1 + (1 * scale2_color(gt_down2, output2)) # same as paper
        scale3color = 1 + (1 * scale3_color(labels, output3))   # same as paper
        laplaceloss2 = laplace_loss2(laplace2, Scale2)  # same as paper
        laplaceloss3 = laplace_loss3(laplace3, Scale3)  # same as paper

        loss = 2 * scale1loss + scale2loss + scale3loss + 2 * scale1color + scale2color + scale3color + laplaceloss2 + laplaceloss3

    grads1 = stage1.gradient(loss, Stage1.trainable_variables)
    grads2 = stage2.gradient(loss, Stage2.trainable_variables)
    grads3 = stage3.gradient(loss, Stage3.trainable_variables)
    optim1.apply_gradients(zip(grads1, Stage1.trainable_variables))
    optim2.apply_gradients(zip(grads2, Stage2.trainable_variables))
    optim3.apply_gradients(zip(grads3, Stage3.trainable_variables))

    return loss

def main():
    
    Stage1 = LMSB(input_shape=(FLAGS.img_size // 4, FLAGS.img_size // 4, 3))
    Stage2 = LMSB_2(input_shape=(FLAGS.img_size // 2, FLAGS.img_size // 2, 3))
    Stage3 = LMSB_2(input_shape=(FLAGS.img_size, FLAGS.img_size, 3))

    Stage1.summary()
    Stage2.summary()
    Stage3.summary()

    if FLAGS.pre_checkpoint:
        ckpt = tf.train.Checkpoint(Stage1=Stage1, 
                                    Stage2=Stage2,
                                    Stage3=Stage3,
                                    optim1=optim1,
                                    optim2=optim2,
                                    optim3=optim3)
        ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.pre_checkpoint_path, 5)
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print("Restored!!!")

    data_list = np.loadtxt(FLAGS.tr_txt_path, dtype="<U200", skiprows=0, usecols=0)
    img_data = [FLAGS.tr_img_path + data for data in data_list]
    img_data = np.array(img_data)
    lab_data = [FLAGS.tr_lab_path + data for data in data_list]
    lab_data = np.array(lab_data)
    
    if FLAGS.train:
        count = 0
        for epoch in range(FLAGS.epochs):
            tr_gener = tf.data.Dataset.from_tensor_slices((img_data, lab_data))
            tr_gener = tr_gener.shuffle(len(img_data))
            tr_gener = tr_gener.map(tr_func)
            tr_gener = tr_gener.batch(FLAGS.batch_size)
            tr_gener = tr_gener.prefetch(tf.data.experimental.AUTOTUNE)

            tr_iter = iter(tr_gener)
            tr_idx = len(img_data) // FLAGS.batch_size
            for step in range(tr_idx):
                batch_images, batch_labels = next(tr_iter)

                loss = cal_loss(Stage1, Stage2, Stage3, batch_images, batch_labels)

                if count % 10 == 0:
                    print("Epoch: {} DSLR loss = {} [{}/{}]".format(epoch, loss, step+1, tr_idx))

                # sample_images
                if count % 100 == 0:
                    x_down2 = tf.image.resize(batch_images, [int(batch_images.shape[1] / 2), int(batch_images.shape[2] / 2)])
                    x_down4 = tf.image.resize(x_down2, [int(x_down2.shape[1] / 2), int(x_down2.shape[2] / 2)])

                    x_reup2 = tf.image.resize(x_down4, [x_down4.shape[1]*2, x_down4.shape[2]*2])
                    x_reup = tf.image.resize(x_down2, [x_down2.shape[1]*2, x_down2.shape[2]*2])

                    Laplace_2 = x_down2 - x_reup2
                    Laplace_1 = batch_images - x_reup

                    Scale1 = Stage1(x_down4, True)
                    Scale2 = Stage2(Laplace_2, True)
                    Scale3 = Stage3(Laplace_1, True)

                    output1 = Scale1
                    output2 = tf.image.resize(Scale1, [Scale1.shape[1]*2, Scale1.shape[2]*2]) + Scale2
                    output3 = tf.image.resize(output2, [output2.shape[1]*2, output2.shape[2]*2]) + Scale3
                    #output3 = tf.nn.tanh(output3).numpy()

                    plt.imsave(FLAGS.sample_images + "/{}_predict_0.png".format(count), output3[0] * 0.5 + 0.5)
                    plt.imsave(FLAGS.sample_images + "/{}_predict_1.png".format(count), output3[1] * 0.5 + 0.5)
                    plt.imsave(FLAGS.sample_images + "/{}_predict_2.png".format(count), output3[2] * 0.5 + 0.5)
                    plt.imsave(FLAGS.sample_images + "/{}_predict_3.png".format(count), output3[3] * 0.5 + 0.5)
                    plt.imsave(FLAGS.sample_images + "/{}_label_0.png".format(count), batch_labels[0] * 0.5 + 0.5)
                    plt.imsave(FLAGS.sample_images + "/{}_label_1.png".format(count), batch_labels[1] * 0.5 + 0.5)
                    plt.imsave(FLAGS.sample_images + "/{}_label_2.png".format(count), batch_labels[2] * 0.5 + 0.5)
                    plt.imsave(FLAGS.sample_images + "/{}_label_3.png".format(count), batch_labels[3] * 0.5 + 0.5)



                count += 1


            ckpt = tf.train.Checkpoint(Stage1=Stage1, 
                                       Stage2=Stage2,
                                       Stage3=Stage3,
                                       optim1=optim1,
                                       optim2=optim2,
                                       optim3=optim3)
            ckpt.save(FLAGS.save_checkpoint + "/" + "DSLR_Net.ckpt")

    else:
        data_list = os.listdir(FLAGS.te_img_path)
        img_data = [FLAGS.te_img_path + data for data in data_list]
        img_data = np.array(img_data)

        te_gener = tf.data.Dataset.from_tensor_slices(img_data)
        te_gener = te_gener.map(te_func)
        te_gener = te_gener.batch(1)
        te_gener = te_gener.prefetch(tf.data.experimental.AUTOTUNE)

        te_iter = iter(te_gener)
        te_idx = len(img_data) // 1
        for step in range(te_idx):
            print("Saving restored images....{}".format(step + 1))
            images = next(te_iter)

            x_down2 = tf.image.resize(images, [int(images.shape[1] / 2), int(images.shape[2] / 2)])
            x_down4 = tf.image.resize(x_down2, [int(x_down2.shape[1] / 2), int(x_down2.shape[2] / 2)])

            x_reup2 = tf.image.resize(x_down4, [x_down4.shape[1]*2, x_down4.shape[2]*2])
            x_reup = tf.image.resize(x_down2, [x_down2.shape[1]*2, x_down2.shape[2]*2])

            Laplace_2 = x_down2 - x_reup2
            Laplace_1 = images - x_reup

            Scale1 = Stage1(x_down4, False)
            Scale2 = Stage2(Laplace_2, False)
            Scale3 = Stage3(Laplace_1, False)

            output1 = Scale1
            output2 = tf.image.resize(Scale1, [Scale1.shape[1]*2, Scale1.shape[2]*2]) + Scale2
            output3 = tf.image.resize(output2, [output2.shape[1]*2, output2.shape[2]*2]) + Scale3

            name = img_data[step].split("/")[-1]
            plt.imsave(FLAGS.test_images + "/{}".format(name), output3[0] * 0.5 + 0.5)


if __name__ == "__main__":
    main()
