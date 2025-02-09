import os
import cv2
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from config import config
from utils.pyt_utils import ensure_dir, link_file, load_model, parse_devices
from utils.visualize import print_iou, show_img, color_mask
from engine.evaluator import Evaluator
from engine.logger import get_logger
from utils.metric import hist_info, compute_score
from dataloader.RGBXDataset import RGBXDataset
from models.builder import EncoderDecoder as segmodel
from dataloader.dataloader import ValPre
import seaborn as sns

logger = get_logger()

class SegEvaluator(Evaluator):
    confusion_matrix = np.zeros((config.num_classes, config.num_classes))
    def func_per_iteration(self, data, device):
        img = data['data'].cpu().numpy()
        label = data['label'].cpu().numpy()
        modal_x = data['modal_x'].cpu().numpy()
        name = data['fn']
        pred = self.sliding_eval_rgbX(img, modal_x, config.eval_crop_size, config.eval_stride_rate, device)
        hist_tmp, labeled_tmp, correct_tmp = hist_info(config.num_classes, pred, label)
        self.confusion_matrix = np.add(self.confusion_matrix, hist_tmp)
        results_dict = {'hist': hist_tmp, 'labeled': labeled_tmp, 'correct': correct_tmp}

        if self.save_path is not None or self.show_image:
            colors = self.dataset.get_class_colors()
            image = img
            clean = np.zeros(label.shape)
            comp_img = show_img(colors, config.background, image, clean,
                                label,
                                pred)
            if self.save_path is not None:
                # ensure the directories
                save_path = self.save_path + '/' + '/'.join(name.split('/')[:-1])
                filename = name.split('/')[-1].split('.')[0] + '.png'
                ensure_dir(save_path)
                # result_img = Image.fromarray(pred.astype(np.uint8), mode='P')
                # palette_list = list(np.array(colors).flat)
                # if len(palette_list) < 768:
                #     palette_list += [0] * (768 - len(palette_list))
                # result_img.putpalette(palette_list)
                result_img = color_mask(pred.astype(np.uint8), colors)
                # result_img.save(os.path.join(save_path, filename))
                cv2.imwrite(os.path.join(save_path, filename), result_img)

                # save raw result
                cv2.imwrite(os.path.join(save_path, filename.split('.')[0] + '_raw.png'), pred)
                logger.info('Save the image ' + filename)

                # also save the GT "label" image
                # label_img = color_mask(label.astype(np.uint8), colors)
                # cv2.imwrite(os.path.join(save_path, filename.split('.')[0] + '_gt.png'), label_img)
                # save the rgb
                cv2.imwrite(os.path.join(save_path, filename.split('.')[0] + '_rgb.png'), img*255)
                # save the x
                # cv2.imwrite(os.path.join(save_path, filename.split('.')[0] + '_x.png'), modal_x*255)


                
            if self.show_image: 
                cv2.imshow('comp_image', comp_img)
                cv2.waitKey(0)

        return results_dict

    def compute_metric(self, results):
        hist = np.zeros((config.num_classes, config.num_classes))
        correct = 0
        labeled = 0
        count = 0
        for d in results:
            hist += d['hist']
            correct += d['correct']
            labeled += d['labeled']
            count += 1

        iou, mean_IoU, _, freq_IoU, mean_pixel_acc, pixel_acc = compute_score(hist, correct, labeled)
        result_line = print_iou(iou, freq_IoU, mean_pixel_acc, pixel_acc,
                                dataset.class_names, show_no_back=False)
        return result_line
    
    def render_confusion_matrix(self, confusion_matrix, class_names):
        plt.figure(figsize=(10, 8))
        # numbers have one decimal place and NOT show the legend
        sns.heatmap(confusion_matrix, annot=True, cmap='Blues', xticklabels=class_names, yticklabels=class_names, fmt='.1%', cbar=False)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.savefig(config.log_dir + '/confusion_matrix.png')
        print('Confusion matrix saved at: ', config.log_dir + '/confusion_matrix.png')
        # print the mean IoU on the console
        print('Mean IoU: ', np.mean(np.diag(confusion_matrix)))
    
    def process_confusion_matrix(self):
        """
        Takes the already built confusion matrix and processes it to get the IoU in percentage.
        """
        percentage_iou = np.zeros((config.num_classes, config.num_classes))
        # for each row, count the number of actual pixels
        for row in range(config.num_classes):
            row_sum = np.sum(self.confusion_matrix[row])
            # for each column, divide the number of correctly classified pixels by the total number of actual pixels
            for col in range(config.num_classes):
                percentage_iou[row][col] = self.confusion_matrix[row][col] / row_sum
        # create an image from the confusion matrix
        self.render_confusion_matrix(percentage_iou, config.class_names)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', default='last', type=str)
    parser.add_argument('-d', '--devices', default='0', type=str)
    parser.add_argument('-v', '--verbose', default=False, action='store_true')
    parser.add_argument('--show_image', '-s', default=False,
                        action='store_true')
    parser.add_argument('--save_path', '-p', default=None)

    args = parser.parse_args()
    all_dev = parse_devices(args.devices)

    network = segmodel(cfg=config, criterion=None, norm_layer=nn.BatchNorm2d)
    data_setting = {
        'rgb_root': config.rgb_root_folder,
        'rgb_format': config.rgb_format,
        'transform_gt': False,
        'x_root': config.x_root_folder,
        'x_format': config.x_format,
        'x_single_channel': config.x_is_single_channel,
        'class_names': config.class_names,
        'train_json': config.train_json,
        'val_json': config.val_json,
    }
    val_pre = ValPre()
    dataset = RGBXDataset(data_setting, 'val', val_pre)
 
    with torch.no_grad():
        segmentor = SegEvaluator(dataset, config.num_classes, config.norm_mean,
                                 config.norm_std, network,
                                 config.eval_scale_array, config.eval_flip,
                                 all_dev, args.verbose, args.save_path,
                                 args.show_image)
        segmentor.run(config.checkpoint_dir, args.epochs, config.val_log_file,
                      config.link_val_log_file)
        segmentor.process_confusion_matrix()