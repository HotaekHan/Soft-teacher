import json
from albumentations.pytorch.transforms import img_to_tensor
import cv2
import numpy as np
import os

import torch
import torch.utils.data as data

from models.retina.encoder import DataEncoder


class jsonDataset(data.Dataset):
    def __init__(self, path, classes, transforms, img_size, view_image=False,
                 min_cols=1, min_rows=1):
        self.path = path
        self.classes = classes
        self.transforms = transforms
        self.img_size = img_size
        self.encoder = DataEncoder()
        self.view_img = view_image

        self.fnames = list()
        self.boxes = list()
        self.labels = list()

        self.num_classes = len(self.classes)

        self.label_map = dict()
        self.class_idx_map = dict()
        # 0 is background class
        for idx in range(0, self.num_classes):
            self.label_map[self.classes[idx]] = idx + 1
            self.class_idx_map[idx + 1] = self.classes[idx]

        fp_read = open(self.path, 'r')
        gt_dict = json.load(fp_read)

        all_boxes = list()
        all_labels = list()
        all_img_path = list()

        '''read gt files'''
        for gt_key in gt_dict:
            gt_data = gt_dict[gt_key][0]

            box = list()
            label = list()

            num_boxes = len(gt_data['labels'])

            img = cv2.imread(gt_data['image_path'])
            img_rows = img.shape[0]
            img_cols = img.shape[1]

            for iter_box in range(0, num_boxes):
                xmin = gt_data['boxes'][iter_box][0]
                ymin = gt_data['boxes'][iter_box][1]
                xmax = gt_data['boxes'][iter_box][2]
                ymax = gt_data['boxes'][iter_box][3]
                rows = ymax - ymin
                cols = xmax - xmin

                if xmin < 0 or ymin < 0:
                    print('negative coordinate: [xmin: ' + str(xmin) + ', ymin: ' + str(ymin) + ']')
                    print(gt_data['image_path'])
                    continue

                if xmax > img_cols or ymax > img_rows:
                    print('over maximum size: [xmax: ' + str(xmax) + ', ymax: ' + str(ymax) + ']')
                    print(gt_data['image_path'])
                    continue

                if cols < min_cols:
                    print('cols is lower than ' + str(min_cols) + ': [' + str(xmin) + ', ' + str(ymin) + ', ' +
                          str(xmax) + ', ' + str(ymax) + '] '
                          + str(gt_data['image_path']))
                    continue
                if rows < min_rows:
                    print('rows is lower than ' + str(min_rows) + ': [' + str(xmin) + ', ' + str(ymin) + ', ' +
                          str(xmax) + ', ' + str(ymax) + '] '
                          + str(gt_data['image_path']))
                    continue

                class_name = gt_data['labels'][iter_box][0]
                if class_name not in self.label_map:
                    print('weired class name: ' + class_name)
                    print(gt_data['image_path'])
                    continue

                class_idx = self.label_map[class_name]
                box.append([float(xmin), float(ymin), float(xmax), float(ymax)])
                label.append(int(class_idx))

            if len(box) == 0 or len(label) == 0:
                print('none of object exist in the image: ' + gt_data['image_path'])
                continue

            all_boxes.append(box)
            all_labels.append(label)
            all_img_path.append(gt_data['image_path'])

        if len(all_boxes) == len(all_labels) and len(all_boxes) == len(all_img_path):
            num_images = len(all_img_path)
        else:
            print('num. of boxes: ' + str(len(all_boxes)))
            print('num. of labels: ' + str(len(all_labels)))
            print('num. of paths: ' + str(len(all_img_path)))
            raise ValueError('num. of elements are different(all boxes, all_labels, all_img_path)')

        for idx in range(0, num_images, 1):
            self.fnames.append(all_img_path[idx])
            self.boxes.append(torch.tensor(all_boxes[idx], dtype=torch.float32))
            self.labels.append(torch.tensor(all_labels[idx], dtype=torch.int64))

        self.num_samples = len(self.fnames)

    def __getitem__(self, idx):
        '''Load image and boxes.'''
        fname = self.fnames[idx]
        boxes = self.boxes[idx]
        labels = self.labels[idx]
        img = cv2.imread(fname)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        bboxes = [bbox.tolist() + [label.item()] for bbox, label in zip(boxes, labels)]
        augmented = self.transforms(image=img, bboxes=bboxes)
        img = augmented['image']
        rows, cols = img.shape[1:]
        boxes = augmented['bboxes']
        boxes = [list(bbox) for bbox in boxes]
        labels = [bbox.pop() for bbox in boxes]

        if self.view_img is True:
            np_img = img.numpy()
            np_img = np.transpose(np_img, (1, 2, 0))
            # np_img = np.uint8(np_img * 255)
            np_img = np.ascontiguousarray(np_img)
            np_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
            for idx_box, box in enumerate(boxes):
                cv2.rectangle(np_img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0))
                class_idx = labels[idx_box]
                text_size = cv2.getTextSize(self.class_idx_map[class_idx], cv2.FONT_HERSHEY_PLAIN, 1, 1)
                cv2.putText(np_img, self.class_idx_map[class_idx], 
                            (int(box[0]), int(box[1]) - text_size[1]), 
                            cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
            cv2.imwrite(os.path.join("crop_test", str(idx)+".jpg"), np_img)

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        return img, boxes, labels, fname

    def __len__(self):
        return self.num_samples
    
    def collate_fn(self, batch):
        imgs = [x[0] for x in batch]
        boxes = [x[1] for x in batch]
        labels = [x[2] for x in batch]
        paths = [x[3] for x in batch]

        num_batch = len(batch)

        if isinstance(self.img_size, int) is True:
            inputs = torch.zeros([num_batch, 3, self.img_size, self.img_size], dtype=torch.float32)
            # mask_targets = torch.zeros([num_imgs, self.img_size, self.img_size], dtype=torch.int64)
        elif isinstance(self.img_size, tuple) is True:
            inputs = torch.zeros([num_batch, 3, self.img_size[0], self.img_size[1]], dtype=torch.float32)
            # mask_targets = torch.zeros([num_imgs, self.img_size[0], self.img_size[1]], dtype=torch.int64)
        else:
            raise ValueError('input size should be int or tuple of ints')

        loc_targets = list()
        cls_targets = list()
        for iter_batch in range(num_batch):
            im = imgs[iter_batch]
            imh, imw = im.size(1), im.size(2)
            inputs[iter_batch, :, :imh, :imw] = im

            # Encode data.
            if isinstance(self.img_size, int) is True:
                loc_target, cls_target = self.encoder.encode(boxes=boxes[iter_batch], 
                                                             labels=labels[iter_batch],
                                                             input_size=(self.img_size, self.img_size))
            elif isinstance(self.img_size, tuple) is True:
                loc_target, cls_target = self.encoder.encode(boxes=boxes[iter_batch], 
                                                             labels=labels[iter_batch],
                                                             input_size=(self.img_size[1], self.img_size[0]))
            else:
                raise ValueError('input size should be int or tuple of ints')

            loc_targets.append(loc_target)
            cls_targets.append(cls_target)

            # mask = masks[i]
            # mask_targets[i, :imh, :imw] = mask

        # return inputs, torch.stack(loc_targets), torch.stack(cls_targets), mask_targets, paths
        return inputs, torch.stack(loc_targets), torch.stack(cls_targets), paths

    # def _resize(self, img, boxes):
    #     if isinstance(self.input_size, int) is True:
    #         w = h = self.input_size
    #     elif isinstance(self.input_size, tuple) is True:
    #         h = self.input_size[0]
    #         w = self.input_size[1]
    #     else:
    #         raise ValueError('input size should be int or tuple of ints')
    #
    #     ws = 1.0 * w / img.shape[1]
    #     hs = 1.0 * h / img.shape[0]
    #     scale = torch.tensor([ws, hs, ws, hs], dtype=torch.float32)
    #     if boxes.numel() == 0:
    #         scaled_box = boxes
    #     else:
    #         scaled_box = scale * boxes
    #     return cv2.resize(img, (w, h)), scaled_box
    
    
class jsonUnlabelDataset(data.Dataset):
    def __init__(self, path, view_image=False):
        self.path = path
        # self.transforms = transforms
        # self.img_size = img_size
        self.view_img = view_image

        self.fnames = list()

        fp_read = open(self.path, 'r')
        gt_dict = json.load(fp_read)

        '''read gt files'''
        for gt_key in gt_dict:
            gt_data = gt_dict[gt_key][0]
            self.fnames.append(gt_data['image_path'])

        self.num_samples = len(self.fnames)

    def __getitem__(self, idx):
        '''Load image and boxes.'''
        fname = self.fnames[idx]
        # img = cv2.imread(fname)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # augmented = self.transforms(image=img, bboxes=list())
        # img = augmented['image']

        # if self.view_img is True:
        #     np_img = img.numpy()
        #     np_img = np.transpose(np_img, (1, 2, 0))
        #     np_img = np.uint8(np_img * 255)
        #     np_img = np.ascontiguousarray(np_img)
        #     cv2.imwrite(os.path.join("crop_test", str(idx)+".jpg"), np_img)

        return fname

    def __len__(self):
        return self.num_samples
    

if __name__ == '__main__':
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    
    json_path = 'data/coco_sample.json'
    img_size = (320, 320)
    target_classes = ['person', 'car', 'chair']
    
    bbox_params = A.BboxParams(format='pascal_voc', min_visibility=0.3)
    transforms = A.Compose([
        A.Resize(height=320, width=320, p=1.0),
        A.HorizontalFlip(p=0.5),

        A.OneOf([
            A.HorizontalFlip(p=0.0),  # identity
            A.Equalize(),
            A.Solarize(threshold=128),
            A.ColorJitter(),
            A.RandomBrightnessContrast(brightness_limit=0.0),
            A.RandomBrightnessContrast(contrast_limit=0.0),
            A.Sharpen(),
            A.Posterize()
        ], p=1.0),
        A.OneOf([
            A.Affine(scale=1.0, translate_percent={'x': (-0.1, 0.1), 'y': 0}),
            A.Affine(scale=1.0, translate_percent={'x': 0, 'y': (-0.1, 0.1)}),
            A.Affine(scale=1.0, rotate=(-30, 30)),
            A.Compose([
                A.Affine(scale=1.0, shear={'x': (-30, 30), 'y': 0}),
                A.Affine(scale=1.0, shear={'x': 0, 'y': (-30, 30)}),
            ])
        ], p=1.0),
        A.Cutout(num_holes=8, max_h_size=20, max_w_size=40),

        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
        ToTensorV2()
    ], bbox_params=bbox_params, p=1.0)

    dataset = jsonDataset(path=json_path, classes=target_classes, 
                          transforms=transforms, img_size=img_size,
                          view_image=True)
    
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=5, shuffle=False,
        collate_fn=dataset.collate_fn
    )

    data_iterator = iter(loader)
    num_iter = int(len(loader.batch_sampler.sampler) / loader.batch_size) + 1
    for iter_idx in range(num_iter):
        out = next(data_iterator)
        tmp=0
    
    for batch_idx, (img, loc_target, cls_target, paths) in enumerate(loader):
        tmp=0



    dataset = jsonUnlabelDataset(path=json_path, transforms=transforms, img_size=img_size)

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=5, shuffle=False
    )

    for batch_idx, (img, paths) in enumerate(loader):
        tmp=0
    
    