# python
import os
import argparse
import random
import numpy as np
import shutil
import cv2
import sys

# pytorch
import torch
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
import albumentations as A
from albumentations.pytorch import ToTensorV2

# user defined
from datagen import jsonDataset, jsonUnlabelDataset
from models.retina import retinanet, encoder
from models.retina.loss import FocalLoss
from models.retina import utils

from timer import Timer


class Trainer(object):
    def __init__(self, config, trainset, validset, unlabelset, num_classes, num_anchors, img_size) -> None:
        super().__init__()
        self.config = config
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.view_pseudo_boxes = True

        self.img_size = img_size
        self.pseudo_cls_threshold = float(self.config['soft_teacher']['pseudo_cls_th'])
        self.unsup_loss_weight = float(self.config['soft_teacher']['unsup_loss_weight'])
        self.ema_decay = float(self.config['soft_teacher']['ema_decay'])
        ''' variables '''
        self.best_valid_loss = float('inf')
        self.global_iter_train = 0
        self.global_iter_valid = 0
        self.start_epoch = 0
        
        ''' cuda '''
        is_pin = False
        self.is_data_parallel = False
        if torch.cuda.is_available():
            if not self.config['cuda']['using']:
                print("WARNING: You have a CUDA device, so you should probably run with using cuda")
                device_str = 'cpu'
                is_pin = False
            else:
                if isinstance(config['cuda']['gpu_id'], list):
                    self.is_data_parallel = True
                    device_str = f"cuda:{config['cuda']['gpu_id'][0]}"
                elif isinstance(config['cuda']['gpu_id'], int):
                    device_str = f"cuda:{config['cuda']['gpu_id']}"
                else:
                    raise ValueError('Check out gpu id in config')
                is_pin = True
        else:
            print("ALARM: You have no CUDA device. Only work with CPU")
            device_str = 'cpu'
            is_pin = False
            
        self.device = torch.device(device_str)
        
        ''' data loader'''
        assert trainset
        assert validset
        assert unlabelset

        self.class_idx_map = trainset.class_idx_map
        self.target_classes = trainset.classes
            
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=self.config['params']['batch_size'],
            shuffle=True, num_workers=self.config['params']['data_worker'],
            collate_fn=trainset.collate_fn,
            pin_memory=is_pin)
        valid_loader = torch.utils.data.DataLoader(
            validset, batch_size=self.config['params']['batch_size'],
            shuffle=False, num_workers=self.config['params']['data_worker'],
            collate_fn=validset.collate_fn,
            pin_memory=is_pin)
        unlabel_loader = torch.utils.data.DataLoader(
            unlabelset, batch_size=self.config['params']['batch_size'],
            shuffle=True, num_workers=self.config['params']['data_worker'],
            pin_memory=is_pin)
        
        self.dataloaders = {'train': train_loader, 'valid': valid_loader, 
                            'unlabeled': unlabel_loader}

        '''tensorboard'''
        log_dir = os.path.join(self.config['model']['exp_path'], 'log')
        self.summary_writer = SummaryWriter(log_dir)

        ''' init '''
        self._init_net()
        self._init_criterion()
        self._init_optimizer()

    def _init_net(self):
        ''' student '''
        self.student = retinanet.load_model(num_classes=self.num_classes,
                                            num_anchors=self.num_anchors,
                                            basenet=self.config['basenet']['arch'],
                                            is_pretrained_base=self.config['basenet']['pretrained'],
                                            do_freeze=False)
        # print out net
        print(self.student)
        n_parameters = sum(p.numel() for p in self.student.parameters() if p.requires_grad)
        print('number of params:', n_parameters)
        ''' copy to device '''
        self.student.to(self.device)

        '''set data parallel'''
        if self.is_data_parallel is True:
            self.student = torch.nn.DataParallel(module=self.student,
                                                 device_ids=self.config['cuda']['gpu_id'])

        ''' teacher '''
        self.teacher = retinanet.load_model(num_classes=self.num_classes,
                                            num_anchors=self.num_anchors,
                                            basenet=self.config['basenet']['arch'],
                                            is_pretrained_base=self.config['basenet']['pretrained'],
                                            do_freeze=False)
        # print out net
        print(self.teacher)
        n_parameters = sum(p.numel() for p in self.teacher.parameters() if p.requires_grad)
        print('number of params:', n_parameters)
        ''' copy to device '''
        self.teacher.to(self.device)

        ''' copy the weights '''
        missing_keys = self.teacher.load_state_dict(utils._load_weights(self.student.state_dict()), strict=True)
        print(missing_keys)

        ''' anchor encoder '''
        self.anchor_encoder = encoder.DataEncoder()
    
    def _init_criterion(self):
        self.criterion = FocalLoss(num_classes=self.num_classes)
        self.criterion.to(self.device)
        
    def _init_optimizer(self):
        param_dicts = [
            # head params
            {
                "params": [p for n, p in self.student.named_parameters() if "base_networks" not in n and p.requires_grad]
            },
            # base params
            {
                "params": [p for n, p in self.student.named_parameters() if "base_networks" in n and p.requires_grad],
                "lr": float(self.config['basenet']['lr_backbone']),
            },
        ]
        
        if self.config['optimizer']['which'] == 'SGD':
            self.optim = optim.SGD(param_dicts, 
                                   lr=float(self.config['optimizer']['lr']),
                                   momentum=0.9, weight_decay=5e-4)
        elif self.config['optimizer']['which'] == 'Adam':
            self.optim = optim.Adam(param_dicts, 
                                    lr=float(self.config['optimizer']['lr']))
        elif self.config['optimizer']['which'] == 'AdamW':
            self.optim = optim.AdamW(param_dicts, 
                                     lr=float(self.config['optimizer']['lr']),
                                     weight_decay=float(self.config['optimizer']['weight_decay']))
        else:
            raise ValueError('not supported optimizer')

    def weak_aug(self, unsup_paths):
        weak_transforms = A.Compose([
            A.Resize(height=self.img_size[0], width=self.img_size[1], p=1.0),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
            ToTensorV2()
        ], p=1.0)

        batch = list()
        for img_path in unsup_paths:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            augmented = weak_transforms(image=img)
            img = augmented['image']
            batch.append(img)

        return torch.stack(batch, 0)

    def strong_aug(self, unsup_paths, pseudo_boxes, pseudo_labels):
        rand_solarize = random.randint(1, 255)
        rand_num_hole = random.randint(1, 5)
        rand_hole_size = random.uniform(0.0, 0.2)
        rand_max_h_size = int(self.img_size[0] * rand_hole_size)
        rand_max_w_size = int(self.img_size[1] * rand_hole_size)
        # bbox_params = A.BboxParams(format='pascal_voc')
        bbox_params = A.BboxParams(format='albumentations')

        strong_transforms = A.Compose([
            A.Resize(height=self.img_size[0], width=self.img_size[1], p=1.0),
            A.HorizontalFlip(p=0.5),

            A.OneOf([
                A.HorizontalFlip(p=0.0),  # identity
                A.Equalize(),
                A.Solarize(threshold=rand_solarize),
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
            A.Cutout(num_holes=rand_num_hole, max_h_size=rand_max_h_size, max_w_size=rand_max_w_size),

            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
            ToTensorV2()
        ], bbox_params=bbox_params, p=1.0)

        batch_img = list()
        encoded_loc = list()
        encoded_cls = list()

        for data_idx, img_path in enumerate(unsup_paths):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            bboxes = [bbox.tolist() + [label.item()] for bbox, label in zip(pseudo_boxes[data_idx], pseudo_labels[data_idx])]
            augmented = strong_transforms(image=img, bboxes=bboxes)
            img = augmented['image']
            rows, cols = img.shape[1:]
            boxes = augmented['bboxes']
            boxes = [list(bbox) for bbox in boxes]
            labels = [bbox.pop() for bbox in boxes]

            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

            ''' convert to pascal voc format'''
            if boxes.numel() != 0:
                boxes = boxes * torch.tensor([cols, rows, cols, rows], dtype=torch.float32)

            loc_target, cls_target = self.anchor_encoder.encode(boxes=boxes,
                                                                labels=labels,
                                                                input_size=(self.img_size[1], self.img_size[0]))

            batch_img.append(img)
            encoded_loc.append(loc_target)
            encoded_cls.append(cls_target)

        return torch.stack(batch_img, 0), torch.stack(encoded_loc, 0), torch.stack(encoded_cls, 0)

    def make_pseudo_boxes(self, unsup_inputs, unsup_paths):
        input_size = (unsup_inputs.shape[2], unsup_inputs.shape[3])
        num_batch = unsup_inputs.shape[0]

        ''' cls targets '''
        loc_preds, cls_preds = self.teacher(unsup_inputs)

        all_pseudo_boxes = list()
        all_pseudo_socres = list()
        all_pseudo_labels = list()
        for iter_batch in range(num_batch):
            pseudo_boxes, pseudo_labels, pseudo_scores = \
                self.anchor_encoder.decode(loc_preds=loc_preds[iter_batch].squeeze(),
                                           cls_preds=cls_preds[iter_batch].squeeze(),
                                           input_size=(input_size[1], input_size[0]), # w, h
                                           cls_threshold=self.pseudo_cls_threshold,
                                           top_k=1000)

            if len(pseudo_boxes) > 0:
                # nms mode = 0: soft-nms(liner), 1: soft-nms(gaussian), 2: hard-nms
                keep = utils.box_nms(pseudo_boxes, pseudo_scores, nms_threshold=0.5, mode=2)
                pseudo_boxes = pseudo_boxes[keep]
                pseudo_scores = pseudo_scores[keep]
                pseudo_labels = pseudo_labels[keep]

                if self.view_pseudo_boxes is True:
                    result_dir = os.path.join(self.config['model']['exp_path'], 'pseudo_label')
                    os.makedirs(result_dir, exist_ok=True)
                    bbox_colormap = utils._get_rand_bbox_colormap(class_names=self.target_classes)

                    utils._write_results(result_dir, unsup_paths[iter_batch],
                                         pseudo_boxes, pseudo_scores, pseudo_labels,
                                         self.class_idx_map,
                                         input_size, bbox_colormap)

                ''' to albumentations format '''
                ''' 
                This is hack. convert to albumentations format because of float precision error. 
                albumentations converts the bbox to albumentations format before augment the boxes.
                albumentations format is normalized bbox format. so sometimes occur some float precision error.
                check out https://github.com/albumentations-team/albumentations/issues/459
                '''
                pseudo_boxes = pseudo_boxes / torch.tensor([input_size[1], input_size[0], input_size[1], input_size[0]],
                                                           dtype=torch.float32, device=pseudo_boxes.device)
                weird_coord_idx = pseudo_boxes < 0
                pseudo_boxes[weird_coord_idx] = 0.0
                weird_coord_idx = pseudo_boxes >= 1.0
                pseudo_boxes[weird_coord_idx] = 0.999

            all_pseudo_boxes.append(pseudo_boxes)
            all_pseudo_socres.append(pseudo_scores)
            all_pseudo_labels.append(pseudo_labels)


        # ''' loc targets '''
        # jitter_preds = list()
        # for iter_jitter in range(self.num_jittering):
        #     jittered_img = self.jittering_img(unsup_inputs)
        #     jitter_loc_preds, _ = self.teacher(jittered_img)
        #     pos_jitter_preds = jitter_loc_preds[pos_ids]
        #     jitter_preds.append(pos_jitter_preds)
        #
        # #TODO: calc std from jitter_preds
        #
        # #TODO: std is smaller than threshold. these samples are pseudo loc targets.

        return all_pseudo_boxes, all_pseudo_labels, all_pseudo_socres

    def update_teacher(self, ema_decay):
        with torch.no_grad():
            for teacher_val, student_val in zip(self.teacher.state_dict().values(), self.student.state_dict().values()):
                teacher_val.copy_((ema_decay * teacher_val) + ((1.0 - ema_decay) * student_val))
    
    def train_one_epoch(self, epoch, phase):
        is_train = False
        if phase == "train":
            self.student.train()
            self.teacher.train()
            self.criterion.train()
            is_train = True
        elif phase == "valid":
            self.student.eval()
            self.teacher.eval()
            self.criterion.eval()
        else:
            raise ValueError(f'Unexpected phase mode: {phase}')
        
        acc_loss = 0.
        avg_loss = 0.

        sup_dataloader = self.dataloaders[phase]
        unsup_dataloader = self.dataloaders['unlabeled']

        sup_iter = iter(sup_dataloader)
        unsup_iter = iter(unsup_dataloader)

        sup_num_iter = int(len(sup_dataloader.batch_sampler.sampler) / sup_dataloader.batch_size)
        unsup_num_iter = int(len(unsup_dataloader.batch_sampler.sampler) / unsup_dataloader.batch_size)
        num_iter = sup_num_iter if sup_num_iter < unsup_num_iter else unsup_num_iter

        with torch.set_grad_enabled(is_train):
            for batch_idx in range(num_iter):
                ''' supervised samples '''
                try:
                    inputs, loc_targets, cls_targets, paths = next(sup_iter)
                except StopIteration:
                    print(f'[sup] occur stop iteration error. iter:{batch_idx}')
                    break

                inputs = inputs.to(self.device)
                loc_targets = loc_targets.to(self.device)
                cls_targets = cls_targets.to(self.device)

                ''' supervised loss '''
                loc_preds, cls_preds = self.student(inputs)

                loc_loss, cls_loss, sup_num_matched_anchors = \
                    self.criterion(loc_preds, loc_targets, cls_preds, cls_targets)

                # sup_num_matched_anchors = float(sup_num_matched_anchors) + 1e-4
                # sup_loss = (loc_loss + cls_loss) / sup_num_matched_anchors

                if sup_num_matched_anchors == 0:
                    ''' no clac loss for background image '''
                    # print('No matched anchor')
                    sup_loss = (loc_loss + cls_loss) * 0.0
                else:
                    sup_num_matched_anchors = float(sup_num_matched_anchors)
                    sup_loss = (loc_loss + cls_loss) / sup_num_matched_anchors

                ''' unsupervised samples '''
                try:
                    unsup_paths = next(unsup_iter)
                except StopIteration:
                    print(f'[unsup] occur stop iteration error. iter:{batch_idx}')
                    break
                # unsup_inputs = unsup_inputs.to(self.device)

                ''' make pseudo labels '''
                weak_unsup_inputs = self.weak_aug(unsup_paths)
                weak_unsup_inputs = weak_unsup_inputs.to(self.device)
                pseudo_boxes, pseudo_labels, pseudo_scores = self.make_pseudo_boxes(weak_unsup_inputs, unsup_paths)

                strong_unsup_inputs, pseudo_loc_targets, pseudo_cls_targets = \
                    self.strong_aug(unsup_paths=unsup_paths,
                                    pseudo_boxes=pseudo_boxes,
                                    pseudo_labels=pseudo_labels)
                strong_unsup_inputs = strong_unsup_inputs.to(self.device)

                loc_preds, cls_preds = self.student(strong_unsup_inputs)

                ''' unsupervised loss '''
                pseudo_loc_targets = pseudo_loc_targets.to(self.device)
                pseudo_cls_targets = pseudo_cls_targets.to(self.device)
                unsup_loc_loss, unsup_cls_loss, unsup_num_matched_anchors = \
                    self.criterion(loc_preds, pseudo_loc_targets, cls_preds, pseudo_cls_targets)

                # unsup_num_matched_anchors = float(unsup_num_matched_anchors) + 1e-4
                # unsup_loss = (unsup_loc_loss + unsup_cls_loss) / unsup_num_matched_anchors

                if unsup_num_matched_anchors == 0:
                    # print('No matched anchor')
                    unsup_loss = (unsup_loc_loss + unsup_cls_loss) * 0.0
                else:
                    unsup_num_matched_anchors = float(unsup_num_matched_anchors)
                    unsup_loss = (unsup_loc_loss + unsup_cls_loss) / unsup_num_matched_anchors

                loss = sup_loss + (self.unsup_loss_weight * unsup_loss)

                if is_train is True:
                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()

                self.update_teacher(ema_decay=self.ema_decay)

                acc_loss += loss.item()
                avg_loss = acc_loss / (batch_idx + 1)
                print(f'[{phase}] epoch: {epoch:3d} | iter: {batch_idx:4d} | avg_loss: {avg_loss:.4f} | '
                      f'sup. ce_loss: {cls_loss:.4f} | sup. bbox_loss: {loc_loss:.4f} | '
                      f'sup. num anchors: {sup_num_matched_anchors} | '
                      f'unsup. ce_loss: {unsup_cls_loss:.4f} | unsup. bbox_loss: {unsup_loc_loss:.4f} | '
                      f'unsup. num anchors: {unsup_num_matched_anchors}')

                if is_train is True:
                    self.summary_writer.add_scalar('train/avg_loss', avg_loss, self.global_iter_train)
                    self.summary_writer.add_scalar('train/sup_ce_loss', cls_loss, self.global_iter_train)
                    self.summary_writer.add_scalar('train/sup_bbox_loss', loc_loss, self.global_iter_train)
                    self.summary_writer.add_scalar('train/unsup_ce_loss', unsup_cls_loss, self.global_iter_train)
                    self.summary_writer.add_scalar('train/unsup_bbox_loss', unsup_loc_loss, self.global_iter_train)
                    self.global_iter_train += 1
                else:
                    self.summary_writer.add_scalar('valid/avg_loss', avg_loss, self.global_iter_valid)
                    self.summary_writer.add_scalar('valid/sup_ce_loss', cls_loss, self.global_iter_valid)
                    self.summary_writer.add_scalar('valid/sup_bbox_loss', loc_loss, self.global_iter_valid)
                    self.summary_writer.add_scalar('valid/unsup_ce_loss', unsup_cls_loss, self.global_iter_valid)
                    self.summary_writer.add_scalar('valid/unsup_bbox_loss', unsup_loc_loss, self.global_iter_valid)
                    self.global_iter_valid += 1

        return avg_loss
    
    def start(self):
        for epoch in range(self.start_epoch, self.config['params']['epoch'], 1):
            self.train_one_epoch(epoch, "train")

            state = {
                "epoch": epoch,
                "best_loss": self.best_valid_loss,
                "student": self.student.state_dict(),
                "optimizer": self.optim.state_dict(),
            }
            torch.save(state, os.path.join(self.config['model']['exp_path'], 'latest.pth'))

            valid_loss = self.train_one_epoch(epoch, "valid")
            if valid_loss < self.best_valid_loss:
                print("******** New optimal found, saving state ********")
                self.best_valid_loss = valid_loss
                # torch.save(state, os.path.join(self.exp_dir, "ckpt-" + str(epoch) + '.pth'))
                torch.save(state, os.path.join(self.config['model']['exp_path'], 'best.pth'))
            print()
        self.summary_writer.close()
        print("best valid loss : " + str(self.best_valid_loss))

    def test(self):
        ''' evaluate on valid dataset '''
        ''' get random box color '''
        bbox_colormap = utils._get_rand_bbox_colormap(class_names=self.target_classes)

        best_ckpt_path = os.path.join(self.config['model']['exp_path'], 'best.pth')
        print(best_ckpt_path)
        ckpt = torch.load(best_ckpt_path, map_location=self.device)

        weights = utils._load_weights(ckpt['student'])
        missing_keys = self.student.load_state_dict(weights, strict=True)
        print(missing_keys)
        self.student.to(self.device)
        self.student.eval()

        n_params = sum(p.numel() for p in self.student.parameters() if p.requires_grad)
        print(f'num. of params: {n_params}')

        result_dir = os.path.join(self.config['model']['exp_path'], 'results', 'valid')
        os.makedirs(result_dir, exist_ok=True)

        data_loader = self.dataloaders['valid']
        num_data = len(data_loader.batch_sampler.sampler)

        timer_infer = Timer()
        timer_post = Timer()
        warmup_period = 10

        with torch.set_grad_enabled(False):
            for batch_idx, (inputs, loc_targets, cls_targets, paths) in enumerate(data_loader):
                if batch_idx == warmup_period:
                    timer_infer.reset()
                    timer_post.reset()
                sys.stdout.write('\r' + str(batch_idx * self.config['params']['batch_size']) + ' / ' + str(num_data))
                inputs = inputs.to(self.device)
                # loc_preds, cls_preds, mask_preds = net(inputs)

                # torch.cuda.synchronize()
                timer_infer.tic()
                loc_preds, cls_preds = self.student(inputs)
                # torch.cuda.synchronize()
                timer_infer.toc()

                num_batch = loc_preds.shape[0]
                input_size = (inputs.shape[2], inputs.shape[3])

                for iter_batch in range(num_batch):
                    # torch.cuda.synchronize()
                    timer_post.tic()

                    boxes, labels, scores = \
                        self.anchor_encoder.decode(loc_preds=loc_preds[iter_batch].squeeze(),
                                                   cls_preds=cls_preds[iter_batch].squeeze(),
                                                   input_size=(input_size[1], input_size[0]),  # w, h
                                                   cls_threshold=self.config['inference']['cls_th'],
                                                   top_k=int(self.config['inference']['top_k']))

                    if len(boxes) > 0:
                        # nms mode = 0: soft-nms(liner), 1: soft-nms(gaussian), 2: hard-nms
                        keep = utils.box_nms(boxes, scores, nms_threshold=self.config['inference']['nms_th'], mode=2)
                        boxes = boxes[keep]
                        scores = scores[keep]
                        labels = labels[keep]

                    # torch.cuda.synchronize()
                    timer_post.toc()

                    utils._write_results(result_dir, paths[iter_batch], boxes, scores, labels,
                                         self.class_idx_map,
                                         input_size, bbox_colormap)





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='path of config file')
    opt = parser.parse_args()

    ''' read config '''
    config = utils.get_config(opt.config)

    '''make output folder'''
    if not os.path.exists(config['model']['exp_path']):
        os.makedirs(config['model']['exp_path'], exist_ok=True)

    if not os.path.exists(os.path.join(config['model']['exp_path'], 'config.yaml')):
        shutil.copy(opt.config, os.path.join(config['model']['exp_path'], 'config.yaml'))
    else:
        os.remove(os.path.join(config['model']['exp_path'], 'config.yaml'))
        shutil.copy(opt.config, os.path.join(config['model']['exp_path'], 'config.yaml'))

    '''set random seed'''
    random.seed(config['params']['random_seed'])
    np.random.seed(config['params']['random_seed'])
    torch.manual_seed(config['params']['random_seed'])

    '''Data'''
    target_classes = utils.read_txt(config['params']['classes'])
    num_classes = len(target_classes)
    img_size = config['params']['image_size'].split('x')
    img_size = (int(img_size[0]), int(img_size[1]))

    '''  data load '''
    ''' load both labeled and unlabeld data '''
    print('==> Preparing data..')
    bbox_params = A.BboxParams(format='pascal_voc')
    train_transforms = A.Compose([
        A.Resize(height=img_size[0], width=img_size[1], p=1.0),
        A.HorizontalFlip(p=0.5),
        A.OneOf([
            A.HorizontalFlip(p=0.0),  # identity
            A.Equalize(),
            A.Solarize(),
            A.ColorJitter(),
            A.RandomBrightnessContrast(brightness_limit=0.0),
            A.RandomBrightnessContrast(contrast_limit=0.0),
            A.Sharpen(),
            A.Posterize()
        ], p=1.0),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
        ToTensorV2()
    ], bbox_params=bbox_params, p=1.0)
    valid_transforms = A.Compose([
        A.Resize(height=img_size[0], width=img_size[1], p=1.0),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
        ToTensorV2()
    ], bbox_params=bbox_params, p=1.0)

    train_dataset = jsonDataset(path=config['data']['train'].split(' ')[0], classes=target_classes,
                                transforms=train_transforms, img_size=img_size)
    valid_dataset = jsonDataset(path=config['data']['valid'].split(' ')[0], classes=target_classes,
                                transforms=valid_transforms, img_size=img_size)
    unlabeled_train_dataset = jsonUnlabelDataset(path=config['data']['unlabeled'].split(' ')[0])
    
    ''' trainer '''
    trainer = Trainer(config=config, 
                      trainset=train_dataset, validset=valid_dataset,
                      unlabelset=unlabeled_train_dataset,
                      num_classes=num_classes,
                      num_anchors=train_dataset.encoder.num_anchors,
                      img_size=img_size)

    trainer.start()
    trainer.test()
