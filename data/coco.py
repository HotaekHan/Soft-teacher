from pycocotools.coco import COCO
import cv2
import os
import json
from tqdm import tqdm

is_out = True


''' coco 2014  '''
# out_path = 'coco_train2014.json'
# out_path = 'coco_val2014.json'
''' 2014 minival '''
out_path = 'coco_trainval35k.json'
# out_path = 'coco_minival5k.json'
''' coco 2017 '''
# out_path = 'coco_train2017.json'
# out_path = 'coco_val2017.json'
# out_path = 'coco_unlabel2017.json'


''' coco 2014 '''
# annFile = '/data/data/COCO/annotations_trainval2014/annotations/instances_train2014.json'
# dir_name = 'train2014'
# annFile = '/data/data/COCO/annotations_trainval2014/annotations/instances_val2014.json'
# dir_name = 'val2014'
''' 2014 minival '''
annFile = '/data/data/COCO/coco_minival2014/instances_valminusminival2014.json'
dir_name = 'val2014'
# annFile = '/data/data/COCO/coco_minival2014/instances_minival2014.json'
# dir_name = 'val2014'
''' coco 2017 '''
# annFile = '/data/data/COCO/annotations_trainval2017/annotations/instances_train2017.json'
# dir_name = 'train2017'
# annFile = '/data/data/COCO/annotations_trainval2017/annotations/instances_val2017.json'
# dir_name = 'val2017'
# annFile = '/data/data/COCO/image_info_unlabeled2017/annotations/image_info_unlabeled2017.json'
# dir_name = 'unlabeled2017'

coco_root = '/data/data/COCO'

coco = COCO(annFile)
categories = coco.loadCats(coco.getCatIds())
class_names = [cat['name'] for cat in categories]
print('COCO categories: \n{}\n'.format('|'.join(class_names)))
fout = open('coco_classes.txt', 'w')
for class_name in class_names:
    fout.write(f'{class_name}\n')
fout.close()

img_ids = coco.getImgIds()

out_dict = dict()
img_idx = 0
for img_id in tqdm(img_ids):
    img_info = coco.loadImgs(img_id)[0]
    img_name = img_info['file_name']
    img_path = os.path.join(coco_root, dir_name, img_name)
    img = cv2.imread(img_path)
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)

    boxes = list()
    labels = list()
    for ann in anns:
        if ann['image_id'] != img_id or ann['iscrowd'] != 0:
            continue
        bbox = ann['bbox']
        xmin = bbox[0]
        ymin = bbox[1]
        xmax = xmin + bbox[2]
        ymax = ymin + bbox[3]
        class_name = coco.loadCats(ann['category_id'])[0]['name']


        # ''' 2 interest classes (table and chair) '''
        # if class_name != 'dining table' and class_name != 'chair':
        #     continue

        # cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0))
        # cv2.putText(img, class_name, (int(xmin), int(ymin)), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)
        # cv2.imshow('test', img)
        # cv2.waitKey(0)

        boxes.append([xmin, ymin, xmax, ymax])
        labels.append([class_name])

    if len(boxes) == 0:
        continue

    gt_key = 'img_' + str(img_idx)
    img_idx += 1
    out_dict[gt_key] = list()
    out_data = dict()
    out_data['boxes'] = boxes
    out_data['labels'] = labels
    out_data['image_path'] = img_path

    out_dict[gt_key].append(out_data)

if is_out is True:
    json_str = json.dumps(out_dict, indent=4)
    out_file = open(out_path, 'w')
    out_file.writelines(json_str)
    out_file.close()