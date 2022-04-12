# End to End Semi Supervised Object Detection with Soft Teacher
참고문헌 [Paper](https://arxiv.org/abs/2106.09018)  

### Notes
- 논문에서는 Two-stage detector(RCNN)를 기준으로 구현하였으나, 해당 repo에서는 Single stage detector(RetinaNet)으로 구현됨.
- Single stage detector의 특성에 따라, 논문에 나온 trick을 모두 동일하게 적용할 수 없으므로 일부 수정됨.

## Usage

### Environments
- Ubuntu 18.04
- cuda version == 11.2  
- cudnn == 
- Pytorch == 
- Albumentations == 

### Train
- Data Preparation
    - download COCO data  
    [COCO Download](https://cocodataset.org/#download)
    - data/coco.py 에서 경로 수정  
    [Output file 이름](https://gitlab.dspace.kt.co.kr/soft_teacher/soft_teacher/-/blob/master/data/coco.py#L10)  
    [Annotation 파일 경로](https://gitlab.dspace.kt.co.kr/soft_teacher/soft_teacher/-/blob/master/data/coco.py#L22)  
    [COCO 폴더 경로](https://gitlab.dspace.kt.co.kr/soft_teacher/soft_teacher/-/blob/master/data/coco.py#L40)
    - python 스크립트 실행
            
            python data/coco.py
    - data 폴더 아래에 json 파일 생성 확인

- config 파일 설정
```
data:
  train: data/coco_trainval35k.json         # train 파일경로
  valid: data/coco_minival5k.json           # valid 파일 경로
  unlabeled: data/coco_unlabel2017.json     # unlabel 파일 경로
basenet:
  arch: Res50                               # backbone 구조
  pretrained: True                          # imagenet weight 다운로드
  lr_backbone: 1e-4                         # backbone LR
optimizer:
  which: Adam                               # optimizer 종류
  lr: 1e-4                                  # LR
  weight_decay: 1e-4                        # SGD의 weight decay
params:
  batch_size: 8                             # batch size
  epoch: 30                                 # epoch
  random_seed: 3000                         # random seed
  image_size: 320x320 #rowsxcols            # input size (rows x cols)
  classes: data/coco_classes.txt            # targe class 이름
  data_worker: 0                            # dataloader의 worker 숫자
soft_teacher:
  pseudo_cls_th: 0.9                        # pseudo label 고를 때 classification threshold. 예시는 0.9보다 높은 것들을 선택
  unsup_loss_weight: 1.0                    # loss 구할 때 unsupervised loss에 대한 weight
  ema_decay: 0.99                           # EMA로 teacher 업데이트할때 weight. 예시는 student의 0.01만큼만 반영
inference:
  cls_th: 0.05                              # 학습 이후 valid set에 대해 inference할 때 threshold
  nms_th: 0.5                               # NMS threshold
  top_k: 1000                               # inference 시 Top k개의 결과를 뽑음
model:
  model_path: None                          # 현재 사용안함
  exp_path: /data/tmp                       # 학습 결과물을 저장하는 경로
  is_finetune: False                        # 현재 사용안함
cuda:
  using: True                               # GPU 사용 여부
  gpu_id: 0                                 # 사용할 GPU Id. [0, 1] 이런식으로 주면 data parallel 동작

```
- python 스크립트 실행

      python train.py --config=configs/coco.yaml

### Evaluation
- train.py에서 학습이 끝나면, trainer.test()함수가 실행되면서 validation set에 대한 inference를 진행함
- inference가 끝난 후 evaluation.py를 이용하여 mAP 계산 가능
      python evaluation.py --config={exp_path}   # exp_path는 실험 결과물 저장된 경로
