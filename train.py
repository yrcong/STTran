import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
np.set_printoptions(precision=3)
import time
import os
import pandas as pd
import copy

from dataloader.action_genome import AG, cuda_collate_fn
from lib.object_detector import detector
from lib.config import Config
from lib.evaluation_recall import BasicSceneGraphEvaluator
from lib.AdamW import AdamW
from lib.sttran import STTran

"""------------------------------------some settings----------------------------------------"""
conf = Config()
print('The CKPT saved here:', conf.save_path)
if not os.path.exists(conf.save_path):
    os.mkdir(conf.save_path)
print('spatial encoder layer num: {} / temporal decoder layer num: {}'.format(conf.enc_layer, conf.dec_layer))
for i in conf.args:
    print(i,':', conf.args[i])
"""-----------------------------------------------------------------------------------------"""

AG_dataset_train = AG(mode="train", datasize=conf.datasize, data_path=conf.data_path, filter_nonperson_box_frame=True,
                      filter_small_box=False if conf.mode == 'predcls' else True)
dataloader_train = torch.utils.data.DataLoader(AG_dataset_train, shuffle=True, num_workers=4,
                                               collate_fn=cuda_collate_fn, pin_memory=False)
AG_dataset_test = AG(mode="test", datasize=conf.datasize, data_path=conf.data_path, filter_nonperson_box_frame=True,
                     filter_small_box=False if conf.mode == 'predcls' else True)
dataloader_test = torch.utils.data.DataLoader(AG_dataset_test, shuffle=False, num_workers=4,
                                              collate_fn=cuda_collate_fn, pin_memory=False)

gpu_device = torch.device("cuda:0")
# freeze the detection backbone
object_detector = detector(train=True, object_classes=AG_dataset_train.object_classes, use_SUPPLY=True, mode=conf.mode).to(device=gpu_device)
object_detector.eval()

model = STTran(mode=conf.mode,
               attention_class_num=len(AG_dataset_train.attention_relationships),
               spatial_class_num=len(AG_dataset_train.spatial_relationships),
               contact_class_num=len(AG_dataset_train.contacting_relationships),
               obj_classes=AG_dataset_train.object_classes,
               enc_layer_num=conf.enc_layer,
               dec_layer_num=conf.dec_layer).to(device=gpu_device)

evaluator =BasicSceneGraphEvaluator(mode=conf.mode,
                                    AG_object_classes=AG_dataset_train.object_classes,
                                    AG_all_predicates=AG_dataset_train.relationship_classes,
                                    AG_attention_predicates=AG_dataset_train.attention_relationships,
                                    AG_spatial_predicates=AG_dataset_train.spatial_relationships,
                                    AG_contacting_predicates=AG_dataset_train.contacting_relationships,
                                    iou_threshold=0.5,
                                    constraint='with')

# loss function, default Multi-label margin loss
if conf.bce_loss:
    ce_loss = nn.CrossEntropyLoss()
    bce_loss = nn.BCELoss()
else:
    ce_loss = nn.CrossEntropyLoss()
    mlm_loss = nn.MultiLabelMarginLoss()

# optimizer
if conf.optimizer == 'adamw':
    optimizer = AdamW(model.parameters(), lr=conf.lr)
elif conf.optimizer == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=conf.lr)
elif conf.optimizer == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=conf.lr, momentum=0.9, weight_decay=0.01)

scheduler = ReduceLROnPlateau(optimizer, "max", patience=1, factor=0.5, verbose=True, threshold=1e-4, threshold_mode="abs", min_lr=1e-7)

# some parameters
tr = []

for epoch in range(conf.nepoch):
    model.train()
    object_detector.is_train = True
    start = time.time()
    train_iter = iter(dataloader_train)
    test_iter = iter(dataloader_test)
    for b in range(len(dataloader_train)):
        data = next(train_iter)

        im_data = copy.deepcopy(data[0].cuda(0))
        im_info = copy.deepcopy(data[1].cuda(0))
        gt_boxes = copy.deepcopy(data[2].cuda(0))
        num_boxes = copy.deepcopy(data[3].cuda(0))
        gt_annotation = AG_dataset_train.gt_annotations[data[4]]

        # prevent gradients to FasterRCNN
        with torch.no_grad():
            entry = object_detector(im_data, im_info, gt_boxes, num_boxes, gt_annotation ,im_all=None)

        pred = model(entry)

        attention_distribution = pred["attention_distribution"]
        spatial_distribution = pred["spatial_distribution"]
        contact_distribution = pred["contacting_distribution"]

        attention_label = torch.tensor(pred["attention_gt"], dtype=torch.long).to(device=attention_distribution.device).squeeze()
        if not conf.bce_loss:
            # multi-label margin loss or adaptive loss
            spatial_label = -torch.ones([len(pred["spatial_gt"]), 6], dtype=torch.long).to(device=attention_distribution.device)
            contact_label = -torch.ones([len(pred["contacting_gt"]), 17], dtype=torch.long).to(device=attention_distribution.device)
            for i in range(len(pred["spatial_gt"])):
                spatial_label[i, : len(pred["spatial_gt"][i])] = torch.tensor(pred["spatial_gt"][i])
                contact_label[i, : len(pred["contacting_gt"][i])] = torch.tensor(pred["contacting_gt"][i])

        else:
            # bce loss
            spatial_label = torch.zeros([len(pred["spatial_gt"]), 6], dtype=torch.float32).to(device=attention_distribution.device)
            contact_label = torch.zeros([len(pred["contacting_gt"]), 17], dtype=torch.float32).to(device=attention_distribution.device)
            for i in range(len(pred["spatial_gt"])):
                spatial_label[i, pred["spatial_gt"][i]] = 1
                contact_label[i, pred["contacting_gt"][i]] = 1

        losses = {}
        if conf.mode == 'sgcls' or conf.mode == 'sgdet':
            losses['object_loss'] = ce_loss(pred['distribution'], pred['labels'])

        losses["attention_relation_loss"] = ce_loss(attention_distribution, attention_label)
        if not conf.bce_loss:
            losses["spatial_relation_loss"] = mlm_loss(spatial_distribution, spatial_label)
            losses["contact_relation_loss"] = mlm_loss(contact_distribution, contact_label)

        else:
            losses["spatial_relation_loss"] = bce_loss(spatial_distribution, spatial_label)
            losses["contact_relation_loss"] = bce_loss(contact_distribution, contact_label)

        optimizer.zero_grad()
        loss = sum(losses.values())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5, norm_type=2)
        optimizer.step()

        tr.append(pd.Series({x: y.item() for x, y in losses.items()}))

        if b % 1000 == 0 and b >= 1000:
            time_per_batch = (time.time() - start) / 1000
            print("\ne{:2d}  b{:5d}/{:5d}  {:.3f}s/batch, {:.1f}m/epoch".format(epoch, b, len(dataloader_train),
                                                                                time_per_batch, len(dataloader_train) * time_per_batch / 60))

            mn = pd.concat(tr[-1000:], axis=1).mean(1)
            print(mn)
            start = time.time()

    torch.save({"state_dict": model.state_dict()}, os.path.join(conf.save_path, "model_{}.tar".format(epoch)))
    print("*" * 40)
    print("save the checkpoint after {} epochs".format(epoch))

    model.eval()
    object_detector.is_train = False
    with torch.no_grad():
        for b in range(len(dataloader_test)):
            data = next(test_iter)

            im_data = copy.deepcopy(data[0].cuda(0))
            im_info = copy.deepcopy(data[1].cuda(0))
            gt_boxes = copy.deepcopy(data[2].cuda(0))
            num_boxes = copy.deepcopy(data[3].cuda(0))
            gt_annotation = AG_dataset_test.gt_annotations[data[4]]

            entry = object_detector(im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all=None)
            pred = model(entry)
            evaluator.evaluate_scene_graph(gt_annotation, pred)
        print('-----------', flush=True)
    score = np.mean(evaluator.result_dict[conf.mode + "_recall"][20])
    evaluator.print_stats()
    evaluator.reset_result()
    scheduler.step(score)



