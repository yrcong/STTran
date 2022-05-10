import os
import json
import numpy as np

def load_ag_cls(root_path='/nobackup/users/bowu/data/STAR/AG_annotations/'):
    object_classes = []
    with open(os.path.join(root_path, 'object_classes.txt'), 'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            object_classes.append(line)
    f.close()    
    relationship_classes = []
    with open(os.path.join(root_path, 'relationship_classes.txt'), 'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            relationship_classes.append(line)
    f.close()
    
    object_classes[9-1] = 'closet/cabinet'
    object_classes[11-1] = 'cup/glass/bottle'
    object_classes[23-1] = 'paper/notebook'
    object_classes[24-1] = 'phone/camera'
    object_classes[31-1] = 'sofa/couch'
    relationship_classes[0] = 'looking_at'
    relationship_classes[1] = 'not_looking_at'
    relationship_classes[5] = 'in_front_of'
    relationship_classes[7] = 'on_the_side_of'
    relationship_classes[10] = 'covered_by'
    relationship_classes[11] = 'drinking_from'
    relationship_classes[13] = 'have_it_on_the_back'
    relationship_classes[15] = 'leaning_on'
    relationship_classes[16] = 'lying_on'
    relationship_classes[17] = 'not_contacting'
    relationship_classes[18] = 'other_relationship'
    relationship_classes[19] = 'sitting_on'
    relationship_classes[20] = 'standing_on'
    relationship_classes[25] = 'writing_on'
    
    return object_classes, relationship_classes

def load_star_cls(root_path='/nobackup/users/bowu/data/STAR/Annotations/'):
    object_classes = {}
    with open(os.path.join(root_path, 'object_classes.txt'), 'r') as f:
        for line in f.readlines():
            oid = line.strip('\n').split()[0]
            ocls = line.strip('\n').split()[1]
            object_classes[ocls] = oid
    f.close()
    
    relationship_classes = {}
    with open(os.path.join(root_path, 'relationship_classes.txt'), 'r') as f:
        for line in f.readlines():
            rid = line.strip('\n').split()[0]
            rcls = line.strip('\n').split()[1]
            relationship_classes[rcls] = rid
    f.close()
    
    return object_classes, relationship_classes


ag_obj_cls, ag_rel_cls = load_ag_cls()
star_obj_dict, star_rel_dict = load_star_cls()


def constrain(pred_rel_inds, rel_scores):
    pred_rels = np.column_stack((pred_rel_inds, rel_scores.argmax(1))) #1+  dont add 1 because no dummy 'no relations'
    predicate_scores = rel_scores.max(1)
    return pred_rels, predicate_scores

def remove_att_rel(pred_rels):
    pred_rels = pred_rels.tolist()
    pred_rels = [rel for rel in pred_rels if rel[2]!=0 and rel[2]!=1 and rel[2]!=3]
    return np.array(pred_rels)

def nms(dets, scores, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

def reformat_to_star(graph_dict):
    star_graph = {'rel_pairs':[],'rel_labels':[],'bbox':[],'bbox_labels':[]}
    pred_rel_inds = graph_dict['pred_rel_inds']
    pred_boxes = graph_dict['pred_boxes']
    pred_classes = graph_dict['pred_classes']
    star_obj_cls =  [star_obj_dict[ag_obj_cls[cls-1]] for cls in pred_classes]
    
    star_graph['bbox_labels'] = star_obj_cls
    star_obj_box = np.around(pred_boxes, 2).tolist()
    star_graph['bbox'] = star_obj_box
    
    for rel in pred_rel_inds:
        id1, id2, rel_id = rel[0], rel[1], rel[2]
        obj1 = ag_obj_cls[pred_classes[id1]-1]
        rel = ag_rel_cls[rel_id]
        obj2 = ag_obj_cls[pred_classes[id2]-1]
        if rel not in star_rel_dict:
            continue
        rel_pair = [star_obj_dict[obj1],star_obj_dict[obj2]]
        rel = star_rel_dict[rel]
        star_graph['rel_pairs'].append(rel_pair)
        star_graph['rel_labels'].append(rel)
    return star_graph


def generate_scene_graph(pred_entry):
    pred_boxes = pred_entry['pred_boxes']
    pred_classes = pred_entry['pred_classes']
    obj_scores = pred_entry['obj_scores']
    pred_rel_inds = pred_entry['pred_rel_inds']
    rel_scores = pred_entry['rel_scores']
    
    pred_rels, predicate_scores = constrain(pred_rel_inds,rel_scores)
    pred_rels = remove_att_rel(pred_rels)
    #print(pred_rels)
    exsit_obj_idx = list(set(pred_rels[:,0].tolist() + pred_rels[:,1].tolist()))
    keep_idx = nms(pred_boxes[exsit_obj_idx], obj_scores[exsit_obj_idx],0.3)
    keep_obj_idx = sorted(np.array(exsit_obj_idx)[keep_idx])
    new_obj_idx = [i for i in range(len(keep_obj_idx))]
    
    obj_class = pred_classes[keep_obj_idx]
    obj_box = pred_boxes[keep_obj_idx]

    keep_rel = []
    for rel in pred_rels:
        if rel[0] in keep_obj_idx and rel[1] in keep_obj_idx:
            id1 = new_obj_idx[keep_obj_idx.index(rel[0])]
            id2 = new_obj_idx[keep_obj_idx.index(rel[1])]
            keep_rel.append([id1,id2,rel[2]])
    
    graph_dict = {'pred_rel_inds': keep_rel, 'pred_boxes':obj_box.tolist(),'pred_classes':obj_class.tolist()}

    star_graph = reformat_to_star(graph_dict)
    
    return star_graph




