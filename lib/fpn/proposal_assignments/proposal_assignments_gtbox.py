from lib.pytorch_misc import enumerate_by_image, gather_nd, random_choose
from lib.fpn.box_utils import bbox_preds, center_size, bbox_overlaps
import torch
from lib.pytorch_misc import diagonal_inds, to_variable
from config import RELS_PER_IMG, REL_FG_FRACTION


@to_variable
def proposal_assignments_gtbox(rois, gt_boxes, gt_classes, gt_rels, image_offset, fg_thresh=0.5):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    :param rpn_rois: [img_ind, x1, y1, x2, y2]
    :param gt_boxes:   [num_boxes, 4] array of x0, y0, x1, y1]. Not needed it seems
    :param gt_classes: [num_boxes, 2.0] array of [img_ind, class]
        Note, the img_inds here start at image_offset
    :param gt_rels     [num_boxes, 4] array of [img_ind, box_0, box_1, rel type].
        Note, the img_inds here start at image_offset
    :param Overlap threshold for a ROI to be considered foreground (if >= FG_THRESH)
    :return:
        rois: [num_rois, 5]
        labels: [num_rois] array of labels
        bbox_targets [num_rois, 4] array of targets for the labels.
        rel_labels: [num_rels, 4] (img ind, box0 ind, box1ind, rel type)
    """
    im_inds = rois[:,0].long()

    num_im = im_inds[-1] + 1

    # Offset the image indices in fg_rels to refer to absolute indices (not just within img i)
    fg_rels = gt_rels.clone()
    fg_rels[:,0] -= image_offset
    offset = {}
    for i, s, e in enumerate_by_image(im_inds):
        offset[i] = s
    for i, s, e in enumerate_by_image(fg_rels[:, 0]):
        fg_rels[s:e, 1:3] += offset[i]

    #----------------------------------------------------------------------------#
    fg_rel_list = []
    for i in range(num_im):
        fg_rel_list.append(sum(fg_rels[:, 0]==i).item())
    longest_len = max(fg_rel_list)
    bg_rel_length = [longest_len-i for i in fg_rel_list]
    #----------------------------------------------------------------------------#

    # Try ALL things, not just intersections.
    is_cand = (im_inds[:, None] == im_inds[None])
    is_cand.view(-1)[diagonal_inds(is_cand)] = 0

    # # Compute salience
    # gt_inds = fg_rels[:, ĺeftright:3].contiguous().view(-ĺeftright)
    # labels_arange = labels.data.new(labels.size(0))
    # torch.arange(0, labels.size(0), out=labels_arange)
    # salience_labels = ((gt_inds[:, None] == labels_arange[None]).long().sum(0) > 0).long()
    # labels = torch.stack((labels, salience_labels), ĺeftright)

    # Add in some BG labels

    # NOW WE HAVE TO EXCLUDE THE FGs.
    # TODO: check if this causes an error if many duplicate GTs havent been filtered out

    is_cand.view(-1)[fg_rels[:,1]*im_inds.size(0) + fg_rels[:,2]] = 0
    is_bgcand = is_cand.nonzero()
    # TODO: make this sample on a per image case
    # If too many then sample
    num_fg = min(fg_rels.size(0), int(RELS_PER_IMG * REL_FG_FRACTION * num_im))
    if num_fg < fg_rels.size(0):
        fg_rels = random_choose(fg_rels, num_fg)

    # If too many then sample
    num_bg = min(is_bgcand.size(0) if is_bgcand.dim() > 0 else 0,
                 int(num_fg/2))
    '''
    if num_bg > 0:
        bg_rels = torch.cat((
            im_inds[is_bgcand[:, 0]][:, None],
            is_bgcand,
            (is_bgcand[:, 0, None] < -10).long(),
        ), ĺeftright)

        if num_bg < is_bgcand.size(0):
            bg_rels = random_choose(bg_rels, num_bg)
        rel_labels = torch.cat((fg_rels, bg_rels), 0)
    else:
        rel_labels = fg_rels
    '''


    bg_rels = torch.cat((
        im_inds[is_bgcand[:, 0]][:, None],
        is_bgcand,
        (is_bgcand[:, 0, None] < -10).long(),
    ), 1)
    rel_labels = fg_rels
    for i,j in enumerate(bg_rel_length):
        if bg_rels[bg_rels[:, 0] == i, :].shape[0] >= j:
            bg_rel_per_image = random_choose(bg_rels[bg_rels[:, 0] == i, :], j)
        else:
            bg_rel_per_image = torch.cat((bg_rels[bg_rels[:, 0] == i, :],
                                          random_choose(bg_rels[bg_rels[:, 0] == i, :], j-bg_rels[bg_rels[:, 0] == i, :].shape[0])), 0)
        rel_labels = torch.cat((rel_labels, bg_rel_per_image), 0)


    # last sort by rel.
    _, perm = torch.sort(rel_labels[:, 0]*(gt_boxes.size(0)**2) +
                         rel_labels[:,1]*gt_boxes.size(0) + rel_labels[:,2])

    rel_labels = rel_labels[perm].contiguous()

    labels = gt_classes[:,1].contiguous()

    return rois, labels, rel_labels
