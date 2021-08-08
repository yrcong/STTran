import numpy as np
from lib.fpn.box_utils import bbox_overlaps
import cv2


def assign_relations(prediction, gt_annotations, assign_IOU_threshold):
    '''
    :param prediction(list): results from FasterRCNN, each element is a dictionary including the predicted boxes,
                            labels, scores, base_feature(image), features(rois), im_info (w,h,scale)
    :param gt_annotations(list):  ground-truth, each element is a list including person info(always element 0) and objects
    :param assign_IOU_threshold: hyperparameter for SGDET, 0.5
    :return: DETECTOR_FOUND_IDX
             GT_RELATIONS
             SUPPLY_RELATIONS
    '''
    FINAL_BBOXES = prediction['FINAL_BBOXES']
    FINAL_LABELS = prediction['FINAL_LABELS']
    DETECTOR_FOUND_IDX = []
    GT_RELATIONS = []
    SUPPLY_RELATIONS = []

    assigned_labels = np.zeros(FINAL_LABELS.shape[0])

    for i, j in enumerate(gt_annotations):

        gt_boxes = np.zeros([len(j), 4])
        gt_labels = np.zeros(len(j))
        gt_boxes[0] = j[0]['person_bbox']
        gt_labels[0] = 1
        for m, n in enumerate(j[1:]):
            gt_boxes[m+1,:] = n['bbox']
            gt_labels[m+1] = n['class']

        pred_boxes = FINAL_BBOXES[FINAL_BBOXES[:,0] == i, 1:].detach().cpu().numpy()
        #labels = FINAL_LABELS[FINAL_BBOXES[:,0] == i].detach().cpu().numpy()

        IOUs = bbox_overlaps(pred_boxes, gt_boxes)
        IOUs_bool = IOUs > assign_IOU_threshold
        #
        assigned_labels[(FINAL_BBOXES[:, 0].cpu().numpy() == i).nonzero()[0][np.max(IOUs, axis=1)> 0.5]] = gt_labels[np.argmax(IOUs, axis=1)][np.max(IOUs, axis=1)> 0.5]

        detector_found_idx = []
        gt_relations = []
        supply_relations = []
        candidates = []
        for m, n in enumerate(gt_annotations[i]):
            if m == 0:
                # 1 is the person index, np.where find out the pred_boxes and check which one corresponds to gt_label
                if sum(IOUs[:, m]>assign_IOU_threshold) > 0:
                    candidate = IOUs[:, m].argmax() #[labels[np.where(IOUs_bool[:, m])[0]] == 1]
                    detector_found_idx.append(candidate)
                    gt_relations.append(n)
                    candidates.append(candidate)
                else:
                    supply_relations.append(n) #no person box is found...i think it is rarely(impossible)
            else:
                if sum(IOUs[:, m]>assign_IOU_threshold) > 0:
                    candidate = IOUs[:, m].argmax()
                    if candidate in candidates:
                        # one predbox is already assigned with one gtbox
                        for c in np.argsort(-IOUs[:, m]):
                            if c not in candidates:
                                candidate = c
                                break
                    detector_found_idx.append(candidate)
                    gt_relations.append(n)
                    candidates.append(candidate)
                    assigned_labels[(FINAL_BBOXES[:, 0].cpu().numpy() == i).nonzero()[0][candidate]] = n['class']
                else:
                    #no overlapped box
                    supply_relations.append(n)

        DETECTOR_FOUND_IDX.append(detector_found_idx)
        GT_RELATIONS.append(gt_relations)
        SUPPLY_RELATIONS.append(supply_relations)

    return DETECTOR_FOUND_IDX, GT_RELATIONS, SUPPLY_RELATIONS, assigned_labels



def _get_image_blob(im):
  """Converts an image into a network input.
  Arguments:
    im (ndarray): a color image in BGR order
  Returns:
    blob (ndarray): a data blob holding an image pyramid
    im_scale_factors (list): list of image scales (relative to im) used
      in the image pyramid
  """
  im_orig = im.astype(np.float32, copy=True)
  im_orig -= np.array([[[102.9801, 115.9465, 122.7717]]])

  im_shape = im_orig.shape
  im_size_min = np.min(im_shape[0:2])
  im_size_max = np.max(im_shape[0:2])

  processed_ims = []
  im_scale_factors = []

  for target_size in [600]:
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > 1000:
      im_scale = float(1000) / float(im_size_max)
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
            interpolation=cv2.INTER_LINEAR)
    im_scale_factors.append(im_scale)
    processed_ims.append(im)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, np.array(im_scale_factors)

def im_list_to_blob(ims):
    """Convert a list of images into a network input.

    Assumes images are already prepared (means subtracted, BGR order, ...).
    """
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    num_images = len(ims)
    blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
                    dtype=np.float32)
    for i in range(num_images):
        im = ims[i]
        blob[i, 0:im.shape[0], 0:im.shape[1], :] = im

    return blob


def enumerate_by_image(im_inds):
    im_inds_np = im_inds.cpu().numpy()
    initial_ind = int(im_inds_np[0])
    s = 0
    for i, val in enumerate(im_inds_np):
        if val != initial_ind:
            yield initial_ind, s, i
            initial_ind = int(val)
            s = i
    yield initial_ind, s, len(im_inds_np)

def transpose_packed_sequence_inds(lengths):
    """
    Goes from a TxB packed sequence to a BxT or vice versa. Assumes that nothing is a variable
    :param ps: PackedSequence
    :return:
    """

    new_inds = []
    new_lens = []
    cum_add = np.cumsum([0] + lengths)
    max_len = lengths[0]
    length_pointer = len(lengths) - 1
    for i in range(max_len):
        while length_pointer > 0 and lengths[length_pointer] <= i:
            length_pointer -= 1
        new_inds.append(cum_add[:(length_pointer+1)].copy())
        cum_add[:(length_pointer+1)] += 1
        new_lens.append(length_pointer+1)
    new_inds = np.concatenate(new_inds, 0)
    return new_inds, new_lens

def pad_sequence(frame_idx):

    lengths = []
    for i, s, e in enumerate_by_image(frame_idx):  # i img_index s:start_idx e:end_idx
        lengths.append(e - s)
    lengths = sorted(lengths, reverse=True)

    _, ls_transposed = transpose_packed_sequence_inds(lengths)  # move it to TxB form
    return ls_transposed
