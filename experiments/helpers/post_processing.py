import numpy as np
from scipy import ndimage

def remove_connected_components(segmentation_mask, threshold):
    #https://stackoverflow.com/questions/50015458/remove-connected-components-below-a-threshold-in-a-3-d-array
    binary_mask = segmentation_mask.copy()
    binary_mask[binary_mask != 0] = 1
    labelled_mask, num_labels = ndimage.label(binary_mask)
    refined_mask = segmentation_mask.copy()
    minimum_cc_sum = threshold*segmentation_mask.size

    for label in range(num_labels):
        if 0 < np.sum(refined_mask[labelled_mask == label]) < minimum_cc_sum:
            refined_mask[labelled_mask == label] = 0

    return refined_mask
