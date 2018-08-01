import itertools

import numpy as np

def perform_voting(patches, output_shape, expected_shape, extraction_step, num_classes) :
    vote_img = np.zeros(expected_shape + (num_classes, ))

    coordinates = generate_indexes(
        output_shape, extraction_step, expected_shape)

    for count, coord in enumerate(coordinates) :
        selection = [slice(coord[i] - output_shape[i], coord[i]) for i in range(len(coord))]
        vote_img[selection] += patches[count]

    return np.argmax(vote_img, axis=3)

def generate_indexes(output_shape, extraction_step, expected_shape) :
    ndims = len(output_shape)

    poss_shape = [output_shape[i] + extraction_step[i] * ((expected_shape[i] - output_shape[i]) // extraction_step[i]) for i in range(ndims)]
    
    idxs = [range(output_shape[i], poss_shape[i] + 1, extraction_step[i]) for i in range(ndims)]

    return itertools.product(*idxs)