import numpy as np
import nibabel as nib

from utils.extraction import extract_patches
from utils.reconstruction import perform_voting
from network import generate_combined_model

def execute_robex(input_filename, output_filename) :
def execute_fast(input_filename, output_filename) :
def execute_spm(input_filename, output_filename) :
def execute_segmentation(files, output_filename) :
    actual_num_channels = 6
    curr_patch_shape = (32, 32, 16)
    model_a_idxs = [0, 1, 2, 3, 4, 5]
    model_b_idxs = [0, 1, 2, 6, 7, 8]
    output_patch_shape = (32, 32, 16)
    scale = 1
    step = (8, 8, 4)
    model_pattern = 'models/unet_{}_{}_3C_seg_fast_spm.h5'
    patch_shape = (actual_num_channels, ) + curr_patch_shape
    output_shape = (num_classes, np.product(curr_patch_shape))

    filename = volume_pattern.format(folder, files[1])
    volume_data = nib.load(filename)

    for i, a_file in enumerate(files) :
        if i == 0 :
            continue

        volume_init = nib.load(files[i]).get_data()

        patches = extract_patches(volume_init, curr_patch_shape, step)
        patches = patches.reshape((-1, 1, ) + curr_patch_shape)
        if i == 1 :
            N = len(patches)
            X_test = np.empty((N, num_channels, ) + curr_patch_shape)

        X_test[:, i-1:i] = patches
        del patches
    
    normalisation_info = np.load('models/model_normalisation_data.npy').item()
    for i in range(0, 7) :
        X_mean = normalisation_info[i][0]
        X_std = normalisation_info[i][1]

        model = generate_combined_model(patch_shape, output_shape, num_classes, scale)
        model.load_weights(model_pattern.format(curr_patch_shape, i))

        for c in channels_to_normalise :
            X_test_tmp[:, c] = (X_test[:, c] - X_mean[c]) / X_std[c]

        pred = model.predict([X_test_tmp[:, model_a_idxs], X_test_tmp[:, model_b_idxs]], verbose=2)[2]
        pred = pred.reshape((len(pred), ) + output_patch_shape + (num_classes, ))

        acumm_pred = pred if i == 0 else acumm_pred + pred

    volume = perform_voting(
        acumm_pred.reshape((-1, ) + curr_patch_shape + (num_classes, )),
        output_patch_shape, volume_init.shape, step, num_classes)

    nib.save(nib.Nifti1Image(volume.astype('uint8'), volume_data.affine), output_filename)

def execute_pipeline() :
    files = ['segm.nii.gz', 'pre/FLAIR.nii.gz', 'pre/reg_T1.nii.gz', 'reg_T1_brain_mask.nii.gz',
             'c1reg_T1_brain.nii.gz', 'c2reg_T1_brain.nii.gz', 'c3reg_T1_brain.nii.gz',
             'reg_T1_brain_pve_0.nii.gz', 'reg_T1_brain_pve_1.nii.gz', 'reg_T1_brain_pve_2.nii.gz']

if __name__ == "__main__" :
    execute_pipeline()