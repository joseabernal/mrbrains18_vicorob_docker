import os
import numpy as np
import nibabel as nib

from utils.extraction import extract_patches
from utils.reconstruction import perform_voting
from network import generate_combined_model

input_dir = 'input'
output_dir = 'output'

def execute_robex(input_filename, output_filename) :
    command = 'sh tools/ROBEX/runROBEX.sh {} {}'
    os.system(command.format(input_filename, output_filename))

def binarise_brain_mask(input_filename, output_filename) :
    command = 'fsl5.0-fslmaths {} -thr 0 -bin {}'
    os.system(command.format(input_filename, output_filename))

def execute_fast(input_filename, output_filename) :
    command = 'fsl5.0-fast -S 1 -n 3 -t 1 {} -o {}'
    os.system(command.format(input_filename, output_filename))

def execute_unzip(input_filename) :
    command = 'gunzip {}'
    os.system(command.format(input_filename))

def execute_zip() :
    command = 'gzip output/c{}reg_T1_brain.nii'
    for i in range(1, 6) :
        os.system(command.format(i))
    command = 'gzip output/reg_T1_brain.nii'
    os.system(command.format(i))

def execute_spm() :
    command = 'sh tools/run_spm12.sh /usr/local/MATLAB/MATLAB_Compiler_Runtime/v84 batch tools/matlabbatch.mat'
    os.system(command)

def clean_output_dir() :
    command = 'rm output/*reg_T1_brain*'
    os.system(command)

def execute_segmentation(files, output_filename) :
    actual_num_channels = 6
    channels_to_normalise = [0, 1]
    curr_patch_shape = (32, 32, 16)
    model_a_idxs = [0, 1, 2, 3, 4, 5]
    model_b_idxs = [0, 1, 2, 6, 7, 8]
    num_channels = 9
    num_classes = 9
    output_patch_shape = (32, 32, 16)
    scale = 1
    step = (8, 8, 4)
    model_pattern = 'models/unet_{}_{}_3C_seg_fast_spm.h5'
    patch_shape = (actual_num_channels, ) + curr_patch_shape
    output_shape = (num_classes, np.product(curr_patch_shape))

    volume_data = nib.load(files[1])
    for i, a_file in enumerate(files) :
        if i == 0 :
            continue

        volume_init = nib.load(a_file).get_data()
        patches = extract_patches(volume_init, curr_patch_shape, step)
        patches = patches.reshape((-1, 1, ) + curr_patch_shape)

        if i == 1 :
            N = len(patches)
            X_test = np.empty((N, num_channels, ) + curr_patch_shape)

        X_test[:, i-1:i] = patches
        del patches
    
    normalisation_info = np.load('models/model_normalisation_data.npy').item()
    for model_idx in range(0, 7) :
        X_mean = normalisation_info[model_idx][0]
        X_std = normalisation_info[model_idx][1]

        model = generate_combined_model(patch_shape, output_shape, num_classes, scale)
        model.load_weights(model_pattern.format(curr_patch_shape, model_idx))

        X_test_tmp = np.copy(X_test)
        for c in channels_to_normalise :
            X_test_tmp[:, c] = (X_test[:, c] - X_mean[c]) / X_std[c]

        pred = model.predict([X_test_tmp[:, model_a_idxs], X_test_tmp[:, model_b_idxs]], verbose=1)[2]
        pred = pred.reshape((len(pred), ) + output_patch_shape + (num_classes, ))

        acumm_pred = pred if model_idx == 0 else acumm_pred + pred

    volume = perform_voting(
        acumm_pred.reshape((-1, ) + curr_patch_shape + (num_classes, )),
        output_patch_shape, volume_init.shape, step, num_classes)

    nib.save(nib.Nifti1Image(volume.astype('uint8'), volume_data.affine), output_filename)

def execute_pipeline() :
    files = ['segm.nii.gz', 'pre/FLAIR.nii.gz', 'pre/reg_T1.nii.gz', 'reg_T1_brain_mask.nii.gz',
             'c1reg_T1_brain.nii.gz', 'c2reg_T1_brain.nii.gz', 'c3reg_T1_brain.nii.gz',
             'reg_T1_brain_pve_0.nii.gz', 'reg_T1_brain_pve_1.nii.gz', 'reg_T1_brain_pve_2.nii.gz']

    robex_in_filename = os.path.join(input_dir, files[2])
    robex_out_filename = os.path.join(output_dir, files[3].replace('_mask', ''))
    bin_mask_out_filename = os.path.join(output_dir, files[3])
    seg_out_filename = os.path.join(output_dir, 'result.nii.gz')

    execute_robex(robex_in_filename, robex_out_filename)
    binarise_brain_mask(robex_out_filename, bin_mask_out_filename)
    execute_fast(robex_out_filename, robex_out_filename)
    execute_unzip(robex_out_filename)
    execute_spm()
    execute_zip()
    execute_segmentation(
        [os.path.join(input_dir, f) if 'pre/' in f else os.path.join(output_dir, f) for f in files],
        seg_out_filename)
    clean_output_dir()


if __name__ == "__main__" :
    execute_pipeline()