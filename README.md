# MR brain segmentation using an ensemble of multi-path u-shaped convolutional neural networks and tissue segmentation priors
### Docker submitted to the MICCAI MRBrainS18, team name 'nic_vicorob'

Authors: J. Bernal, M. Salem, K. Kushibar, A. Clèrigues, S. Valverde, M. Cabezas, S. Gonzáles-Villà,  J. Salvi, A. Oliver, X. Lladó

Affiliations: Research Institute of Computer Vision and Robotics, Universitat de Girona, Girona, Spain

## Pipeline
<p align="center">
<img src="https://user-images.githubusercontent.com/1215598/45596014-15bd2c00-b9ad-11e8-835a-2758c864c360.png" width="70%"></p>

Pre-processing consists of skull stripping with ROBEX [3], and tissue segmentation using SPM [1] and FAST [4]. First, we remove non-brain areas using the preprocessed T1-w. Second, we input the obtained volume into the two segmentation algorithms. Note that ROBEX will remove vessels and non-brain structures (e.g. cerebral falx and choroid plexus) which are labelled in the challenge dataset as CSF. Thus, we solely consider this mask as guide of the brain area. 

Data preparation corresponds to tiling volumes up and selecting relevant blocks. For both training and testing, blocks are extracted with 50% overlap. For training only, patches are considered if their brain content corresponds to at least 30% of the whole block. Data are extracted from nine sources: T1-w, FLAIR, brain mask, and three tissue segmentation outputs obtain with both FAST and SPM.

Segmentation is carried out by an ensemble of multi-path u-shaped networks. As there are seven training cases in the MRBrainS18 challenge, we train seven different multi-path u-nets using a leave-one-out cross-validation strategy andput them together to achieve a robust segmentation outcome. All the networks are trained for a maximum of 100 epochs using an early stopping policy with patience equal to 10.

Postprocessing consists of reconstructing the segmented volume by overlaying neighbouring predictions. As output patches overlap, voxel labels are provided through majority voting.

## Architecture
Each network is composed of two u-shaped paths, as shown in the figure. The two paths are input with T1-w, FLAIR and brain mask. While one path is provided with FAST segmentation, the other one is given the ones of SPM. The outputs of both paths are fused in a late fusion fashion. Similarly, the outputs of each network within the ensemble are combined in the same way to provide a final verdict.

<p align="center">
<img src="https://user-images.githubusercontent.com/1215598/45596024-5321b980-b9ad-11e8-87b1-1a09a62aa4f9.png" width="50%"></p>

Each path is inspired by the work of Guerrero et al. [2]. Unlike the original work, 3D volumes are processed directly, PReLU activations are used instead of ReLU and activations are used after every addition module. 

<p align="center">
<img src="https://user-images.githubusercontent.com/1215598/45596025-54eb7d00-b9ad-11e8-96a1-b59132b39e6b.png" width="70%"></p>

## References
1.  Ashburner, J., et al.: SPM12 Manual.www.fil.ion.ucl.ac.uk(2012), [Online; accessed 21 Jun 2018]
2.  Guerrero,  R.,  et  al.:  White  matter  hyperintensity  and  stroke  lesion  segmentation and differentiation using convolutional neural networks. NeuroImage: Clinical 17,918–934 (2018)
3.  Iglesias, J.E., et al.: Robust brain extraction across datasets and comparison with publicly available methods. IEEE transactions on medical imaging 30(9), 1617–1634(2011)
4.  Zhang,  Y.,  et  al.:  Segmentation  of  brain  MR  images  through  a  hidden  Markov random field model and the expectation-maximization algorithm. IEEE transactions on medical imaging 20(1), 45–57 (2001)
