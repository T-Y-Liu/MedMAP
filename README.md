# MedMAP: Promoting Incomplete Multi-modal Brain Tumor Segmentation with Alignment

This is the implementation for the paper:

[MedMAP: Promoting Incomplete Multi-modal Brain Tumor Segmentation with Alignment](https://arxiv.org/pdf/2408.09465)
[Under Review]

## Abstract

Brain tumor segmentation is often based on multiple magnetic resonance imaging (MRI). However, in clinical practice, certain modalities of MRI may be missing, which presents a more difficult scenario. To cope with this challenge, Knowledge Distillation, Domain Adaption, and Shared Latent Space have emerged as commonly promising strategies.  However, recent efforts to address the missing modality problem in brain tumor segmentation typically overlook the modality gaps and thus fail to learn important invariant feature representations across different modalities. Such drawback consequently leads to limited performance for missing modality models. To ameliorate these problems, pre-trained models are used in natural visual segmentation tasks to minimize the gaps. However, promising pre-trained models are difficult to obtain in the brain tumor segmentation task due to the lack of sufficient data. Along this line,
in this paper, we propose a novel paradigm that aligns latent features of involved modalities to a well-defined distribution anchor as the substitution of the pre-trained model. As a major contribution, we prove that our novel training paradigm ensures a tight evidence lower bound, thus theoretically certifying its effectiveness. Extensive experiments on different backbones validate that the proposed paradigm can enable invariant feature representations and produce models with narrowed modality gaps. Models with our alignment paradigm show their superior performance on both BraTS2018, BraTS2020 and Brain Metastasis datasets. 

<!-- ![image](https://github.com/YaoZhang93/mmFormer/blob/main/figs/overview.png) -->

## Usage. 

Please refer to MEDMAP.py in this project. It can be added to the backbones mentioned in the paper directly.
* It is a draft version aiming to show how the anchor will be obtained.

<!-- * Environment Preparation
  * Download the cuda and pytorch from [Google Drive](https://drive.google.com/drive/folders/1x6z7Ot3Xfrg1dokR9cdeoRSKbQJRTpv7?usp=sharing).
  * Set the environment path in `job.sh`.
* Data Preparation
  * Download the data from [MICCAI 2018 BraTS Challenge](https://www.med.upenn.edu/sbia/brats2018/data.html).
  * Set the data path in `preprocess.py` and then run `python preprocess.py`.
  * Set the data path in `job.sh`.
* Train
  * Train the model by `sh job.sh`. 

* Test
  * The trained model should be located in `mmFormer/output`. 
  * Uncomment the evaluation command in  `job.sh` and then inference on the test data by `sh job.sh`.
  * The pre-trained [model](https://drive.google.com/file/d/1oKgjXzSfWOG5VT64EE1lfV6rdtjkyC5B/view?usp=sharing) and [log](https://drive.google.com/file/d/165u-MGAiS0_PkExXRkI4KrainRlc_Ibo/view?usp=sharing) are available. -->

## Citation

<!-- If you find this code and paper useful for your research, please kindly cite our paper. -->

```
@article{liu2024medmap,
  title={MedMAP: Promoting Incomplete Multi-modal Brain Tumor Segmentation with Alignment},
  author={Liu, Tianyi and Tan, Zhaorui and Chen, Muyin and Yang, Xi and Jiang, Haochuan and Huang, Kaizhu},
  journal={arXiv preprint arXiv:2408.09465},
  year={2024}
}
```

<!-- ## Reference
* [RFNet](https://github.com/dyh127/RFNet) -->

