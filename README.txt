PatchCore-Based Anomaly Detection for Ceiling Damage Assessment

Overview
--------
This repository provides the data, configuration files, trained model weights, and scripts required to reproduce the principal results presented in our manuscript submitted to Earthquake Engineering & Structural Dynamics (EESD).

The proposed framework applies PatchCore-based anomaly detection to post-earthquake assessment of suspended ceilings. The Memory Bank is constructed exclusively from normal-condition ceiling images, and simulated damage is detected as a deviation from the nominal feature distribution.

Associated Paper
----------------
Title: Monitoring for rapid damage detection of office ceilings based on undamaged photographs
Authors: Yoshihiro Nitta, Masashi Abe, Yu Fukutomi, Yoshitaka Suzuki, Akira Nishitani and Masayoshi Nakashima
Journal: Earthquake Engineering & Structural Dynamics
Status: Submitted for peer review

Code Attribution
----------------
The train.py and test.py files and the models/ and common/ directories included in this repository are unmodified files from the following open-source PatchCore implementation:

ComputermindCorp, "patchcore"
https://github.com/ComputermindCorp/patchcore

That repository is distributed under the Apache License 2.0. These original PatchCore files are redistributed here without modification so that users can reproduce the experiments using the same implementation employed in this study.

The inference.py script was developed for this study by modifying and extending the original test.py script from the ComputermindCorp PatchCore implementation to perform inference and generate the outputs required for the anomaly area ratio analysis.

The ceiling-image datasets, configuration files used for the experiments, trained Memory Banks, and other study-specific materials provided in this repository are organized to reproduce the results reported in the associated manuscript.

Repository Structure
--------------------
.
├── README.md
├── installation.txt
├── train.py
├── test.py
├── inference.py
├── kcenter_greedy_demo.py
├── visualize_features.py
├── models/
├── common/
├── cfg/
│   ├── train/
│   ├── test/
│   └── inference/
├── data/
│   ├── image_ki/
│   │   ├── train/
│   │   ├── val/
│   │   ├── test/
│   │   └── inference/
│   ├── output_ki/
│   └── weights_ki/
│       ├── resnet18_size224_param_0.1.9.ki.pth
│       ├── resnet18_size224_param_0.1.9.kia.pth
│       └── resnet18_size224_param_0.1.9.kid.pth
└── LICENSE

Requirements
------------
The experiments were conducted using the following environment:

* Ubuntu 22.04.5 LTS
* Python 3.10.12
* PyTorch 2.2.2
* CUDA 13.0
* NVIDIA GPU

The Python dependencies are listed in installation.txt.

Dataset
-------
The dataset used in this study consists of normal and simulated abnormal ceiling images.

The data are divided into:

* data/image_ki/train/OK_b0:
  normal images used to construct the initial PatchCore Memory Bank

* data/image_ki/train/OK_b0a:
  normal images that were misclassified as abnormal during testing and added to the normal training set for Memory Bank reconstruction

* data/image_ki/val/OK_b0:
  normal images used for validation and threshold determination

* data/image_ki/val/OK_b0a:
  normal images that were misclassified as abnormal during testing and added to the normal validation set for the reconstructed case

* data/image_ki/val/NG_b0:
  simulated abnormal images used only in the mixed validation dataset for threshold determination

* data/image_ki/test/OK_b0/:
  normal test images

* data/image_ki/test/NG_b0/:
  simulated abnormal test images

* data/image_ki/inference/test_Raa/:
  images used to calculate the anomaly area ratio

The directory structure should be kept unchanged when reproducing the experiments.

How to Reproduce the Results
----------------------------

1. Construct the Memory Bank

For the initial Memory Bank, run:

python train.py ./cfg/train/resnet18_ki.yaml

The initial Memory Bank is constructed from 100 normal training images. An additional set of 10 normal images is used for validation and threshold determination, consistent with Section 3.1 of the manuscript. The resulting Memory Bank will be saved in:

data/weights_ki/resnet18_size224_param_0.1.9.ki.pth


For the reconstructed Memory Bank, run:

python train.py ./cfg/train/resnet18_kia.yaml

The reconstructed Memory Bank is constructed from an expanded set of 115 normal training images, consisting of the original 100 normal training images and 15 normal images that were misclassified as abnormal during testing. The validation dataset is also expanded from 10 to 25 normal images by adding the same 15 misclassified normal images, consistent with Section 3.2 of the manuscript. The resulting Memory Bank will be saved in:

data/weights_ki/resnet18_size224_param_0.1.9.kia.pth


For the case using the mixed validation dataset, run:

python train.py ./cfg/train/resnet18_kid.yaml

The Memory Bank is constructed from the same 100 normal training images as the initial case. The mixed validation dataset (10 normal and 10 abnormal images) is used only for threshold determination, consistent with Section 3.3 of the manuscript. The resulting Memory Bank will be saved in:

data/weights_ki/resnet18_size224_param_0.1.9.kid.pth


2. Evaluate the Test Dataset

For the initial Memory Bank, run:

python test.py ./cfg/test/resnet18_ki.yaml

For the reconstructed Memory Bank, run:

python test.py ./cfg/test/resnet18_kia.yaml

For the case using the mixed validation dataset, run:

python test.py ./cfg/test/resnet18_kid.yaml

The test.py script calculates the anomaly scores and evaluates the normal and abnormal test images. The principal evaluation metrics reported in the manuscript, including Precision, Recall, and F1 score, are generated during this step.


3. Generate the Anomaly Area Ratio

Run:

python inference.py ./cfg/inference/resnet18_kia.yaml

The generated anomaly maps, corresponding anomaly scores, and anomaly area ratios are saved in:

data/output_ki/inference/test_Raa


Expected Outputs
----------------
Running the scripts above should reproduce the principal results reported in the manuscript, including:

* anomaly scores for the normal and abnormal test images;
* anomaly maps for representative ceiling conditions;
* classification results based on the specified anomaly threshold;
* Precision, Recall, and F1 score; and
* anomaly area ratio, where applicable.

Small numerical differences may occur depending on the hardware, CUDA version, PyTorch version, or other computational environment.

Memory Bank
-----------
For convenient reproduction of the results, the Memory Banks used to generate the results reported in the manuscript are provided in:

data/weights_ki/

Users can therefore either:

1. reconstruct the Memory Bank from the provided training images; or
2. directly reproduce the evaluation results using the provided Memory Bank.

Data Availability
-----------------
The data and code provided in this repository are intended to support reproduction of the principal results reported in the associated manuscript.

At the time of submission, the processed training, validation, and test datasets, scripts and configuration files, and the Memory Banks required
to reproduce the key results are publicly available in this repository.

Upon publication, these materials will also be archived in a stable, DOI-bearing research repository.


License and Third-Party Code
----------------------------
The train.py and test.py files and the models/ and common/ directories are redistributed without modification from the ComputermindCorp PatchCore repository:

https://github.com/ComputermindCorp/patchcore

The inference.py script is a study-specific derivative work based on the original test.py script from the same repository.

The original PatchCore repository is licensed under the Apache License 2.0. A copy of the Apache License 2.0 is included in this repository in the LICENSE file. The applicable copyright, attribution, and license notices from the original PatchCore distribution should be retained.

The Apache License 2.0 permits modification and redistribution of the licensed source code subject to its terms. Accordingly, both the unmodified PatchCore files and the study-specific derivative inference.py script are provided under the applicable Apache License 2.0 terms.

Licensing of any other study-specific materials in this repository should be interpreted according to the licensing information provided with this repository and the applicable licenses of any third-party components they depend on.
