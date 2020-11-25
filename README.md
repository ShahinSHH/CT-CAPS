# CT-CAPS
<h4>Feature Extraction-Based Automated Framework For COVID-19 Disease Identification From Chest CT Scans Using Capsule Networks.</h4>

CT-CAPS is a Capsule Network-based feature extractor developed to detect specific characteristics of CT slices, followed by a Max Pooling Layer to convert slice-level feature maps into patientlevel ones. Finally, a stack of fully connected layers are added to make the final decision. The CT-CAPS is trained on a dataset of CT slices labeled by three experienced radiologists to determine slices demonstrating infection and slices without an evidence of infection. CT-CAPS framework benefits from a fast and timely labeling process, which is highly valuable when we are facing an early emergence of a new type of data.

CT-CAPS utilizes segmented lung areas as the input of the model instead of the original CT scans. In  order to extract lung areas from the original CT images, a recently developed <a href="https://github.com/JoHof/lungmask"> U-Net based segmentation model</a> is used to preprocess the data. Besides segmenting the lung regions, all images are normalized between 0 and 1, and resized from the original size of [512,512] to [256,256] as the preprocessing step.

<img src="https://github.com/ShahinSHH/CT-CAPS/blob/main/Figures/pipeline.png"/>

<h3>Note : Please don’t use CT-CAPS as the self-diagnostic model without performing a clinical study and consulting with a medical specialist.</h3>

## Dataset
The publically available <a href="https://github.com/ShahinSHH/COVID-CT-MD">COVID-CT-MD dataset</a> is used to train/test the model.
This dataset contains volumetric chest CT scans of 171 patients positive for COVID-19 infection, 60 patients with CAP (Community Acquired Pneumonia), and 76 normal patients. Slice-Level labels (slices with the evidence of infection) are provided in this dataset.

For the detail description of the COVID-CT-MD dataset, please refer to the <a href="https://arxiv.org/abs/2009.14623">https://arxiv.org/abs/2009.14623</a>.

## Lung Segmentation
The lungmask module for the lung segmentation is adopted from <a href="https://github.com/JoHof/lungmask">here</a> and can be installed using the following line of code:
```
pip install git+https://github.com/JoHof/lungmask
```
Make sure to have torch installed in your system. Otherwise you can't use the lungmask module.
<a href = "https://pytorch.org">https://pytorch.org</a>

## Requirements
* Tested with (tensorflow-gpu 2 and keras-gpu 2.2.4) , and (tensorflow 1.14.0 and keras 2.2.4)<br>
-- Try tensorflow.keras instead of keras if it doesn't work in your system.
* Python 3.6
* PyTorch 1.4.0
* Torch 1.5.1
* PyDicom 1.4.2 (<a href="https://pydicom.github.io/pydicom/stable/tutorials/installation.html">Installation<a/>)
* SimpleITK (<a href="https://simpleitk.readthedocs.io/en/v1.1.0/Documentation/docs/source/installation.html">Installation</a>)
* lungmask (<a href="https://github.com/JoHof/lungmask">Installation</a>)
* OpenCV
* OS
* Numpy
* Matplotlib

## Citation
If you found this code and the related paper useful in your research, please consider citing:

```
@article{Heidarian2020,
archivePrefix = {arXiv},
arxivId = {2010.16041},
author = {Heidarian, Shahin and Afshar, Parnian and Enshaei, Nastaran and Naderkhani, Farnoosh and Oikonomou, Anastasia and Atashzar, S. Farokh and Fard, Faranak Babaki and Samimi, Kaveh and Plataniotis, Konstantinos N. and Mohammadi, Arash and Rafiee, Moezedin Javad},
eprint = {2010.16041},
month = {oct},
title = {{COVID-FACT: A Fully-Automated Capsule Network-based Framework for Identification of COVID-19 Cases from Chest CT scans}},
url = {http://arxiv.org/abs/2010.16041},
year = {2020}
}

```
