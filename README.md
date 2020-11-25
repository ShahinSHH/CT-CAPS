# CT-CAPS
<h4>Feature Extraction-Based Automated Framework For COVID-19 Disease Identification From Chest CT Scans Using Capsule Networks.</h4>

CT-CAPS is a Capsule Network-based feature extractor developed to detect specific characteristics of CT slices, followed by a Max Pooling Layer to convert slice-level feature maps into patientlevel ones. Finally, a stack of fully connected layers are added to make the final decision. The CT-CAPS is trained on a dataset of CT slices labeled by three experienced radiologists to determine slices demonstrating infection and slices without an evidence of infection. CT-CAPS framework benefits from a fast and timely labeling process, which is highly valuable when we are facing an early emergence of a new type of data.

CT-CAPS utilizes segmented lung areas as the input of the model, instead of the original CT scans. To extract lung areas from the original images, a recently developed <a href="https://github.com/JoHof/lungmask"> U-Net based segmentation model</a>, which works well on the COVID-19 data, has been used at the first step of the CT-CAPS methodology. Besides segmenting the lung regions, all images are normalized between 0 and 1, and resized from the original size of [512,512] to [256,256] as the preprocessing step.

<img src="https://github.com/ShahinSHH/CT-CAPS/blob/main/Figures/pipeline.png"/>

<h3>Note : Please don’t use CT-CAPS as the self-diagnostic model without performing a clinical study and consulting with a medical specialist.</h3>

## Dataset
The publically available <a href="https://github.com/ShahinSHH/COVID-CT-MD">COVID-CT-MD dataset</a> is used to train/test the model.
This dataset contains volumetric chest CT scans of 171 patients positive for COVID-19 infection, 60 patients with CAP (Community Acquired Pneumonia), and 76 normal patients. Slice-Level labels (slices with the evidence of infection) are provided in this dataset.

For the detail description of the COVID-CT-MD dataset, please refer to the <a href="https://arxiv.org/abs/2009.14623">https://arxiv.org/abs/2009.14623</a>.
