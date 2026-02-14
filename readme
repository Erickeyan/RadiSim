## Learning Like a Radiologist: A Medical Vision-Language Model for Radiological Image Analysis via Curriculum Learning


### Introduction

We curate RadiSim, a 12-million image-text pair dataset aligned to these phases. We evaluate the model using a five-stage coarse-to-fine validation framework: (1) modality recognition, (2) anatomical recognition, (3) anatomical localization, (4) abnormality and disease diagnosis, and (5) disease differentiation and grading. This framework spans 24 zero-shot subtasks across MR, CT, and DR imaging. RadiSim-CL achieves comparable performance to state-of-the-art baselines in both foundational and anatomical tasks, and demonstrates superior capabilities in complex reasoning (e.g., an AUC of 0.953 for brain tumor diagnosis and an accuracy of 0.764 for meningioma grading). Ablation studies further confirm the curriculum's effectiveness. RadiSim-CL thus offers a scalable, clinically aligned solution to enhance diagnostic precision.

### Dataset

| Dataset | Modality | Description | Link |
| :--- | :--- | :--- | :--- |
| **PMC-OA dataset** | DR, CT, MRI | radiological image-text pairs | [Link]() |
| **RadImageNet** | CT, MRI | descriptions enhanced via qwen-turbo | [Link](https://github.com/BMEII-AI/RadImageNet) |
| **CheXpert** | Chest X-ray | 14 pathologies; image-text pairs generated via qwen-turbo | [Link](https://stanfordmlgroup.github.io/competitions/chexpert/) |
| **Eurorad** | MRI | Peer-reviewed radiology case reports from the European Society of Radiology (ESR) | [Link](https://www.eurorad.org/) |
| **MIMIC-CXR** | Chest X-ray | Training split; processed using XrayGPT framework with report summaries | [Link](https://physionet.org/content/mimic-cxr/2.0.0/) |
| **OpenI** | Chest X-ray | Indiana University Hospital collection; image-text pairs | [Link](https://openi.nlm.nih.gov/) |


Our downstream validation is conducted on a diverse range of public and internal datasets, including **[CHAOS](https://chaos.grand-challenge.org/)**, **[AMOS](https://amos22.grand-challenge.org/)**, **[TotalSegmentator](https://github.com/wasserth/TotalSegmentator)**, **[CT-ORG](https://wiki.cancerimagingarchive.net/display/Public/CT-ORG%3A+CT+volumes+with+multiple+organ+segmentations)**, **[LiTS (MediMeTA)](https://competitions.codalab.org/competitions/17094)**, **[Br35H](https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection)**, **[MRNet](https://stanfordmlgroup.github.io/competitions/mrnet/)**, **[Brain Hemorrhage CT](https://physionet.org/content/ct-ich/1.3.1/)**, **[COVID-19 CT](https://github.com/UCSD-AI4H/COVID-CT)**, **[CT Lymph Nodes](https://wiki.cancerimagingarchive.net/display/Public/CT+Lymph+Nodes)**, **[PneumoniaMNIST](https://medmnist.com/)**, **[RSNA Pneumonia](https//www.kaggle.com/c/rsna-pneumonia-detection-challenge)**, and **[Kaggle Brain Tumor MRI](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)**, as well as proprietary internal clinical collections for modality recognition and disease grading.

### Performance
![Performance](Images/performance.png)
### Get start

### Environment

- Python >= 3.9 (recommended: 3.11)
- PyTorch >= 2.2 (match with your CUDA version)
- openc-clip-torch == 2.24
- Common packages: `tqdm`, `numpy`, `scikit-learn`, `einops`, `matplotlib`

## Installation

```bash
conda create -n radisimcl python=3.11 -y
conda activate radisimcl
pip install -r requirements.txt
```

## Distributed training:

```bash
torchrun --nproc_per_node=4 ./scripts/run.py
```

## Pretrained model and processed datasets
Our pretrained model and some processed datasets are availble at .


### Acknowledgements

Some codes are borrowed from [openclip](https://github.com/mlfoundations/open_clip)
We thank collaborators, data contributors, and supporting organizations.

---

## License

This project is released under the Apache 2.0 license. Please see the [LICENSE](LICENSE) file for more information.

