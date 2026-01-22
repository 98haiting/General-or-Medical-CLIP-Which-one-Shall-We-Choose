# General or Medical CLIP: Which one shall we choose?
This is the repository of paper: General or Medical CLIP: Which one shall we choose?

![Experiment pipeline](images/pipeline.png)

## Experiments
The experiments include four CLIP-based models: CLIP, Biomed-CLIP, PMC-CLIP and CXR-CLIP, containing zero-shot, fine-tuning testing. All the settings are included in each model folder.

### Environment Setup
```
conda env create --file environment.yml
conda activate CompFM
```

### Code Structure
- model_name
  - VQA.py: finetuning/prediction for VQA
  - classification.py: prediction for classification
  - finetuning.py: finetuning for classification
  - i2t.py: finetuning/prediction for I2T

### Data Preparation
| Data                 | link         | comment     |
| :------------------- | :----------: | ----------: |
| MedMNIST             | [Link](https://medmnist.com/)| 224*224 resolution|
| VQA-RAD              | [Link](https://osf.io/89kps/overview)| self-split based on image occurance|
| SLAKE                | [Link](https://huggingface.co/datasets/BoKelvin/SLAKE)| English version only|
| ROCO                 | [Link](https://github.com/razorx89/roco-dataset)|Radiology & Non-radiology|

### Run the Code
Classification
```
python model_name/classification.py
```

VQA
```
python model_name/VQA.py --dataset SLAKE --mode test --SLAKE_version english
```

Image-to-text retrieval
```
python model_name/i2t.py --mode test --data_type radiology
```

## Citation
```
@inproceedings{
      huang2025general,
      title={General or Medical {CLIP}, Which one Shall We Choose?},
      author={Haiting Huang and Emmanuelle Salin and Dario Zanca and Bjoern Eskofier},
      booktitle={Submitted to Medical Imaging with Deep Learning},
      year={2025},
      url={https://openreview.net/forum?id=4DfIpoTtHk},
      note={under review}
}
```
