# Comparison-between-CLIPs-in-Healthcare-using-Explainability-Methods
This is the repository of paper: General or Medical CLIP: Which one shall we choose?

![Experiment pipeline](images/pipeline.png)

## Experiments
The experiments include four CLIP-based models: CLIP, Biomed-CLIP, PMC-CLIP and CXR-CLIP, containing zero-shot, fine-tuning testing. All the settings are included in each model folder.

### Environment Setup
```
conda env create --file environment.yml
conda activate CompFM
python model_name/task
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
