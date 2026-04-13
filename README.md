# General or Medical CLIP: Which one shall we choose?
This is the repository of paper: General or Medical CLIP: Which one shall we choose?

![Experiment pipeline](images/pipeline.png)

## Experiments
The experiments include six CLIP-based models: CLIP, Open-CLIP, UniMed-CLIP, Biomed-CLIP, PMC-CLIP and CXR-CLIP, containing zero-shot, fine-tuning testing. All the settings are included in each model folder.

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
| Data                 | link         | comment     | split
| :------------------- | :----------: | :----------: |:----------:
| MedMNIST (Breast)            | [Link](https://medmnist.com/)| 224*224 resolution|546/78/156
| MedMNIST (Pneumonia)            | [Link](https://medmnist.com/)| 224*224 resolution|4708/524/624
| MedMNIST (OrganA)            | [Link](https://medmnist.com/)| 224*224 resolution|34561/6491/17778
| MedMNIST (OrganC)            | [Link](https://medmnist.com/)| 224*224 resolution|12975/2392/8216
| VQA-RAD              | [Link](https://osf.io/89kps/overview)| self-split based on image occurance|1473/354/421
| SLAKE                | [Link](https://huggingface.co/datasets/BoKelvin/SLAKE)| English version only|4919/1053/1061
| ROCO                 | [Link](https://github.com/razorx89/roco-dataset)|Radiology |65414/8171/8176
| ROCO                 | [Link](https://github.com/razorx89/roco-dataset)|Non-radiology |4888/610/610

### Run the Code
* Classification
```
python model_name/classification.py
```

* VQA
```
python model_name/VQA.py --dataset SLAKE --mode test --SLAKE_version english
```

* Image-to-text retrieval
```
python model_name/i2t.py --mode test --data_type radiology
```

### Comparison to SOTA
#### Classification
| Models (Acc.) | Breast | Pneumonia | OrganA | OrganC 
|:--------------|:------:|:---------:|:------:|:-------:
|CLIP           |0.84    |0.89       |0.97    |0.93
|Open-CLIP      |0.85    |0.90       |0.96    |0.92
|UniMed-CLIP    |0.89    |0.97       |0.96    |0.93
|Biomed-CLIP    |0.88    |0.94       |0.97    |0.94
|PMC-CLIP       |0.90    |0.96       |0.96    |0.94
|CXR-CLIP       |0.30    |0.96       |0.93    |0.90
|Google         |0.86    |0.95       |0.89    |0.88
|ResNet         |0.86    |0.86       |0.95    |0.92

#### Visual-Question Answering
|Datasets (Acc.) | CLIP | Open-CLIP | UniMed-CLIP | Biomed-CLIP | PMC-CLIP | CXR-CLIP | PeFoMed | B-GPT
|:---------------|:----:|:---------:|:-----------:| :----------:|:--------:|:--------:|:-------:|:-------:
|VQA-RAD         | 0.53 | 0.51 | 0.47 | 0.48 | 0.55 | 0.70 | 0.82 | -
|SLAKE           | 0.76 | 0.74 | 0.75 | 0.69 | 0.78 | 0.72 | - | 0.86

#### VQA error
| Image | Question | Answer | Model | Prediction | Error type
|:------|:---------|:-------|:-----:|:-----------|:---------:
|![](images/synpic40314.jpg)|What organ is enlarged? | pancreas | CLIP | brain | Cross-modal error