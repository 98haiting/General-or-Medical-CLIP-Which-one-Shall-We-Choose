from pmc_clip.model import blocks, CLIP
from pmc_clip.factory import load_state_dict, load_checkpoint
from pmc_clip.model.pmc_clip_woargs import PMC_CLIP, LoRAPMC_CLIP
from transformers import AutoTokenizer, AutoModel
from torchvision.transforms import Normalize, Resize, CenterCrop, Compose, InterpolationMode, ToTensor
import torch
import torch.nn.functional as F
from PIL import Image
import argparse
from tqdm import tqdm
from medmnist import BreastMNIST, ChestMNIST, PneumoniaMNIST, OrganAMNIST, OrganCMNIST
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.preprocessing import label_binarize
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import json
import math
import wandb
import glob
import pandas as pd

# define input arguments
def parse_args():
    parser = argparse.ArgumentParser(description='CLIP used for classifiction with MedMNIST')
    parser.add_argument('--dataset', type=str, default=None, 
                        help='Dataset to be used for classification: chestmnist, pneumoniamnist, breastmnist, organamnist, organcmnist')
    parser.add_argument('--combination', action='store_true', default=False, help='If the combination of datasets is used')
    parser.add_argument('--datasets_list', type=str, default='chestmnist,pneumoniamnist,breastmnist,organamnist,organcmnist')
    parser.add_argument('--download', action='store_true', default=False, help='Download the datasets')
    parser.add_argument('--threshold', type=float, default=0.9, help='Threshold for multi-label, binary-class task')

    parser.add_argument('--checkpoint_path', action="store_true", default=False, help='Load the checkpoint')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint path')
    parser.add_argument("--lora_rank", type=int, default=4, help="Rank of the LoRA matrix")
    parser.add_argument("--wandbName", type=str, default=None, help="Wandb name for the run")
    return parser.parse_args()


# load the image and text encoder
def load_model_part(device, model_name):
    if model_name == "pmc_clip":
        # image encoder
        image_encoder = blocks.ModifiedResNet(layers=[3, 4, 6, 3], output_dim=768, heads=8, image_size=224, width=64)
        image_encoder.load_state_dict(torch.load('pmc_clip/checkpoints/image_encoder(resnet50).pth', map_location=device))

        # text encoder
        tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract")
        text_encoder = AutoModel.from_pretrained("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract")
        state_dict = torch.load('pmc_clip/checkpoints/text_encoder.pth', map_location=device)
        text_encoder.load_state_dict(state_dict)
        # text_encoder.load_state_dict(torch.load('pmc_clip/checkpoints/text_encoder.pth', map_location=device))

        # text projection layer
        text_projection_layer = torch.load('pmc_clip/checkpoints/text_projection_layer.pth', map_location=device)
        text_projection_layer = torch.nn.Parameter(text_projection_layer)

        print(f"{model_name} loaded successfully on {device}: image_encoder, text_encoder, tokenizer, text_projection_layer")
        state_dict = torch.load('pmc_clip/checkpoints/checkpoint.pt', map_location=device)


        return image_encoder, text_encoder, tokenizer, text_projection_layer

def merge_lora(model, state_dict_lora, mode=None):

    for key in state_dict_lora.keys():
        if "lora_A" in key:
            lora_a = state_dict_lora[key]   # (4, 768)
            lora_b = state_dict_lora[key.replace("lora_A", "lora_B")]   # (2304, 4)
            if ".lora_A" in key:
                base_name = key.replace(".lora_A", "")
            elif "_lora_A" in key:
                base_name = key.replace("_lora_A", "")
            if "weight" in base_name:
                state_dict_lora[base_name] += (lora_b @ lora_a).view(state_dict_lora[base_name].shape)
            else:
                state_dict_lora[base_name+".weight"] += (lora_b @ lora_a).view(state_dict_lora[base_name+".weight"].shape)
    state_dict = state_dict_lora
    if mode == "clf":
        state_dict["model.text_projection"] = state_dict_lora["model.lora_text_projection.weight"].T
    else:
        state_dict["text_projection"] = state_dict_lora["lora_text_projection.weight"].T
    model.load_state_dict(state_dict, strict=False)
    return model

def load_model(device, model_name, checkpoint=None):
    if model_name == "pmc_clip":
        # model config
        file = "pmc_clip/pmc_clip/model_configs/RN50_fusion4.json"
        model_cfg = json.load(open(file, 'r'))
        model_class = {
            "CLIP": CLIP,
            "PMC_CLIP": PMC_CLIP,
        }[model_cfg["clip_model"]]
        model_cfg.pop("clip_model")
        model = model_class(**model_cfg, device=device)
        # load pretrained checkpoint
        checkpoint_path = 'pmc_clip/checkpoints/checkpoint.pt'
        load_checkpoint(model, checkpoint_path, strict=False)

        # load finetuning checkpoint
        if checkpoint is not None:
            checkpoint_path = f"saved_models/pmc_clip/classification/{checkpoint}"
            state_dict = torch.load(checkpoint_path, map_location=device)
            if "LoRA_Vision" in checkpoint:
                lora_mode = "vision"
                model = LoRAPMC_CLIP(**model_cfg, device=device, lora_mode=lora_mode, lora_rank=args.lora_rank)
            elif "LoRA_Projection" in checkpoint:
                lora_mode = "projection"
                model = LoRAPMC_CLIP(**model_cfg, device=device, lora_mode=lora_mode, lora_rank=args.lora_rank)
            else:
                model.load_state_dict(state_dict, strict=False)
            if "clf" in checkpoint:
                if "breastmnist" in checkpoint or "pneumoniamnist" in checkpoint:
                    nClasses = 2
                    task = "binary-class"
                elif "organa" in checkpoint or "organc" in checkpoint:
                    nClasses = 11
                    task = "multi-class"
                model = PMC_CLIP_Classifier(model, nClasses, task=task)
            model.load_state_dict(state_dict)
        model.to(device)
        
        print(f"{model_name} loaded successfully on {device}: model")

        return model
# define image preprocessing
def image_transform(
        image_size: int,
        mean: Optional[Tuple[float, ...]] = None,
        std: Optional[Tuple[float, ...]] = None,
        resize_longest_max: bool = False,
        fill_color: int = 0,
):
    if isinstance(image_size, (list, tuple)) and image_size[0] == image_size[1]:
        # for square size, pass size as int so that Resize() uses aspect preserving shortest edge
        image_size = image_size[0]

    mean = mean or (0.48145466, 0.4578275, 0.40821073)  # OpenAI dataset mean
    std = std or (0.26862954, 0.26130258, 0.27577711)  # OpenAI dataset std
    normalize = Normalize(mean=mean, std=std)

    transforms = [
            Resize(image_size, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(image_size),
        ]
    transforms.extend([
        ToTensor(),
        normalize,
    ])
    return Compose(transforms)

class PMC_CLIP_Classifier(torch.nn.Module):
    def __init__(self, model, num_classes, task="binary-class"):
        super().__init__()
        self.model = model
        self.classifier = torch.nn.Linear(model.visual.output_dim, num_classes)
        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=-1)
        self.task = task

    def forward(self, x):
        features = self.model.encode_image(x)
        if isinstance(features, dict):
            features = features["image_features"]
        predictions = self.classifier(features)
        if self.task == "binary-class":
            predictions = self.sigmoid(predictions)
        elif self.task == "multi-class":
            predictions = self.softmax(predictions)

        return predictions
    
# prepare the datasets:
# ChestMNIST, PneumoniaMNIST, BreastMNIST, OrganAMNIST, OrganCMNIST
def download_datasets(args):
    datasets_test = []
    if args.combination is False:
        datapath = f"datasets/classification/{args.dataset}"
        datapath = Path(datapath)
        datapath.mkdir(parents=True, exist_ok=True)

        if args.dataset == 'chestmnist':
            datasets_test.append(ChestMNIST(root=datapath, download=args.download, size=224, split='test'))
        elif args.dataset == 'pneumoniamnist':
            datasets_test.append(PneumoniaMNIST(root=datapath, download=args.download, size=224, split='test'))
        elif args.dataset == 'breastmnist':
            datasets_test.append(BreastMNIST(root=datapath, download=args.download, size=224, split='test'))
        elif args.dataset == 'organamnist':
            datasets_test.append(OrganAMNIST(root=datapath, download=args.download, size=224, split='test'))
        elif args.dataset == 'organcmnist':
            datasets_test.append(OrganCMNIST(root=datapath, download=args.download, size=224, split='test'))
        else:
            raise ValueError('Invalid dataset name, please choose from chestmnist, pneumoniamnist, breastmnist, organamnist, organcmnist')

    else:
        dataset_list = args.datasets_list.split(',')
        for dataset in dataset_list:
            datapath = f"datasets/classification/{dataset}"
            datapath = Path(datapath)
            datapath.mkdir(parents=True, exist_ok=True)
            if dataset == 'chestmnist':
                datasets_test.append(ChestMNIST(root=datapath, download=args.download, size=224, split='test'))
            elif dataset == 'pneumoniamnist':
                datasets_test.append(PneumoniaMNIST(root=datapath, download=args.download, size=224, split='test'))
            elif dataset == 'breastmnist':
                datasets_test.append(BreastMNIST(root=datapath, download=args.download, size=224, split='test'))
            elif dataset == 'organamnist':
                datasets_test.append(OrganAMNIST(root=datapath, download=args.download, size=224, split='test'))
            elif dataset == 'organcmnist':
                datasets_test.append(OrganCMNIST(root=datapath, download=args.download, size=224, split='test'))
            else:    
                raise ValueError('Invalid dataset name, please choose from chestmnist, pneumoniamnist, breastmnist, organamnist, organcmnist')
            
    return datasets_test


# predictions for load_model_part

def predictor_part(datasets_test, image_encoder, text_encoder, text_projection, tokenizer, device, args):
    model.eval()
    ground_truths, predictions = [], []
    for dataset_test in datasets_test:
        nSamples = dataset_test.__len__()
        with tqdm(desc=f"case {0:5d}", total=nSamples, unit='case') as pbar:
            for case, (image, class_id) in enumerate(dataset_test):
                preprocess = image_transform(image_size=224, )
                image_input = preprocess(image.convert("RGB")).unsqueeze(0).to(device)
                texts = tokenizer([c for c in list(dataset_test.info['label'].values())], padding="max_length", truncation=True, max_length=77, return_tensors='pt')
                text_inputs = texts['input_ids'].to(device)
        
                with torch.no_grad():
                    image_features = image_encoder(image_input)
                    if isinstance(image_features, dict):
                        image_features = image_features['image_features']
                    text_features = text_encoder(text_inputs)

                    # text projection
                    last_hidden_state = text_features.last_hidden_state
                    pooler_output = text_features.pooler_output
                    text_features = pooler_output @ text_projection
                
                # normalization of features
                image_features /= image_features.norm(dim=-1, keepdim=True) # [1, 512]
                text_features /= text_features.norm(dim=-1, keepdim=True) # [2, 512]

                # similarity matrix, and get accuracy
                similarity = (math.exp(4.4292) * image_features @ text_features.T).softmax(dim=-1)
                values, indices = similarity[0].topk(1)

                if dataset_test.info['task'] == "binary-class" or dataset_test.info['task'] == "multi-class":
                    ground_truths.append(class_id.item())
                    predictions.append(indices.item())          
                elif dataset_test.info['task'] == "multi-label, binary-class":
                    sims = image_features @ text_features.T
                    # normalization
                    sim_norm = (sims - torch.mean(sims)) / torch.std(sims)
                    sim_sig = sim_norm.sigmoid()
                    ground_truths.append(class_id)
                    predictions.append(sim_sig.cpu().numpy()[0])

                pbar.set_description(f"case {case:5d}")
                pbar.update()


    # AUC and accuracy score
    if dataset_test.info['task'] == "binary-class":    
        auc = roc_auc_score(ground_truths, predictions)
        accuracy = accuracy_score(ground_truths, predictions) * 100
        f1 = f1_score(ground_truths, predictions) * 100
    elif dataset_test.info['task'] == "multi-class":
        labels = list(dataset_test.info['label'].keys())
        labels = [int(label) for label in labels]
        ground_truths = np.array(ground_truths)
        predictions = np.array(predictions)
        predictions_binarized = label_binarize(predictions, classes=labels)
        auc = roc_auc_score(ground_truths, predictions_binarized, multi_class='ovr')
        accuracy = accuracy_score(ground_truths, predictions) * 100
        f1 = f1_score(ground_truths, predictions, average='weighted') * 100
    elif dataset_test.info['task'] == "multi-label, binary-class":
        ground_truths = np.array(ground_truths)
        predictions = np.array(predictions)
        auc = roc_auc_score(ground_truths, predictions)
        # set a threshold to be 0.9
        predictions_binarized = np.where(predictions > args.threshold, 1, 0)
        accuracy = accuracy_score(ground_truths, predictions_binarized) * 100
        f1 = f1_score(ground_truths, predictions) * 100
        
    print(f"AUC score: {auc:.2f}, Accuracy: {accuracy:.2f}%, f1_score: {f1:.2f}%")

def predictor(datasets_test, model, device, args):
    model.eval()
    ground_truths, predictions = [], []
    for dataset_test in datasets_test:
        nSamples = dataset_test.__len__()
        with tqdm(desc=f"case {0:5d}", total=nSamples, unit='case') as pbar:
            for case, (image, class_id) in enumerate(dataset_test):
                preprocess = image_transform(image_size=224, )
                image_input = preprocess(image.convert("RGB")).unsqueeze(0).to(device)
                if args.checkpoint is not None:
                    if "clf" in args.checkpoint:
                        with torch.cuda.amp.autocast():
                            with torch.no_grad():
                                output = model(image_input)
                                indices = torch.argmax(output, dim=-1)
                    else:
                        texts = model.tokenizer([c for c in list(dataset_test.info['label'].values())], padding="max_length", truncation=True, max_length=77, return_tensors='pt')
                        text_inputs = texts['input_ids'].to(device)

                        with torch.no_grad():
                            image_features = model.encode_image(image_input)
                            if isinstance(image_features, dict):
                                image_features = image_features['image_features']
                            text_features = model.text_encoder(text_inputs)

                            # text projection
                            last_hidden_state = text_features.last_hidden_state
                            pooler_output = text_features.pooler_output
                            text_features = pooler_output @ model.text_projection
                
                            # normalization of features
                            image_features /= image_features.norm(dim=-1, keepdim=True) # [1, 512]
                            text_features /= text_features.norm(dim=-1, keepdim=True) # [2, 512]

                            # similarity matrix, and get accuracy
                            similarity = (model.logit_scale.mean() * image_features @ text_features.T).softmax(dim=-1)
                            values, indices = similarity[0].topk(1)
                else:
                    texts = model.tokenizer([c for c in list(dataset_test.info['label'].values())], padding="max_length", truncation=True, max_length=77, return_tensors='pt')
                    text_inputs = texts['input_ids'].to(device)
                    with torch.cuda.amp.autocast():
                        with torch.no_grad():
                            image_features = model.encode_image(image_input)
                            if isinstance(image_features, dict):
                                image_features = image_features['image_features']
                            text_features = model.text_encoder(text_inputs)

                            # text projection
                            last_hidden_state = text_features.last_hidden_state
                            pooler_output = text_features.pooler_output
                            text_features = pooler_output @ model.text_projection
                
                            # normalization of features
                            image_features /= image_features.norm(dim=-1, keepdim=True) # [1, 512]
                            text_features /= text_features.norm(dim=-1, keepdim=True) # [2, 512]

                            # similarity matrix, and get accuracy
                            similarity = (model.logit_scale.mean() * image_features @ text_features.T).softmax(dim=-1)
                            values, indices = similarity[0].topk(1)

                if dataset_test.info['task'] == "binary-class" or dataset_test.info['task'] == "multi-class":
                    ground_truths.append(class_id.item())
                    predictions.append(indices.item())          
                elif dataset_test.info['task'] == "multi-label, binary-class":
                    sims = image_features @ text_features.T
                    # normalization
                    sim_norm = (sims - torch.mean(sims)) / torch.std(sims)
                    sim_sig = sim_norm.sigmoid()
                    ground_truths.append(class_id)
                    predictions.append(sim_sig.cpu().numpy()[0])

                pbar.set_description(f"case {case:5d}")
                pbar.update()


    # AUC and accuracy score
    if dataset_test.info['task'] == "binary-class":    
        auc = roc_auc_score(ground_truths, predictions)
        accuracy = accuracy_score(ground_truths, predictions) * 100
        f1 = f1_score(ground_truths, predictions) * 100
    elif dataset_test.info['task'] == "multi-class":
        labels = list(dataset_test.info['label'].keys())
        labels = [int(label) for label in labels]
        ground_truths = np.array(ground_truths)
        predictions = np.array(predictions)
        predictions_binarized = label_binarize(predictions, classes=labels)
        auc = roc_auc_score(ground_truths, predictions_binarized, multi_class='ovr')
        accuracy = accuracy_score(ground_truths, predictions) * 100
        f1 = f1_score(ground_truths, predictions, average='weighted') * 100
    elif dataset_test.info['task'] == "multi-label, binary-class":
        ground_truths = np.array(ground_truths)
        predictions = np.array(predictions)
        auc = roc_auc_score(ground_truths, predictions)
        # set a threshold to be 0.9
        predictions_binarized = np.where(predictions > args.threshold, 1, 0)
        accuracy = accuracy_score(ground_truths, predictions_binarized) * 100
        f1 = f1_score(ground_truths, predictions) * 100
    
    image_ids = np.arange(len(ground_truths))
    label_maps = dataset_test.info['label']
    predictions_label = [label_maps[str(pred)] for pred in predictions]
    gt_label = [label_maps[str(gt)] for gt in ground_truths]
    df = pd.DataFrame({
        "image_id": image_ids,
        "prediction label": predictions_label,
        "ground_truth label": gt_label,
        "prediction": predictions,
        "ground_truth": ground_truths
    })
    save_doc = Path("saved_models/pmc_clip/classification")
    save_doc.mkdir(parents=True, exist_ok=True)
    outfile = f"saved_models/pmc_clip/classification/prediction_{args.dataset}.csv"
    df.to_csv(outfile, index=False)
        
    print("**********")
    if args.checkpoint is not None: 
        print(f"DATASET: {args.dataset}, METHOD: {args.checkpoint}, CLASSIFIER: {True if 'clf' in args.checkpoint else False}, LORA: {args.lora_rank}") 
    print(f"AUC score: {auc:.2f}, Accuracy: {accuracy:.2f}%, f1_score: {f1:.2f}%")
    if args.checkpoint is not None:
        wandb.log({"DATASET": args.dataset, 
                "METHOD": args.checkpoint,
                "CLASSIFIER": True if 'clf' in args.checkpoint else False,
                "LORA": args.lora_rank,
                "AUC": f"{auc:.2f}",
                "Accuracy": f"{accuracy:.2f}%",
                "f1_score": f"{f1:.2f}%"})
    print("**********") 

# define main function
if __name__ == "__main__":
    # prepare the datasets
    args = parse_args()
    wandb.login(key="c2f51df9b4bf2111388ccee68c2d41d301164d2c")
    wandb.init(
    project="Classification-Finetuning-PMCCLIP (Predictions)",
    name=args.wandbName,
    config={
        "task": "classification-finetuning",
        "architecture": "PMCCLIP(ResNet & Bert)",
        "lora_rank": args.lora_rank,
    },
    )

    # load the PMC-CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    """
    image_encoder, text_encoder, tokenizer, text_projection_layer = load_model_part(device, "pmc_clip")
    image_encoder.to(device)
    text_encoder.to(device)
    text_projection_layer.to(device)
    print("Start prediction...")
    predictor_part(datasets_test, image_encoder, text_encoder, text_projection_layer, tokenizer, device, args)
    """
    base_path = "saved_models/pmc_clip/classification/*"
    if args.checkpoint_path:
        for file in glob.glob(base_path):
            checkpoint = file.split("/")[-1]
            dataset = checkpoint.split("_")[0]
            args.dataset = dataset
            args.checkpoint = checkpoint
            datasets_test = download_datasets(args)

            model = load_model(device, "pmc_clip", checkpoint=checkpoint)
            predictor(datasets_test, model, device, args)
    else:
        datasets_test = download_datasets(args)
        model = load_model(device, "pmc_clip")
        print("Start prediction...")
        # prediction
        predictor(datasets_test, model, device, args)

    # wandb.finish()
    