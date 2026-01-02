from pathlib import Path
import argparse
from cxrclip.model import CXRClip, LoRACXRClip
from cxrclip.data.data_utils import load_tokenizer
import torch 
from tqdm import tqdm
from medmnist import BreastMNIST, PneumoniaMNIST, OrganAMNIST, OrganCMNIST, ChestMNIST
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.preprocessing import label_binarize
import numpy as np
from typing import Optional, Tuple
from torchvision.transforms import Normalize, Resize, CenterCrop, Compose, InterpolationMode, ToTensor
from cxrclip.prompt import constants
from transformers import AutoModel
from collections import OrderedDict
import wandb
import glob
import pandas as pd

# define input arguments
def parse_args():
    parser = argparse.ArgumentParser(description='CLIP used for classifiction with MedMNIST')
    parser.add_argument('--dataset', type=str, default=None, 
                        help='Dataset to be used for classification: pneumoniamnist, breastmnist, organamnist, organcmnist')
    parser.add_argument('--combination', action='store_true', default=False, help='If the combination of datasets is used')
    parser.add_argument('--datasets_list', type=str, default='pneumoniamnist,breastmnist,organamnist,organcmnist')
    parser.add_argument('--download', action='store_true', default=False, help='Download the datasets')

    parser.add_argument('--checkpoint_path', action="store_true", default=False, help='Load the checkpoint')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint path')
    parser.add_argument("--lora_rank", type=int, default=4, help="Rank of the LoRA matrix")
    parser.add_argument("--wandbName", type=str, default=None, help="Wandb name for the run")
    return parser.parse_args()


# load the model
def load_model(device, model_name, checkpoint=None):
    if model_name == "cxr_clip":
        ckpt_path = "cxr_clip/checkpoints/r50_mcc.tar"
        ckpt = torch.load(ckpt_path, map_location=device)
        ckpt_config = ckpt["config"]

        tokenizer_config = ckpt_config["tokenizer"] if "tokenizer" in ckpt_config else None
        tokenizer = load_tokenizer(**tokenizer_config)
        text_encoder = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

        if checkpoint is not None and "clf" in checkpoint:
            n_classes = 2 if "breastmnist" in checkpoint or "pneumoniamnist" in checkpoint else 11
        else:
            n_classes = None
        model = CXRClip(model_config=ckpt_config["model"], all_loss_config=ckpt_config["loss"], tokenizer=tokenizer, classifier=n_classes)
        if checkpoint is not None:
            model.load_state_dict(ckpt["model"], strict=False)
            new_checkpoint_path = f"saved_models/cxrclip/classification/{checkpoint}"
            new_state_dict = torch.load(new_checkpoint_path, map_location=device)
            
                # model.load_state_dict(new_state_dict, strict=False)
            if "LoRA_Vision" in checkpoint:
                lora_mode = "vision"
                model = LoRACXRClip(model_config=ckpt_config["model"], all_loss_config=ckpt_config["loss"], tokenizer=tokenizer, r=args.lora_rank, lora_mode=lora_mode, classifier=n_classes)

                model.load_state_dict(new_state_dict, strict=False)

            elif "LoRA_Projection" in checkpoint:
                lora_mode = "projection"
                model = LoRACXRClip(model_config=ckpt_config["model"], all_loss_config=ckpt_config["loss"], tokenizer=tokenizer, r=args.lora_rank, lora_mode=lora_mode, classifier=n_classes)
                model.load_state_dict(new_state_dict, strict=False)       
            else:
                model.load_state_dict(new_state_dict, strict=False)
        else:
            model.load_state_dict(ckpt["model"], strict=False)
        text_dict = {}
        for key in ckpt["model"]:
            if "text_encoder" in key:
                new_key = key.replace("text_encoder.", "")
                text_dict[new_key] = ckpt["model"][key]
        text_encoder.load_state_dict(text_dict, strict=False)

        model.to(device)
        text_encoder.to(device)

    return model, tokenizer, text_encoder, ckpt_config

class CXRCLIP_Classifier(torch.nn.Module):
    def __init__(self, model, num_classes, task="binary-class"):
        super().__init__()
        self.model = model
        self.classifier = torch.nn.Linear(model.image_encoder.out_dim, num_classes)
        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=-1)
        self.task = task

    def forward(self, x):
        features = self.model.encode_image(x)
        predictions = self.classifier(features)
        if self.task == "binary-class":
            predictions = self.sigmoid(predictions)
        elif self.task == "multi-class":
            predictions = self.softmax(predictions)

        return predictions

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
                # TODO: check this part to make it work
                state_dict_lora[base_name] += (lora_b @ lora_a).view(state_dict_lora[base_name].shape)
            else:
                state_dict_lora[base_name+".weight"] += (lora_b @ lora_a).view(state_dict_lora[base_name+".weight"].shape)
    """
    if mode == "clf":
        state_dict_lora["model.text_projection"] = state_dict_lora["model.lora_text_projection.weight"]
    else:
        state_dict_lora["text_projection"] = state_dict_lora["lora_text_projection.weight"]
    """
    model.load_state_dict(state_dict_lora, strict=False)
    return model

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


# predictions
def predictor(datasets_test, model, tokenizer, text_encoder, ckpt_config, device):
    model.eval()
    ground_truths, predictions = [], []
    for dataset_test in datasets_test:
        nSamples = dataset_test.__len__()
        with tqdm(desc=f"case {0:5d}", total=nSamples, unit='case') as pbar:
            for case, (image, class_id) in enumerate(dataset_test):
                preprocess = image_transform(image_size=224, )
                image_input = preprocess(image.convert("RGB")).unsqueeze(0).to(device)
                
                if hasattr(constants, dataset_test.flag.upper()):
                    class_list = getattr(constants, dataset_test.flag.upper())
                
                if args.checkpoint is not None and "clf" in args.checkpoint:
                    with torch.cuda.amp.autocast():
                        with torch.no_grad():
                            image_features = model.encode_image(image_input)
                            output = model.classifier(image_features)
                            output = output.softmax(dim=-1)
                            indices = torch.argmax(output, dim=-1)
                else:
                    with torch.no_grad():
                        image_emb = model.encode_image(image_input)
                        image_emb = model.image_projection(image_emb) if model.projection else image_emb
                        image_features = image_emb / torch.norm(image_emb, dim=1, keepdim=True)
                        
                        if dataset_test.info["task"] == "binary-class":
                            for class_name in class_list:
                                prompts = ["No "+ class_name, class_name]

                                text_emb = model.encode_text(tokenizer(prompts, padding="longest", truncation=True, return_tensors="pt", max_length=ckpt_config["base"]["text_max_length"]).to(device))
                                text_emb = model.text_projection(text_emb) if model.projection else text_emb
                                text_features = text_emb / torch.norm(text_emb, dim=1, keepdim=True)
                        elif dataset_test.info["task"] == "multi-class":
                            prompts = class_list
                            if isinstance(prompts, str) or isinstance(prompts, list):
                                text_tokens = tokenizer(prompts, padding="longest", truncation=True, return_tensors="pt", max_length=ckpt_config["base"]["text_max_length"])
                                text_token = text_tokens["input_ids"]
                                text_emb = text_encoder(text_token.to(device))["last_hidden_state"]

                            if model.text_pooling == "eos":
                                eos_token_indices = text_tokens["attention_mask"].sum(dim=-1) - 1
                                text_emb = text_emb[torch.arange(text_emb.shape[0]), eos_token_indices]
                                text_emb = model.text_projection(text_emb) if model.projection else text_emb
                                text_features = text_emb / torch.norm(text_emb, dim=1, keepdim=True)

                        # similarity matrix, and get accuracy
                        similarity = (model.logit_scale.exp() * image_features @ text_features.T).softmax(dim=1)
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
        predictions_binarized = np.where(predictions > 0.6, 1, 0)
        accuracy = accuracy_score(ground_truths, predictions_binarized) * 100
        f1 = f1_score(ground_truths, predictions_binarized)

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
    save_doc = Path("saved_models/cxrclip/classification")
    save_doc.mkdir(parents=True, exist_ok=True)
    outfile = f"saved_models/cxrclip/classification/prediction_{args.dataset}.csv"
    df.to_csv(outfile, index=False)
        
    print("**********")    
    print(f"DATASET: {args.dataset}, METHOD: {args.checkpoint}, CLASSIFIER: {True if 'clf' in args.checkpoint else False}, LORA: {args.lora_rank}")
    print(f"AUC score: {auc:.2f}, Accuracy: {accuracy:.2f}%, f1_score: {f1:.2f}%")
    wandb.log({"DATASET": args.dataset, 
               "METHOD": args.checkpoint,
               "CLASSIFIER": True if 'clf' in args.checkpoint else False,
               "LORA": args.lora_rank,
               "AUC": f"{auc:.2f}",
               "Accuracy": f"{accuracy:.2f}%",
               "f1_score": f"{f1:.2f}%"})
    print("**********") 


# main function
if __name__ == "__main__":
    # prepare the datasets
    args = parse_args()

    wandb.login(key="c2f51df9b4bf2111388ccee68c2d41d301164d2c")
    wandb.init(
    project="Classification-Finetuning-CXRCLIP (Predictions)",
    name=args.wandbName,
    config={
        "task": "classification-finetuning",
        "architecture": "CXRCLIP(ResNet50 & Bert)",
        "lora_rank": args.lora_rank,
    },
    )

    # load the CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_path = "saved_models/cxrclip/classification/*"
    if args.checkpoint_path:
        for file in glob.glob(base_path):
            checkpoint = file.split("/")[-1]
            dataset = checkpoint.split("_")[0]
            args.dataset = dataset
            args.checkpoint = checkpoint
            datasets_test = download_datasets(args)
            model, tokenizer, text_encoder, ckpt_config = load_model(device, "cxr_clip", checkpoint=checkpoint)
            predictor(datasets_test, model, tokenizer, text_encoder, ckpt_config, device)
    else:
        model, tokenizer, text_encoder, ckpt_config = load_model(device, "cxr_clip")
        datasets_test = download_datasets(args)
        predictor(datasets_test, model, tokenizer, text_encoder, ckpt_config, device)
    
    # prediction
    wandb.finish()
    
    


    

