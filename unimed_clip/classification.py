from pathlib import Path
import argparse
from _open_clip import create_model_and_transforms, HFTokenizer, get_mean_std
import torch 
from tqdm import tqdm
from medmnist import BreastMNIST, ChestMNIST, PneumoniaMNIST, OrganAMNIST, OrganCMNIST
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.preprocessing import label_binarize
import numpy as np
import wandb
import glob
import pandas as pd
from lora import build_LoRA_model, get_lora_parameters
from loralib.utils import lora_state_dict
import loralib.utils as lora_utils


# define input arguments
def parse_args():
    parser = argparse.ArgumentParser(description='CLIP used for classifiction with MedMNIST')
    parser.add_argument('--dataset', type=str, default=None, 
                        help='Dataset to be used for classification: chestmnist, pneumoniamnist, breastmnist, organamnist, organcmnist')
    parser.add_argument('--combination', action='store_true', default=False, help='If the combination of datasets is used')
    parser.add_argument('--datasets_list', type=str, default='chestmnist,pneumoniamnist,breastmnist,organamnist,organcmnist')
    parser.add_argument('--download', action='store_true', default=False, help='Download the datasets')

    parser.add_argument('--checkpoint_path', action="store_true", default=False, help='Load the checkpoint')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint path')
    parser.add_argument("--lora_rank", type=int, default=4, help="Rank of the LoRA matrix")
    parser.add_argument("--wandbName", type=str, default=None, help="distinguish between different runs")
    return parser.parse_args()

class UniMed_Classifier(torch.nn.Module):
    def __init__(self, model, num_classes, task="binary-class"):
        super().__init__()
        self.model = model
        self.classifier = torch.nn.Linear(model.visual.output_dim, num_classes)
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

    if mode == "clf":
        state_dict_lora["model.text_projection"] = state_dict_lora["model.lora_text_projection.weight"]
    else:
        state_dict_lora["text_projection"] = state_dict_lora["lora_text_projection.weight"]

    model.load_state_dict(state_dict_lora, strict=False)
    return model

# load the model
def load_model(device, model_name, checkpoint=None):
    if model_name == "unimed_clip":
        model_name = 'ViT-B-16-quickgelu'
        pretrained_weights = "unimed_clip/unimed_clip_vit_b16.pt"
        text_encoder_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract" # available pretrained weights ["microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract", "microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract"]
        mean, std = get_mean_std()
        model, _, preprocess = create_model_and_transforms(
            model_name,
            pretrained_weights,
            precision='amp',
            device=device,
            force_quick_gelu=True,
            mean=mean,
            std=std,
            text_encoder_name=text_encoder_name,
        )
        print(f"{model_name} loaded on {device}")

    if checkpoint is not None:
        checkpoint_path = f"saved_models/unimed_clip/classification/{checkpoint}"
        state_dict = torch.load(checkpoint_path, map_location=device)
        if "LoRA" in checkpoint:
            # mode = "clf" if "clf" in checkpoint else None
            # model = merge_lora(model, state_dict, mode=mode)
            if "LoRA_Vision" in checkpoint:
                lora_mode = "vision"
            elif "LoRA_projection" in checkpoint:
                lora_mode = "projection"
            model, _, preprocess = create_model_and_transforms(
            model_name,
            pretrained_weights,
            precision="amp",
            device=device,
            force_quick_gelu=True,
            mean=mean,
            std=std,
            text_encoder_name=text_encoder_name,
            lora_rank=args.lora_rank,
            lora_mode=lora_mode
            )
            
            # model.load_state_dict(state_dict)
        if "clf" in checkpoint:
            # nClasses = 2 if "breastmnist" or "pneumoniamnist" in checkpoint else 11
            if "breastmnist" in checkpoint or "pneumoniamnist" in checkpoint:
                nClasses = 2
            elif "organamnist" in checkpoint or "organcmnist" in checkpoint:
                nClasses = 11
            model = UniMed_Classifier(model, num_classes=nClasses, task="binary-class")
            model.load_state_dict(state_dict, strict=False)
        model.load_state_dict(state_dict)
    model.to(device)

    tokenizer = HFTokenizer(
        text_encoder_name,
        context_length=256,
        **{},
    )
    return model, preprocess, tokenizer

    
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
def predictor(datasets_test, model, tokenizer, preprocess, device):
    model.eval()
    ground_truths, predictions = [], []
    for dataset_test in datasets_test:
        nSamples = dataset_test.__len__()
        with tqdm(desc=f"case {0:5d}", total=nSamples, unit='case') as pbar:
            for case, (image, class_id) in enumerate(dataset_test):
                image_input = preprocess(image).unsqueeze(0).to(device)
                if args.checkpoint is not None:
                    if "clf" in args.checkpoint:
                        with torch.cuda.amp.autocast():
                            with torch.no_grad(): 
                                output = model(image_input)
                                indices = torch.argmax(output, dim=-1)
                    else:
                        texts = [tokenizer(cls_text).to(next(model.parameters()).device, non_blocking=True) for cls_text in list(dataset_test.info['label'].values())]
                        text_inputs = torch.cat(texts, dim=0)
                        with torch.no_grad():
                            image_features = model.encode_image(image_input)  #[1, 512]
                            text_features = model.encode_text(text_inputs)  #[n_classes, 512]
                
                            # text_inputs = torch.cat([clip.tokenize(f"this is a photo of {c}") for c in dataset_test.info['label']]).to(device)
                            # with torch.no_grad():
                            #     image_features = model.encode_image(image_input)  #[1, 512]
                            #     text_features = model.encode_text(text_inputs)  #[n_classes, 512]
                            # normalization of features
                            image_features /= image_features.norm(dim=-1, keepdim=True) # [1, 512]
                            text_features /= text_features.norm(dim=-1, keepdim=True) # [2, 512]

                            # similarity matrix, and get accuracy
                            similarity = (model.logit_scale.exp() * image_features @ text_features.T).detach().softmax(dim=-1)
                            values, indices = similarity[0].topk(1)

                else:
                    texts = [tokenizer(cls_text).to(next(model.parameters()).device, non_blocking=True) for cls_text in list(dataset_test.info['label'].values())]
                    text_inputs = torch.cat(texts, dim=0)
                    with torch.no_grad():
                        image_features = model.encode_image(image_input)  #[1, 512]
                        text_features = model.encode_text(text_inputs)  #[n_classes, 512]
            
                        # text_inputs = torch.cat([clip.tokenize(f"this is a photo of {c}") for c in dataset_test.info['label']]).to(device)
                        # with torch.no_grad():
                        #     image_features = model.encode_image(image_input)  #[1, 512]
                        #     text_features = model.encode_text(text_inputs)  #[n_classes, 512]
                        # normalization of features
                        image_features /= image_features.norm(dim=-1, keepdim=True) # [1, 512]
                        text_features /= text_features.norm(dim=-1, keepdim=True) # [2, 512]

                        # similarity matrix, and get accuracy
                        similarity = (model.logit_scale.exp() * image_features @ text_features.T).detach().softmax(dim=-1)
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
    save_doc = Path("saved_models/unimed_clip/classification")
    save_doc.mkdir(parents=True, exist_ok=True)
    if args.checkpoint_path:
        outfile = f"saved_models/unimed_clip/classification/prediction_{args.dataset}.csv"
    else:
        outfile = f"saved_models/unimed_clip/classification/prediction_wo_{args.dataset}.csv"
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

# main function
if __name__ == "__main__":
    # prepare the datasets
    args = parse_args()

    wandb.login(key="c2f51df9b4bf2111388ccee68c2d41d301164d2c")
    wandb.init(
    project="Classification-Finetuning-UniMed-CLIP (Predictions)",
    name=args.wandbName,
    config={
        "task": "classification-finetuning",
        "architecture": "Unimed-CLIP(ViT & Transformer)",
    }
    )

    # load the CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_path = "saved_models/unimed_clip/classification/*"
    if args.checkpoint_path:
        for file in glob.glob(base_path):
            checkpoint = file.split("/")[-1]
            dataset = checkpoint.split("_")[0]
            args.dataset = dataset
            args.checkpoint = checkpoint
            datasets_test = download_datasets(args)
            model, preprocess, tokenizer = load_model(device, "unimed_clip", checkpoint=args.checkpoint)
            predictor(datasets_test, model, tokenizer, preprocess, device)
    else:
        datasets_test = download_datasets(args)   
        model, preprocess, tokenizer = load_model(device, "unimed_clip")
        # prediction
        print("Start prediction...")
        predictor(datasets_test, model, tokenizer, preprocess, device)

    wandb.finish()

    

