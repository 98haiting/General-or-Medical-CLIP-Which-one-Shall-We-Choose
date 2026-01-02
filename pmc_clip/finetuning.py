from pathlib import Path
import argparse
import clip
import torch 
from tqdm import tqdm
from medmnist import BreastMNIST, ChestMNIST, PneumoniaMNIST, OrganAMNIST, OrganCMNIST
from sklearn.preprocessing import label_binarize
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
import torchvision as tv
from PIL import Image
import wandb
import os
from metrices import EvaluationMatrices
from lora import build_LoRA_model, get_lora_parameters
from loralib.utils import lora_state_dict
import loralib.utils as lora_utils
from timm.models.vision_transformer import VisionTransformer as timm_ViT
from pmc_clip.model import blocks, CLIP
from pmc_clip.factory import load_state_dict, load_checkpoint
from pmc_clip.model.pmc_clip_woargs import PMC_CLIP, LoRAPMC_CLIP
from transformers import AutoTokenizer, AutoModel
from torchvision.transforms import InterpolationMode

import json
import math

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

mean = [0.48145466, 0.4578275, 0.40821073]
std = [0.26862954, 0.26130258, 0.27577711]

# define input arguments for fine-tuning
def parse_args():
    parser = argparse.ArgumentParser(description='CLIP used for fine-tuning with MedMNIST')
    parser.add_argument('--dataset', type=str, default=None, 
                        help='Dataset to be used for fine-tuning: chestmnist, pneumoniamnist, breastmnist, organamnist, organcmnist')
    parser.add_argument('--combination', action='store_true', default=False, help='If the combination of datasets is used')
    parser.add_argument('--datasets_list', type=str, default='chestmnist,pneumoniamnist,breastmnist,organamnist,organcmnist')
    parser.add_argument('--download', action='store_true', default=False, help='Download the datasets')

    # add fine-tuning specific hyperparameter arguments
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for fine-tuning')
    parser.add_argument('--epochs', type=int, default=32, help='Number of epochs for fine-tuning')
    parser.add_argument('--lr', type=float, default=5.0e-5, help='Learning rate for fine-tuning')
    parser.add_argument('--lr_visual', type=float, default=5.0e-5, help='Learning rate for vision encoder')
    parser.add_argument('--lr_text', type=float, default=5.0e-5, help='Learning rate for text encoder')
    parser.add_argument('--lr_rest', type=float, default=1.0e-5, help='Learning rate for rest of the model')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='Weight decay for fine-tuning')
    parser.add_argument('--weight_decay_visual', type=float, default=1.0e-3, help='Weight decay for vision encoder')
    parser.add_argument('--weight_decay_text', type=float, default=1.0e-3, help='Weight decay for text encoder')
    parser.add_argument('--weight_decay_rest', type=float, default=0.0, help='Weight decay for rest of the model')
    parser.add_argument('--finetuning', type=str, default=None, help='finetuning strategy: unfreeze_projection, LoRA_Vision, LoRA_Projection, whole')
    parser.add_argument('--lora_rank', type=int, default=4, help='Rank for LoRA')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to the pre-trained model')
    parser.add_argument('--classifier', action='store_true', default=False, help='Add linear classifier to the model')
    parser.add_argument('--wandbID', type=str, default="CLIP-Clf-1", help='Wandb ID for group')
    parser.add_argument('--augmentation', action='store_true', default=False, help='Use data augmentation')
    parser.add_argument('--scheduler', action='store_true', default=False, help='Use learning rate scheduler')
    parser.add_argument('--percentage', type=float, default=None, help="Percentage of the dataset to be used for fine-tuning, if None, all dataset is used")

    # add multiple process arguments: distributed data parallel processing
    # every process - one GPU - initialize model, training - validation, 
    # communication at each iteration for gradient sharing, updating weights separately
    parser.add_argument("--distributed", action="store_true", default=False, help="Use distributed data parallel processing")
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed data parallel processing')

    return parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        # p.data.grad = p.data.grad.float()

def download_datasets(args):
    datasets_train, datasets_val = [], []
    if args.combination is False:
        datapath_train = f"datasets/finetuning/classification/train/{args.dataset}"
        datapath_val = f"datasets/finetuning/classification/val/{args.dataset}"
        datapath_train = Path(datapath_train)
        datapath_val = Path(datapath_val)
        datapath_train.mkdir(parents=True, exist_ok=True)
        datapath_val.mkdir(parents=True, exist_ok=True)

        if args.dataset == 'chestmnist':
            datasets_train.append(ChestMNIST(root=datapath_train, download=args.download, size=224, split='train'))
            datasets_val.append(ChestMNIST(root=datapath_val, download=args.download, size=224, split='val'))
        elif args.dataset == 'pneumoniamnist':
            datasets_train.append(PneumoniaMNIST(root=datapath_train, download=args.download, size=224, split='train'))
            datasets_val.append(PneumoniaMNIST(root=datapath_val, download=args.download, size=224, split='val'))
        elif args.dataset == 'breastmnist':
            datasets_train.append(BreastMNIST(root=datapath_train, download=args.download, size=224, split='train'))
            datasets_val.append(BreastMNIST(root=datapath_val, download=args.download, size=224, split='val'))
        elif args.dataset == 'organamnist':
            datasets_train.append(OrganAMNIST(root=datapath_train, download=args.download, size=224, split='train'))
            datasets_val.append(OrganAMNIST(root=datapath_val, download=args.download, size=224, split='val'))
        elif args.dataset == 'organcmnist':
            datasets_train.append(OrganCMNIST(root=datapath_train, download=args.download, size=224, split='train'))
            datasets_val.append(OrganCMNIST(root=datapath_val, download=args.download, size=224, split='val'))
        else:
            raise ValueError('Invalid dataset name, please choose from chestmnist, pneumoniamnist, breastmnist, organamnist, organcmnist')
        
    else:
        dataset_list = args.datasets_list.split(',')
        for dataset in dataset_list:
            datapath_train = f"datasets/finetuning/classification/train/{dataset}"
            datapath_val = f"datasets/finetuning/classification/val/{dataset}"
            datapath_train = Path(datapath_train)
            datapath_train.mkdir(parents=True, exist_ok=True)
            datapath_val = Path(datapath_val)
            datapath_val.mkdir(parents=True, exist_ok=True)
                               
            if dataset == 'chestmnist':
                datasets_train.append(ChestMNIST(root=datapath_train, download=args.download, size=224, split='train'))
                datasets_val.append(ChestMNIST(root=datapath_val, download=args.download, size=224, split='val'))
            elif dataset == 'pneumoniamnist':
                datasets_train.append(PneumoniaMNIST(root=datapath_train, download=args.download, size=224, split='train'))
                datasets_val.append(PneumoniaMNIST(root=datapath_val, download=args.download, size=224, split='val'))
            elif dataset == 'breastmnist':
                datasets_train.append(BreastMNIST(root=datapath_train, download=args.download, size=224, split='train'))
                datasets_val.append(BreastMNIST(root=datapath_val, download=args.download, size=224, split='val'))
            elif dataset == 'organamnist':
                datasets_train.append(OrganAMNIST(root=datapath_train, download=args.download, size=224, split='train'))
                datasets_val.append(OrganAMNIST(root=datapath_val, download=args.download, size=224, split='val'))
            elif dataset == 'organcmnist':
                datasets_train.append(OrganCMNIST(root=datapath_train, download=args.download, size=224, split='train'))
                datasets_val.append(OrganCMNIST(root=datapath_val, download=args.download, size=224, split='val'))  
            else:
                raise ValueError('Invalid dataset name, please choose from chestmnist, pneumoniamnist, breastmnist, organamnist, organcmnist')
            
    return datasets_train, datasets_val
        
class MedMNISTDataset(Dataset):
    def __init__(self, datasets, percentage, image_size: int):
        self.imgs, self.labels = [], []
        for dataset in datasets:
            self.imgs.extend(dataset.imgs)
            self.labels.extend(dataset.labels)

        if percentage is not None:
            selected_indices = random.sample(range(len(self.imgs)), int(len(self.imgs) * percentage))
            self.imgs = [self.imgs[i] for i in selected_indices]
            self.labels = [self.labels[i] for i in selected_indices]

        self._transforms = tv.transforms.Compose([
            tv.transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
            tv.transforms.CenterCrop(image_size),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean, std)
        ])
        self.label_dict = datasets[0].info["label"]

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img, label = self.imgs[idx], self.labels[idx].astype(int)
        img = self._transforms(Image.fromarray(img).convert('RGB'))
        # img = self._transforms(img.convert('RGB'))

        text = self.label_dict[str(label[0])]

        label = torch.tensor(label, dtype=torch.long)

        return img, label, text

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

def main_worker(datasets_train, datasets_val, device, args):
    # initialize wandb for loggs
    wandb.login(key="c2f51df9b4bf2111388ccee68c2d41d301164d2c")
    wandb.init(
        project="Classification-Finetuning-PMCCLIP",
        name=f"{args.finetuning}-{args.dataset}",
        config={
            "task": "classification-finetuning",
            "architecture": "PMCCLIP(ResNet50 & Bert)",
            "dataset": args.datasets_list if args.combination else args.dataset,
            "finetuning": args.finetuning,
            "learning rate": args.lr,
            "batch size": args.batch_size,
            "epochs": args.epochs,
            "optimizer":"Adam",
            "lora_rank": args.lora_rank
        },
        group=args.wandbID if args.wandbID is not None else None,
        )

    # define checkpoint path for saving the model
    weight_dir = Path("saved_models/pmc_clip/classification")
    weight_dir.mkdir(parents=True, exist_ok=True)
    if args.combination:
        if args.classifier:
            weight_path_best = f"saved_models/pmc_clip/classification/{args.datasets_list}_{args.finetuning}_clf_best.pth"
            weight_path_last = f"saved_models/pmc_clip/classification/{args.datasets_list}_{args.finetuning}_clf_last.pth"
        else:
            weight_path_best = f"saved_models/pmc_clip/classification/{args.datasets_list}_{args.finetuning}_best.pth"
            weight_path_last = f"saved_models/pmc_clip/classification/{args.datasets_list}_{args.finetuning}_last.pth"
    else:
        if args.classifier:
            weight_path_best = f"saved_models/pmc_clip/classification/{args.dataset}_{args.finetuning}_clf_best.pth"
            weight_path_last = f"saved_models/pmc_clip/classification/{args.dataset}_{args.finetuning}_clf_last.pth"
        else:
            weight_path_best = f"saved_models/pmc_clip/classification/{args.dataset}_{args.finetuning}_best.pth"
            weight_path_last = f"saved_models/pmc_clip/classification/{args.dataset}_{args.finetuning}_last.pth"

    # load image encoder, text encoder and text projection layer with original weights
    file = "pmc_clip/pmc_clip/model_configs/RN50_fusion4.json"
    model_cfg = json.load(open(file, 'r'))
    model_class = {
        "CLIP": CLIP,
        "PMC_CLIP": PMC_CLIP,
    }[model_cfg["clip_model"]]
    model_cfg.pop("clip_model")
    model = model_class(**model_cfg, device=device)
    # pretrained checkpoint
    checkpoint_path = 'pmc_clip/checkpoints/checkpoint.pt'
    load_checkpoint(model, checkpoint_path, strict=False)
    model = model.to(device)

    # load finetuning checkpoint
    if args.checkpoint_path is not None:
        if "lora" in args.checkpoint_path:
            lora_state_dict_ = lora_state_dict(torch.load(args.checkpoint_path))
            model.load_state_dict(torch.load(args.checkpoint_path), strict=False)
        else:
            model.load_state_dict(torch.load(args.checkpoint_path))

    # finetuning strategies
    if args.finetuning == "unfreeze_projection":   
        for name, param in model.named_parameters():
            if "visual" in name:
                param.requires_grad = False
            if "text_encoder" in name and "embeddings" not in name:
                param.requires_grad = False
        
    elif "LoRA_Vision" in args.finetuning:
        lora_mode = "vision"
        model = LoRAPMC_CLIP(**model_cfg, device=device, lora_mode=lora_mode, lora_rank=args.lora_rank)
        load_checkpoint(model, checkpoint_path, strict=False, mode="lora")
        lora_utils.mark_only_lora_as_trainable(model) 
        for name, param in model.named_parameters():
            if "positional_embedding" in name:
                param.requires_grad = True
            if "img_special_token" in name:
                param.requires_grad = True
        if "projection" in args.finetuning:
            for name, param in model.named_parameters():
                if "fusion_module" in name:
                    param.requires_grad = True
                if "ln_final" in name:
                    param.requires_grad = True
                if "text_projection" in name:
                    param.requires_grad = True
                if "logit_scale" in name:
                    param.requires_grad = True
    elif args.finetuning == "LoRA_Projection":
        lora_mode = "projection"
        model = LoRAPMC_CLIP(**model_cfg, device=device, lora_mode=lora_mode, lora_rank=args.lora_rank)
        load_checkpoint(model, checkpoint_path, strict=False, mode="lora")
        lora_utils.mark_only_lora_as_trainable(model) 
        for name, param in model.named_parameters():
            if "text_projection" in name:
                param.requires_grad = True
            if "logit_scale" in name:
                param.requires_grad = True
    elif args.finetuning == "whole":
        for param in model.parameters():
            param.requires_grad = True
    
    task = datasets_train[0].info['task']
    if args.classifier:
        model = PMC_CLIP_Classifier(model, num_classes=len(datasets_train[0].info['label']), task=task)
        criterion = torch.nn.CrossEntropyLoss()
    
    sum_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    wandb.config.update({"trainable_parameters": sum_params})
    # for name, param in model.named_parameters():
    #     print(name, param.requires_grad)

    model.to(device)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)   

    # group learning rates
    if args.finetuning == "whole":
        if args.classifier:
            visual_params = model.model.visual.parameters()
            text_params = model.model.text_encoder.parameters()
            fusion_params = model.model.fusion_module.parameters()
            param_groups = [
                {"params": visual_params, "lr": args.lr_visual, "weight_decay": args.weight_decay_visual},
                {"params": text_params, "lr": args.lr_text, "weight_decay": args.weight_decay_text},
                {"params": fusion_params, "lr": args.lr_rest, "weight_decay": args.weight_decay_rest},
                {"params": model.model.logit_scale, "lr": args.lr_rest, "weight_decay": args.weight_decay_rest},
                {"params": model.model.img_special_token, "lr": args.lr_rest, "weight_decay": args.weight_decay_rest},
                {"params": model.model.text_projection, "lr": args.lr_rest, "weight_decay": args.weight_decay_rest},
                {"params": model.model.positional_embedding, "lr": args.lr_rest, "weight_decay": args.weight_decay_rest},
                {"params": model.model.ln_final.parameters(), "lr": args.lr_rest, "weight_decay": args.weight_decay_rest},
                {"params": model.classifier.parameters(), "lr": args.lr_rest, "weight_decay": args.weight_decay_rest}
            ]
        else:
            visual_params = model.visual.parameters()
            text_params = model.text_encoder.parameters()
            fusion_params = model.fusion_module.parameters()
            param_groups = [
                {"params": visual_params, "lr": args.lr_visual, "weight_decay": args.weight_decay_visual},
                {"params": text_params, "lr": args.lr_text, "weight_decay": args.weight_decay_text},
                {"params": fusion_params, "lr": args.lr_rest, "weight_decay": args.weight_decay_rest},
                {"params": model.logit_scale, "lr": args.lr_rest, "weight_decay": args.weight_decay_rest},
                {"params": model.img_special_token, "lr": args.lr_rest, "weight_decay": args.weight_decay_rest},
                {"params": model.text_projection, "lr": args.lr_rest, "weight_decay": args.weight_decay_rest},
                {"params": model.positional_embedding, "lr": args.lr_rest, "weight_decay": args.weight_decay_rest},
                {"params": model.ln_final.parameters(), "lr": args.lr_rest, "weight_decay": args.weight_decay_rest}]
        optimizer = torch.optim.Adam(param_groups, betas=(0.9, 0.999), eps=1.0e-6)
    else:
        # define the optimizer of loss function
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                                            lr=args.lr, 
                                            betas=(0.9, 0.999), 
                                            eps=1.0e-6,
                                            weight_decay=args.weight_decay)
    if args.scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs / 1)

    loss_img = torch.nn.CrossEntropyLoss()
    loss_txt = torch.nn.CrossEntropyLoss()

    # prepare the dataloaders
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(MedMNISTDataset(datasets_train, image_size=224))
        val_sampler = torch.utils.data.distributed.DistributedSampler(MedMNISTDataset(datasets_val, image_size=224))
        train_dataloader = DataLoader(train_sampler, batch_size=args.batch_size, shuffle=True, sampler=train_sampler)
        val_dataloader = DataLoader(val_sampler, batch_size=args.batch_size, shuffle=False, sampler=val_sampler)
    else:
        train_dataloader = DataLoader(MedMNISTDataset(datasets_train, args.percentage, image_size=224), batch_size=args.batch_size, shuffle=True)
        val_dataloader = DataLoader(MedMNISTDataset(datasets_val, args.percentage, image_size=224), batch_size=args.batch_size, shuffle=False)


    # trainer
    training_loss, validation_loss = [], []
    best_validation_loss = 8464
    evaluation = EvaluationMatrices()
    
    if args.distributed:
        torch.distributed.barrier()
    early_stop_counter = 0
    with tqdm(desc=f"Epoch {1:4d}", total=args.epochs) as pbar:
        auc_train_labels, auc_train_predictions = [], []
        auc_val_labels, auc_val_predictions = [], []
        for epoch in range(1, args.epochs + 1): 
            running_train_loss = 0.0
            running_val_loss = 0.0

            running_train_auc = 0.0
            running_val_auc = 0.0

            running_train_acc = 0.0
            running_val_acc = 0.0

            running_train_f1 = 0.0
            running_val_f1 = 0.0

            # training
            model.train()
            for case, (image, label, text) in enumerate(train_dataloader):
                image = image.to(device)   # (b, 3, 224, 224)
                label = label.squeeze(1).to(device)   # (b)

                optimizer.zero_grad()
                
                if args.classifier:
                    with torch.cuda.amp.autocast():
                        predictions = model(image)   # (b, num_classes)
                    if task == "binary-class":
                        _label = torch.eye(2, device=device)[label.long(), :]   # (b, num_classes)
                        _label = _label.to(predictions.dtype)
                    else:
                        _label = label
                    total_loss = criterion(predictions, _label)

                    if task == "binary-class":
                        nLabels = np.arange(len(datasets_train[0].info['label']))
                        predictions_label = predictions.argmax(dim=1)
                        label_binarized = label_binarize(label.detach().cpu().numpy(), classes=nLabels)
                        predictions_binarized = label_binarize(predictions_label.detach().cpu().numpy(), classes=nLabels)
                        auc_train_labels.extend(label_binarized)
                        auc_train_predictions.extend(predictions_binarized)
                        acc = evaluation.accuracy(label_binarized, predictions_binarized)
                        f1 = evaluation.F1_score(label_binarized, predictions_binarized)
                    elif task == "multi-class":
                        nlabels = np.arange(len(datasets_train[0].info['label']))
                        predictions_label = predictions.argmax(dim=1)
                        label_binarized = label_binarize(label.detach().cpu().numpy(), classes=nlabels)
                        predictions_binarized = label_binarize(predictions_label.detach().cpu().numpy(), classes=nlabels)
                        auc_train_labels.extend(label_binarized)
                        auc_train_predictions.extend(predictions_binarized)
                        acc = evaluation.accuracy(label_binarized, predictions_binarized)
                        f1 = evaluation.F1_score(label_binarized, predictions_binarized)

                    running_train_acc += acc
                    running_train_f1 += f1

                else:
                    # logits_per_image, logits_per_text = model(image, target)
                    target_tokens = model.tokenizer(text, padding="max_length", truncation=True, return_tensors="pt", max_length=77)
                    target = target_tokens["input_ids"].to(device)   # (b, 77)
                    with torch.cuda.amp.autocast(): 
                        image_features = model.encode_image(image)   # (b, 512)
                        if isinstance(image_features, dict):
                            image_features = image_features["image_features"]
                        text_features = model.text_encoder(target)   # (b, 512)

                        last_hidden_state= text_features.last_hidden_state
                        pooler_output = text_features.pooler_output
                        text_features = pooler_output @ model.text_projection

                    # normalization
                    image_logits = image_features / image_features.norm(dim=-1, keepdim=True)
                    text_logits = text_features / text_features.norm(dim=-1, keepdim=True)

                    logit_scale = model.logit_scale.exp()
                    logits_per_image = (logit_scale * image_logits @ text_logits.t())   # (b, b)
                    logits_per_text = (logit_scale * text_logits @ image_logits.t())   # (b, b)

                    batch_size = torch.arange(len(image), dtype=torch.long, device=device)   # (b)
                    total_loss = (loss_img(logits_per_image, batch_size) + loss_txt(logits_per_text, batch_size))/2

                    if case == 0:
                        print("Training: AUC, ACC and F1-score are not available without the classifier head.")
                
                total_loss.backward()
                if device.type == "cpu":
                    convert_models_to_fp32(model)
                    optimizer.step()
                    if args.scheduler:
                        scheduler.step()
                    # scheduler.step()
                else:
                    convert_models_to_fp32(model)
                    optimizer.step()
                    if args.scheduler:
                        scheduler.step()
                
                running_train_loss += total_loss.item()

            if args.classifier:
                running_train_auc = evaluation.roc_auc(auc_train_labels, auc_train_predictions, task=task)

            training_loss.append(running_train_loss / len(train_dataloader))
            
            wandb.log({"train_loss": running_train_loss / len(train_dataloader),
                       "train_auc": running_train_auc if args.classifier else None,  
                       "train_acc": running_train_acc / len(train_dataloader) if args.classifier else None,
                       "train_f1": running_train_f1 / len(train_dataloader) if args.classifier else None})                     
            # train_progress.display_summary()
            
            # validation
            model.eval()
            for case, (image, label, text) in enumerate(val_dataloader):
                image = image.to(device)
                label = label.squeeze(1).to(device)
                
                with torch.no_grad():
                    if args.classifier:
                        with torch.cuda.amp.autocast():
                            predictions = model(image)
                        if task == "binary-class":
                            _label = torch.eye(2, device=device)[label.long(), :]
                            _label = _label.to(predictions.dtype)
                        else:
                            _label = label
                        total_loss = criterion(predictions, _label)

                        if task == "binary-class":
                            nLabels = np.arange(len(datasets_train[0].info['label']))
                            predictions_label = predictions.argmax(dim=1)
                            label_binarized = label_binarize(label.detach().cpu().numpy(), classes=nLabels)
                            predictions_binarized = label_binarize(predictions_label.detach().cpu().numpy(), classes=nLabels)
                            auc_val_labels.extend(label_binarized)
                            auc_val_predictions.extend(predictions_binarized)
                            acc = evaluation.accuracy(label_binarized, predictions_binarized)
                            f1 = evaluation.F1_score(label_binarized, predictions_binarized)
  
                        elif task == "multi-class":
                            nlabels = np.arange(len(datasets_train[0].info['label']))
                            predictions_label = predictions.argmax(dim=1)
                            label_binarized = label_binarize(label.detach().cpu().numpy(), classes=nlabels)
                            predictions_binarized = label_binarize(predictions_label.detach().cpu().numpy(), classes=nlabels)
                            auc_val_labels.extend(label_binarized)
                            auc_val_predictions.extend(predictions_binarized)
                            acc = evaluation.accuracy(label_binarized, predictions_binarized)
                            f1 = evaluation.F1_score(label_binarized, predictions_binarized)

                        running_val_acc += acc
                        running_val_f1 += f1
                    
                    else:
                        target_tokens = model.tokenizer(text, padding="max_length", truncation=True, return_tensors="pt", max_length=77)
                        target = target_tokens["input_ids"].to(device)
                        with torch.cuda.amp.autocast():
                            image_features = model.encode_image(image)   # (b, 512)
                            if isinstance(image_features, dict):
                                image_features = image_features["image_features"]
                            text_features = model.text_encoder(target)   # (b, 512)

                            last_hidden_state= text_features.last_hidden_state
                            pooler_output = text_features.pooler_output
                            text_features = pooler_output @ model.text_projection

                        # normalization
                        image_logits = image_features / image_features.norm(dim=-1, keepdim=True)
                        text_logits = text_features / text_features.norm(dim=-1, keepdim=True)

                        logit_scale = model.logit_scale.exp()
                        logits_per_image = (logit_scale * image_logits @ text_logits.t())
                        logits_per_text = (logit_scale * text_logits @ image_logits.t())

                        batch_size = torch.arange(len(image), dtype=torch.long, device=device)   # (b)
                        total_loss = (loss_img(logits_per_image, batch_size) + loss_txt(logits_per_text, batch_size))/2
                        # predictions = logits_per_image.softmax(dim=-1).cpu().numpy()

                        if case == 0:
                            print("Validation: AUC, ACC and F1-score are not available without the classifier head.")
                
                running_val_loss += total_loss.item()

            if args.classifier:
                running_val_auc = evaluation.roc_auc(auc_val_labels, auc_val_predictions, task=task)
            
            validation_loss.append(running_val_loss / len(val_dataloader))
            wandb.log({"val_loss": running_val_loss / len(val_dataloader),
                       "val_auc": running_val_auc if args.classifier else None, 
                       "val_acc": running_val_acc / len(val_dataloader) if args.classifier else None,
                       "val_f1": running_val_f1 / len(val_dataloader) if args.classifier else None})
            # val_progress.display_summary()

            if validation_loss[-1] < best_validation_loss:
                best_validation_loss = validation_loss[-1]
                torch.save(model.state_dict(), weight_path_best)
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            if early_stop_counter >= 7:
                print(f"Early stopping at epoch {epoch}")
                torch.save(model.state_dict(), weight_path_best)
                print("Best model saved.")
                break

            torch.cuda.empty_cache()

            pbar.set_description(f"Epoch {epoch:4d}")
            pbar.set_postfix(
                {
                    "Training Loss": f"{training_loss[-1]:.5f}",
                    "Validation Loss": f"{validation_loss[-1]:.5f}",
                    "Best Validation Loss": f"{best_validation_loss:.5f}"
                }
            )
            pbar.update()
        
        torch.save(model.state_dict(), weight_path_last)
        print("Last model saved.")
        wandb.finish()


# main function
if __name__ == '__main__':
    args = parse_args()
    if args.dataset is None:
        datasets = ["pneumoniamnist", "breastmnist", "organamnist", "organcmnist"]
        for dataset in datasets:
            args.dataset = dataset
            datasets_train, datasets_val = download_datasets(args)

            set_seed(8464)

            # distributed data parallel processing
            if args.distributed:
                torch.distributed.init_process_group(backend='nccl')
                torch.cuda.set_device(args.local_rank)
                device = torch.device("cuda", args.local_rank)
                n_gpu = torch.cuda.device_count()
            else:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                n_gpu = 0
            
            main_worker(datasets_train, datasets_val, device, args)
    else:
        datasets_train, datasets_val = download_datasets(args)

        set_seed(8464)

        # distributed data parallel processing
        if args.distributed:
            torch.distributed.init_process_group(backend='nccl')
            torch.cuda.set_device(args.local_rank)
            device = torch.device("cuda", args.local_rank)
            n_gpu = torch.cuda.device_count()
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            n_gpu = 0
        
        main_worker(datasets_train, datasets_val, device, args)

