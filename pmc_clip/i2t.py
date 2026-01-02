from pathlib import Path
import argparse
import torch 
from tqdm import tqdm
from sklearn.preprocessing import label_binarize
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
import torchvision as tv
from PIL import Image
import wandb
import logging
import os
from metrices import Summary, AverageMeter, ProgressMeter, EvaluationMatrices
from lora import build_LoRA_model, get_lora_parameters
from loralib.utils import lora_state_dict
import loralib.utils as lora_utils
from pmc_clip.model import blocks, CLIP
from pmc_clip.factory import load_state_dict, load_checkpoint
from pmc_clip.model.pmc_clip_woargs import PMC_CLIP, LoRAPMC_CLIP
import json
from typing import NamedTuple
from torchvision.transforms import InterpolationMode
import math
from transformers import get_cosine_schedule_with_warmup

mean = [0.48145466, 0.4578275, 0.40821073]
std = [0.26862954, 0.26130258, 0.27577711]
max_length = 77

# define input arguments for fine-tuning
def parse_args():
    parser = argparse.ArgumentParser(description='CLIP used for fine-tuning with ROCO')
    parser.add_argument('--dataset', type=str, default="ROCO", 
                        help='Dataset to be used for fine-tuning')
    parser.add_argument('--checkpoint_path', type=str, default=None, help="path to checkpoint")
    parser.add_argument('--mode', type=str, default="train", help="whether to train and evaluate the model")
    parser.add_argument('--data_augmentation', action="store_true", default=False, help="whether to use data augmentation")
    parser.add_argument('--data_type', type=str, default="radiology", choices=["radiology", "non-radiology", "all"])
    parser.add_argument('--warmup', action="store_true", default=False, help="whether to use warmup")
    parser.add_argument('--warmup_steps', type=int, default=1, help="number of warmup steps")
    parser.add_argument('--gradient_clip', action='store_true', default=False, help="whether to use gradient_clip to encounter gradient explode")

    # add fine-tuning specific hyperparameter arguments
    parser.add_argument('--batch_size', type=int, default=3, help='Batch size for fine-tuning')
    parser.add_argument('--epochs', type=int, default=32, help='Number of epochs for fine-tuning')
    parser.add_argument('--lr', type=float, default=5.0e-4, help='Learning rate for fine-tuning')
    parser.add_argument('--lr_visual', type=float, default=5.0e-5, help='Learning rate for vision encoder')
    parser.add_argument('--lr_text', type=float, default=5.0e-5, help='Learning rate for text encoder')
    parser.add_argument('--lr_rest', type=float, default=1.0e-5, help='Learning rate for rest of the model')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='Weight decay for fine-tuning')
    parser.add_argument('--weight_decay_visual', type=float, default=1.0e-3, help='Weight decay for vision encoder')
    parser.add_argument('--weight_decay_text', type=float, default=1.0e-3, help='Weight decay for text encoder')
    parser.add_argument('--weight_decay_rest', type=float, default=0.0, help='Weight decay for rest of the model')
    parser.add_argument('--layer_wise', action='store_true', default=False, help="whether to use layer wise finetuning")
    parser.add_argument('--finetuning', type=str, default=None, help='finetuning strategy: normal, lora-vision, vision, projection')
    parser.add_argument('--lora_rank', type=int, default=4, help='Rank for LoRA')
    parser.add_argument('--scheduler', action='store_true', default=False, help='Use learning rate scheduler')
    parser.add_argument('--wandbName', type=str, default=None, help="distinguish between different runs")

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

class ROCOSampler(NamedTuple):
    image_name: str
    image: torch.Tensor
    caption: str

class ROCODatasets(Dataset):
    def __init__(self, args, dataset, mode):
        super(ROCODatasets, self).__init__()
        self.augmentation = args.data_augmentation
        self._transforms_aug = tv.transforms.Compose([
            tv.transforms.ToTensor(),
            tv.transforms.Resize(224, interpolation=tv.transforms.InterpolationMode.BICUBIC),
            tv.transforms.CenterCrop(224),
            tv.transforms.RandomRotation(degrees=(-10.0, 10.0)),
            tv.transforms.RandomHorizontalFlip(0.5),
            tv.transforms.Normalize(mean=mean, std=std)
        ])
        self._transforms = tv.transforms.Compose([
            tv.transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
            tv.transforms.CenterCrop(224),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean, std)
        ])
        self.root_dir = dataset
        self.texts_candidates = []
        if args.data_type == "radiology":
            sample_path = os.path.join(dataset, "radiology", "captions.json")
            self.samples = json.load(open(sample_path, "r"))
            for sample in self.samples:
                image_name = sample["image_name"]
                sample["image_name"] = f"radiology/images/{image_name}.jpg"
                self.texts_candidates.append(sample["caption"])
        elif args.data_type == "non-radiology":
            sample_path = os.path.join(dataset, "non-radiology", "captions.json")
            self.samples = json.load(open(sample_path, "r"))
            for sample in self.samples:
                image_name = sample["image_name"]
                sample["image_name"] = f"non-radiology/images/{image_name}.jpg"
                self.texts_candidates.append(sample["caption"])
        else:
            rad_sample_path = os.path.join(dataset, "radiology", "captions.json")
            non_sample_path = os.path.join(dataset, "non-radiology", "captions.json")
            rad_samples = json.load(open(rad_sample_path, "r"))
            non_samples = json.load(open(non_sample_path, "r"))
            self.samples = rad_samples + non_samples
            
            rad_names, non_names = [], []
            for sample in rad_samples:
                rad_names.append(sample["image_name"])
            for sample in non_samples:
                non_names.append(sample["image_name"])

            for sample in self.samples:
                image_name = sample["image_name"]
                if image_name in rad_names:
                    sample["image_name"] = f"radiology/images/{image_name}.jpg"
                elif image_name in non_names:
                    sample["image_name"] = f"non-radiology/images/{image_name}.jpg"
                else:
                    ValueError("Please check the dataset")
                self.texts_candidates.append(sample["caption"])

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        item = self.samples[idx]
        image_name = item["image_name"]
        caption = item["caption"]
        image = Image.open(os.path.join(self.root_dir, image_name))
        if self.augmentation:
            image = self._transforms_aug(image.convert("RGB"))
        else:
            image = self._transforms(image.convert("RGB"))
        
        sample = ROCOSampler(
            image_name=image_name,
            image=image,
            caption=caption
        )

        return sample

def training(args, device, train_dataset, val_dataset):
    # initialize wandb for loggs
    wandb.login(key="c2f51df9b4bf2111388ccee68c2d41d301164d2c")
    wandb.init(
        project="ITR-Finetuning-PMCCLIP",
        name=f"{args.data_type}-{args.finetuning}",
        config={
            "task": "Image-Text Retrieval-finetuning",
            "architecture": "PMCCLIP(ResNet50 & Bert)",
            "dataset": args.dataset,
            "data type": args.data_type,
            "finetuning": args.finetuning,
            "learning rate": args.lr,
            "batch size": args.batch_size,
            "epochs": args.epochs,
            "optimizer":"AdamW",
            "lora_rank": args.lora_rank,
        }
        )
    
    # define output dir
    weight_dir = Path("saved_models/pmc_clip/i2t")
    weight_dir.mkdir(parents=True, exist_ok=True)
    weight_path_best = f"saved_models/pmc_clip/i2t/{args.data_type}_{args.finetuning}_best.pth"
    weight_path_last = f"saved_models/pmc_clip/i2t/{args.data_type}_{args.finetuning}_last.pth"
    
    # load pretrained model
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

    # preprare Dataloader
    train_data_prepared = ROCODatasets(args, train_dataset, "train")
    val_data_prepared = ROCODatasets(args, val_dataset, "train")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data_prepared, shuffle=True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_data_prepared, shuffle=False)
        train_dataloader = DataLoader(train_data_prepared, batch_size=args.batch_size, sampler=train_sampler)
        val_dataloader = DataLoader(val_data_prepared, batch_size=args.batch_size, shuffle=False, sampler=val_sampler)
    else:
        train_dataloader = DataLoader(train_data_prepared, batch_size=args.batch_size, shuffle=True)
        val_dataloader = DataLoader(val_data_prepared, batch_size=args.batch_size, shuffle=False)

    # finetuning parameters: normal, lora-vision, vision
    if args.finetuning == "vision":
        for name, param in model.text_encoder.named_parameters():
            param.requires_grad = False
    elif args.finetuning == "lora-vision":
        lora_mode = "vision"
        model = LoRAPMC_CLIP(**model_cfg, device=device, lora_mode=lora_mode, lora_rank=args.lora_rank)
        load_checkpoint(model, checkpoint_path, strict=False, mode="lora")
        lora_utils.mark_only_lora_as_trainable(model)
        for name, param in model.named_parameters():
            if "positional_embedding" in name:
                param.requires_grad = True
            if "img_special_token" in name:
                param.requires_grad = True
            if "visual" not in name:
                if "text_encoder" not in name:
                    param.requires_grad = True
    elif args.finetuning == "normal":
        for name, param in model.named_parameters():
            param.requires_grad = True
    elif args.finetuning == "projection":
        for name, param in model.named_parameters():
            if "visual" in name:
                param.requires_grad = False
            if "text_encoder" in name:
                param.requires_grad = False
    else:
        for name, param in model.named_parameters():
            param.requires_grad = True
    
    sum_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    wandb.config.update({"trainable_parameters": sum_params})

    # load checkpoint if necessary and provided
    if args.checkpoint_path is not None:
        if "lora" in args.checkpoint_path:
            lora_state_dict_ = lora_state_dict(torch.load(args.checkpoint_path))
            model.load_state_dict(torch.load(args.checkpoint_path), strict=False)
        else:
            model.load_state_dict(torch.load(args.checkpoint_path))
    model.to(device)

    # group learning rates
    if args.finetuning == "normal":
        if args.layer_wise:
            param_groups = [
                {"params": model.visual.parameters(), "lr":args.lr_visual, "weight_decay":args.weight_decay_visual},
                {"params": model.text_encoder.parameters(), "lr": args.lr_text, "weight_decay":args.weight_decay_text},
                {"params": model.fusion_module.parameters(), "lr": args.lr_rest, "weight_decay":args.weight_decay_rest},
                {"params": model.logit_scale, "lr": args.lr_rest, "weight_decay": args.weight_decay_rest},
                {"params": model.positional_embedding, "lr": args.lr_rest, "weight_decay": args.weight_decay_rest},
                {"params": model.text_projection, "lr": args.lr_rest, "weight_decay": args.weight_decay_rest},
                {"params": model.ln_final.parameters(), "lr": args.lr_rest, "weight_decay": args.weight_decay_rest},
                {"params": model.img_special_token, "lr": args.lr_rest, "weight_decay": args.weight_decay_rest}
            ]
            optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1.0e-6)
        else:
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                         lr=args.lr,
                                         betas=(0.9, 0.999),
                                         eps=1.0e-6,
                                         weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                         lr=args.lr,
                                         betas=(0.9, 0.999),
                                         eps=1.0e-6,
                                         weight_decay=args.weight_decay)
    
    
    if args.scheduler and args.warmup:
        total_train_steps = len(train_dataloader) * args.epochs
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps*len(train_dataloader), num_training_steps=total_train_steps)
    elif args.scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs/1)
    
    loss_img = torch.nn.CrossEntropyLoss()
    loss_text = torch.nn.CrossEntropyLoss()

    training_loss, validation_loss = [], []
    best_validation_loss = 8464
    early_stop_counter = 0

    # start training
    if args.distributed:
        torch.distributed.barrier()
    with tqdm(desc=f"Epoch {1:4d}", total=args.epochs) as pbar:
        for epoch in range(1, args.epochs + 1):
            if args.distributed:
                train_sampler.set_epoch(epoch)
            running_train_loss = 0.0
            running_val_loss = 0.0

            model.train()
            for case, sample in enumerate(train_dataloader):
                image_name = sample.image_name
                image = sample.image
                captions = sample.caption

                image_input = image.to(device)   # [b, 3, 224, 224]
                caption_tokenizer = model.tokenizer(captions, padding="max_length", truncation=True, return_tensors="pt", max_length=max_length)
                caption_token = caption_tokenizer["input_ids"].to(device)
                with torch.cuda.amp.autocast():
                    image_features = model.encode_image(image_input)   # [b, 512]
                    if isinstance(image_features, dict):
                        image_features = image_features["image_features"]

                    logit_scale= model.logit_scale.exp()
                    text_features = model.text_encoder(caption_token)   # [b, 512]
                    pooler_output = text_features.pooler_output
                    text_features = pooler_output @ model.text_projection

                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                logits_per_image = logit_scale * (image_features @ text_features.T)
                logits_per_text = logit_scale * (text_features @ image_features.T)
                
                batch_size = torch.arange(len(image), dtype=torch.long, device=device)
                loss = (loss_img(logits_per_image, batch_size) + loss_text(logits_per_text, batch_size)) / 2
                
                optimizer.zero_grad()

                loss.backward()
                convert_models_to_fp32(model)
                if args.gradient_clip:
                    torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=1.0)
                optimizer.step()
                if args.scheduler and args.warmup:
                    scheduler.step()

                running_train_loss += loss.item()
            
            if args.scheduler and not args.warmup:
                scheduler.step()

            training_loss.append(running_train_loss / len(train_dataloader))
            wandb.log({"train_loss": running_train_loss / len(train_dataloader)})

            model.eval()
            for case, sample in enumerate(val_dataloader):
                image_name = sample.image_name
                image = sample.image
                captions = sample.caption

                image_input = image.to(device)
                caption_tokenizer = model.tokenizer(captions, padding="max_length", truncation=True, return_tensors="pt", max_length=max_length)
                caption_token = caption_tokenizer["input_ids"].to(device)
                
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        image_features = model.encode_image(image_input)   # [b, 512]
                        if isinstance(image_features, dict):
                            image_features = image_features["image_features"]

                        logit_scale= model.logit_scale.exp()
                        text_features = model.text_encoder(caption_token)   # [b, 512]
                        pooler_output = text_features.pooler_output
                        text_features = pooler_output @ model.text_projection

                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                        
                    logits_per_image = logit_scale * (image_features @ text_features.T)
                    logits_per_text = logit_scale * (text_features @ image_features.T)
                    
                    batch_size = torch.arange(len(image), dtype=torch.long, device=device)
                    loss = (loss_img(logits_per_image, batch_size) + loss_text(logits_per_text, batch_size)) / 2
                    
                    running_val_loss += loss.item()
            
            validation_loss.append(running_val_loss / len(val_dataloader))
            wandb.log({"val_loss": running_val_loss / len(val_dataloader)})

            if validation_loss[-1] < best_validation_loss:
                best_validation_loss = validation_loss[-1]
                torch.save(model.state_dict(), weight_path_best)
                early_stop_counter = 0
            else:
                early_stop_counter += 1
            
            if early_stop_counter >= 5:
                print(f"Early stopping at epoch {epoch}")
                break

            torch.cuda.empty_cache()
            pbar.set_description(f"Epoch {epoch:4d}")
            pbar.set_postfix({
                "Training Loss": f"{training_loss[-1]:.5f}",
                "Validation Loss": f"{validation_loss[-1]:.5f}",
                "Best Validation Loss": f"{best_validation_loss:5f}"
            })
            pbar.update()
        
        torch.save(model.state_dict(), weight_path_last)
        print("Last model saved.")
        wandb.finish()
        
def prediction(args, device, test_dataset):
    wandb.login(key="c2f51df9b4bf2111388ccee68c2d41d301164d2c")
    wandb.init(
    project="ITR-Finetuning-PMCCLIP (Predictions)",
    name=args.wandbName,
    config={
        "task": "Image-Text Retrieval-prediction",
        "architecture": "PMCCLIP(ResNet50 & Bert)",
        "lora_rank": args.lora_rank,
    },
    )
    evaluation = EvaluationMatrices()
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

    # load checkpoint
    if args.checkpoint_path is not None:
        loaded_checkpoint_path = f"saved_models/pmc_clip/i2t/{args.checkpoint_path}"
        state_dict = torch.load(loaded_checkpoint_path, map_location=device)
        if "lora" in args.checkpoint_path:
            model = LoRAPMC_CLIP(**model_cfg, device=device, lora_mode="vision", lora_rank=args.lora_rank)
            load_checkpoint(model, checkpoint_path, strict=False, mode="lora")
        model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    test_data_prepared = ROCODatasets(args, test_dataset, "test")
    test_dataloader = DataLoader(test_data_prepared, batch_size=1, shuffle=False)

    texts_candidates = test_data_prepared.texts_candidates

    retrieval_results = []
    ground_truths, candidates, similarity_scores = [], [], []
    with tqdm(desc=f"case {0:5d}", total=len(test_dataloader), unit="case") as pbar:
        for case, sample in enumerate(test_dataloader):
            image_name = sample.image_name
            image = sample.image
            caption_gt = sample.caption

            image_input = image.to(device)
            entry ={}
            entry["image_name"] = image_name[0].split("/")[-1]
            entry["image_type"] = image_name[0].split("/")[0]
            entry["ground_truth"] = caption_gt[0]

            ground_truths.append(caption_gt[0])
            candidates.append(texts_candidates)

            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    # image features from the clip image encoder
                    image_features = model.encode_image(image_input)
                    if isinstance(image_features, dict):
                        image_features = image_features["image_features"]
                    
                    if case == 0:
                        text_tokenized = model.tokenizer(texts_candidates, padding="max_length", truncation=True, return_tensors="pt", max_length=max_length)
                        text_inputs = text_tokenized["input_ids"].to(device)
                        text_features = model.text_encoder(text_inputs)
                        pooler_output = text_features.pooler_output
                        text_features = pooler_output @ model.text_projection
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)

                similarities = math.exp(4.4292) * (image_features @ text_features.T)
                values, indices = similarities[0].topk(10)

            scores = []
            for idx, score in enumerate(similarities[0]):
                scores.append(score.item())
            for value, index in zip(values, indices):
                query = texts_candidates[index.item()]
                entry[query] = value.item()
            
            similarity_scores.append(scores)
            retrieval_results.append(entry)
            
            pbar.set_description(f"case {case:5d}")
            pbar.update()

    # top@k accuracy
    top_1 = evaluation.topk_metrics(ground_truths=ground_truths, candidates=candidates, similarity_scores=similarity_scores, k=1)
    top_5 = evaluation.topk_metrics(ground_truths=ground_truths, candidates=candidates, similarity_scores=similarity_scores, k=5)
    top_10 = evaluation.topk_metrics(ground_truths=ground_truths, candidates=candidates, similarity_scores=similarity_scores, k=10)

    # recall@k accuracy
    recall_1 = evaluation.recallk_metrics(ground_truths=ground_truths, candidates=candidates, similarity_scores=similarity_scores, k=1)
    recall_5 = evaluation.recallk_metrics(ground_truths=ground_truths, candidates=candidates, similarity_scores=similarity_scores, k=5)
    recall_10 = evaluation.recallk_metrics(ground_truths=ground_truths, candidates=candidates, similarity_scores=similarity_scores, k=10)

    doc_dir = Path(f"saved_models/pmc_clip/i2t/result_file")
    doc_dir.mkdir(parents=True, exist_ok=True)
    # save_root = os.path.join(doc_dir, f"results_{args.data_type}_{args.finetuning}.json")
    save_root = f"saved_models/pmc_clip/i2t/result_file/results_{args.finetuning}_{args.data_type}.json"
    if not os.path.exists(os.path.dirname(save_root)):
        os.makedirs(os.path.dirname(save_root))
    json.dump(retrieval_results, open(save_root, "w"))

    print(f"Top@1 Accuracy: {top_1:.2f}%")
    print(f"Top@5 Accuracy: {top_5:.2f}%")
    print(f"Top@10 Accuracy: {top_10:.2f}%")
    print(f"Recall@1 Accuracy: {recall_1:.2f}%")
    print(f"Recall@5 Accuracy: {recall_5:.2f}%")
    print(f"Recall@10 Accuracy: {recall_10:.2f}%")
    wandb.log({
        "DATASET": args.dataset,
        "Finetuning": args.finetuning,
        "Data type": args.data_type,
        "Top-1 Accuracy": f"{top_1:.2f}%",
        "Top-5 Accuracy": f"{top_5:.2f}%",
        "Top-10 Accuracy": f"{top_10:.2f}%",
        "Recall-1": f"{recall_1:.2f}%",
        "Recall-5": f"{recall_5:.2f}%",
        "Recall-10": f"{recall_10:.2f}%",
    })
    wandb.finish()


if __name__ == '__main__':
    args = parse_args()
    set_seed(8464)

    if args.distributed:
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.local_rank = int(os.environ["LOCAL_RANK"])
        print(f"World size: {args.world_size}, Local rank: {args.local_rank}")

        torch.distributed.init_process_group(backend='nccl')
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    base_directory = Path("datasets/I2T retrieval/ROCO/data")
    if args.mode == "train":
        train_path = os.path.join(base_directory, "train")
        val_path = os.path.join(base_directory, "validation")
        training(args, device, train_dataset=train_path, val_dataset=val_path)
    elif args.mode == "test":
        test_path = os.path.join(base_directory, "test")
        prediction(args, device, test_dataset=test_path)