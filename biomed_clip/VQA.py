from pathlib import Path
import argparse
from open_clip import create_model_from_pretrained, get_tokenizer
import torch 
import torch.nn as nn 
from tqdm import tqdm
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
import torchvision as tv
from PIL import Image
import wandb
from lora import BiomedCLIPViT_LoRA, build_lora_projection
from metrices import EvaluationMatrices
# from lora import build_LoRA_model, get_lora_parameters
from loralib.utils import lora_state_dict
from vqa import attention
from vqa.bc import BCNet
from vqa.fc import FCNet
import pickle
from vqa.dataset_RAD import Dictionary
from vqa.classifier import SimpleClassifier
import pandas as pd
from typing import NamedTuple
from torch.nn.functional import normalize
from sklearn.model_selection import train_test_split
import json
import os
# from monai.transforms import Resize, CenterSpatialCrop, ToTensor, NormalizeIntensity, Compose, RandFlip, RandRotate, RandZoom, ScaleIntensity

mean = [0.48145466, 0.4578275, 0.40821073]
std = [0.26862954, 0.26130258, 0.27577711]

def parse_args():
    parser = argparse.ArgumentParser(description='CLIP used for VQA Task')
    parser.add_argument('--dataset', type=str, default=None, 
                        help='Dataset to be used for classification: VQA-RAD, SLAKE')
    parser.add_argument('--dataset_path', type=str, default=None, help="path to load dataset")
    parser.add_argument('--checkpoint_path', type=str, default=None, help="path to checkpoint")
    parser.add_argument('--mode', type=str, default="train", help="whether to train and evaluate the model")
    parser.add_argument('--RAD_split', type=str, default="anstype_split", help="which type of split")
    parser.add_argument('--SLAKE_version', type=str, default=None, help="version of SLAKE")
    parser.add_argument('--data_augmentation', action="store_true", default=False, help="whether to use data augmentation")

    parser.add_argument('--epochs', type=int, default=200, help="number of epochs")
    parser.add_argument('--lr', default=0.005, type=float, help="initial learning rate")
    parser.add_argument('--weight_decay', default=0, type=float, help="learning rate weight decay")
    parser.add_argument('--layer_wise', action="store_true", default=False, help="whether to use layer-wise learning rate")
    parser.add_argument('--lr_clip', default=0.005, type=float, help="initial learning rate")
    parser.add_argument('--weight_decay_clip', default=0, type=float, help='learning rate weight decay')
    parser.add_argument('--lr_ban', default=0.005, type=float, help="initial learning rate")
    parser.add_argument('--weight_decay_ban', default=0, type=float, help='learning rate weight decay')
    parser.add_argument('--batch_size', type=int, default=196, help="batch size")
    parser.add_argument('--finetuning', type=str, default=None, help="finetuning strategies: BAN, lora-clip, normal-clip")
    parser.add_argument('--lora_rank', type=int, default=4, help="lora rank")
    parser.add_argument('--scheduler', action="store_true", default=False, help="Use learning rate scheduler")
    parser.add_argument('--gamma', type=int, default=4)
    parser.add_argument('--embedding_space', type=str, default="ans2vector.pkl")

    parser.add_argument("--wandbName", type=str, default=None, help="distinguish between different runs")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def generate_dic(Dictionary, input_path, output_path):
    if "SLAKE" in input_path:
        with open(input_path+"train.json", "rb") as f:
            loaded_dataset = json.load(f)
            for item in loaded_dataset:
                question = item["question"].lower()
                answer = str(item["answer"]).lower()
                question_words = question.replace("?", "").split(" ")
                answer_words = answer.split(" ")
                for word in question_words:
                    idx = Dictionary.add_word(word)
                for word in answer_words:
                    idx = Dictionary.add_word(word)
        with open(input_path+"validation.json", "rb") as f:
            loaded_dataset = json.load(f)
            for item in loaded_dataset:
                question = item["question"].lower()
                answer = str(item["answer"]).lower()
                question_words = question.replace("?", "").split(" ")
                answer_words = answer.split(" ")
                for word in question_words:
                    idx = Dictionary.add_word(word)
                for word in answer_words:
                    idx = Dictionary.add_word(word)

    else:
        with open(input_path, "r") as f:
            loaded_dataset = json.load(f)
            dataset_train, dataset_test = train_test_split(loaded_dataset, test_size=0.2, random_state=42)
            train_dataset, val_dataset = train_test_split(dataset_train, test_size=0.2, random_state=42)
            for item in train_dataset:
                question = item["question"].lower()
                answer = str(item["answer"]).lower()
                question_words = question.replace("?", "").replace(",", "").replace('\'s', ' \'s').split(" ")
                answer_words = answer.replace("?", "").replace(",", "").replace('\'s', ' \'s').split(" ")
                for word in question_words:
                    idx = Dictionary.add_word(word)
                for word in answer_words:
                    idx = Dictionary.add_word(word)
            for item in val_dataset:
                question = item["question"].lower()
                answer = str(item["answer"]).lower()
                question_words = question.replace("?", "").split(" ")
                answer_words = answer.split(" ")
                for word in question_words:
                    idx = Dictionary.add_word(word)
                for word in answer_words:
                    idx = Dictionary.add_word(word)
        
        new_output_path = output_path.split("/")[0:3]
        new_output_path = "/".join(new_output_path)
        with open(new_output_path+"/train.json", "w") as f:
            json.dump(train_dataset, f)
        with open(new_output_path+"/validation.json", "w")as f:
            json.dump(val_dataset, f)
        with open(new_output_path+"/test.json", "w") as f:
            json.dump(dataset_test, f)
    
    Dictionary.dump_to_file(path=output_path)


def generate_ans_label(input_path, output_path):
    with open(input_path+"train.json", "rb") as f:
        train_dataset = json.load(f)
    with open(input_path+"validation.json", "rb") as f:
        val_dataset = json.load(f)
    
    ans2label, label2ans, ans2vec = {}, {}, {}
    idx = 0
    for item in train_dataset:
        answer = item["answer"]
        if isinstance(answer, int):
            answer = str(answer)
        answer = answer.lower()
        if answer not in ans2label.keys():
            ans2label[answer] = idx
            label2ans[idx] = answer
            idx += 1

    for item in val_dataset:
        answer = item["answer"]
        if isinstance(answer, int):
            answer = str(answer)
        answer = answer.lower()
        if answer not in ans2label.keys():
            ans2label[answer] = idx
            label2ans[idx] = answer
            idx += 1
    
    pickle.dump(ans2label, open(output_path+"ans2label.pkl", "wb"))
    pickle.dump(label2ans, open(output_path+"label2ans.pkl", "wb"))


def generate_ans_vector(model, ans2label, tokenizer, output_path):
    # text_encoder = model.feature_extractor
    text_encoder = model
    ans2vector = {}
    with torch.no_grad():
        for answer in ans2label.keys():
            tokenized_answer = tokenizer([answer]).to(device)
            ans2vector[answer] = text_encoder.encode_text(tokenized_answer).detach().cpu().numpy()

    pickle.dump(ans2vector, open(output_path, "wb"))
    return ans2vector


def load_ans_vector(loading=None):
    ans2vector = pickle.load(open(loading, "rb"))
    return ans2vector


def generate_imgid_idx(input_path, output_path):
    if "RAD" in input_path:
        with open(input_path, "rb") as f:
            dataset = json.load(f)
        imgid2idx = {}
        idx = 0
        for item in dataset:
            image_name = item["image_name"]
            if image_name not in imgid2idx.keys():
                imgid2idx[image_name] = idx
                idx += 1
    else:
        with open(input_path+"train.json", "rb") as f:
            train_dataset = json.load(f)
        with open(input_path+"validation.json", "rb") as f:
            val_dataset = json.load(f)
        with open(input_path+"test.json", "rb") as f:
            test_dataset = json.load(f)
        imgid2idx = {}
        for item in train_dataset:
            image_name = item["img_name"]
            image_id = item["img_id"]
            if image_name not in imgid2idx.keys():
                imgid2idx[image_name] = image_id
        for item in val_dataset:
            image_name = item["img_name"]
            image_id = item["img_id"]
            if image_name not in imgid2idx.keys():
                imgid2idx[image_name] = image_id
        for item in test_dataset:
            image_name = item["img_name"]
            image_id = item["img_id"]
            if image_name not in imgid2idx.keys():
                imgid2idx[image_name] = image_id
    
    pickle.dump(imgid2idx, open(output_path, "wb"))


def generate_imgid_idx_only(dataset, dataset_name):
    if "RAD" in dataset_name:
        imgid2idx = {}
        idx = 0
        for item in dataset:
            image_name = item["image_name"]
            if image_name not in imgid2idx.keys():
                imgid2idx[image_name] = idx
                idx += 1
    elif "SLAKE" in dataset_name:
        for item in dataset:
            image_name = item["img_name"]
            image_id = item["img_id"]
            if image_name not in imgid2idx.keys():
                imgid2idx[image_name] = image_id

    return imgid2idx


class VqaSampler_train(NamedTuple):
    q_token: torch.Tensor
    image: torch.Tensor
    target: torch.Tensor


class VqaSampler_test(NamedTuple):
    qid: int
    image_name: str
    question: str
    question_type: str
    answer: str
    answer_type:str
    q_token: torch.Tensor
    image: torch.Tensor
    target: torch.Tensor


class VQADatasets(Dataset):
    def __init__(self, args, datasets, preprocess, tokenizer, dataset_name, mode="train"):
        super(VQADatasets, self).__init__()
        self.dataset_name = dataset_name
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.mode = mode
    
        self._transforms = tv.transforms.Compose([
            tv.transforms.ToTensor(),
            tv.transforms.Resize(224, interpolation=tv.transforms.InterpolationMode.BICUBIC),
            tv.transforms.CenterCrop(224),
            tv.transforms.RandomRotation(degrees=(-10.0, 10.0)),
            tv.transforms.RandomHorizontalFlip(0.5),
            tv.transforms.Normalize(mean=mean, std=std)
        ])
        self.augmentation = args.data_augmentation
        
        if "SLAKE" in dataset_name:
            if args.SLAKE_version == "english":
                ans2label_path = "datasets/VQA/SLAKE/SLAKE_ans2label_eng.pkl"
                label2ans_path = "datasets/VQA/SLAKE/SLAKE_label2ans_eng.pkl"
            elif args.SLAKE_version == "chinese":
                ans2label_path = "datasets/VQA/SLAKE/SLAKE_ans2label_ch.pkl"
                label2ans_path = "datasets/VQA/SLAKE/SLAKE_label2ans_ch.pkl"
            else:
                ans2label_path = "datasets/VQA/SLAKE/SLAKE_ans2label.pkl"
                label2ans_path = "datasets/VQA/SLAKE/SLAKE_label2ans.pkl"
            dictionary_path = "datasets/VQA/SLAKE/SLAKE_Dictionary.pkl"
            img_id2idx_path = "datasets/VQA/SLAKE/SLAKE_img2idx.pkl"
        elif "RAD" in dataset_name:
            if args.RAD_split == "image_split":
                ans2label_path = "datasets/VQA/VQA-RAD/image_split/RAD_ans2label.pkl"
                label2ans_path = "datasets/VQA/VQA-RAD/image_split/RAD_label2ans.pkl"
            elif args.RAD_split == "question_split":
                ans2label_path = "datasets/VQA/VQA-RAD/question_split/RAD_ans2label.pkl"
                label2ans_path = "datasets/VQA/VQA-RAD/question_split/RAD_label2ans.pkl"
            elif args.RAD_split == "anstype_split":
                ans2label_path = "datasets/VQA/VQA-RAD/image_anstype_split/RAD_ans2label.pkl"
                label2ans_path = "datasets/VQA/VQA-RAD/image_anstype_split/RAD_label2ans.pkl"
            dictionary_path = "datasets/VQA/VQA-RAD/RAD_Dictionary.pkl"
            # self.img_id2idx = generate_imgid_idx_only(datasets, "RAD")
            img_id2idx_path = "datasets/VQA/VQA-RAD/RAD_img2idx.pkl"

        self.ans2label = pickle.load(open(ans2label_path, "rb"))
        self.label2ans = pickle.load(open(label2ans_path, "rb"))
        self.dictionary = pickle.load(open(dictionary_path, "rb"))
        self.img_id2idx = pickle.load(open(img_id2idx_path, "rb"))
        self.ans2label["unknown"] = len(self.ans2label)
        self.label2ans[len(self.ans2label)] = "unknown"

        self.num_ans_candidates = len(self.ans2label)   # RAD: 458 + 1(unknown)   # SLAKE: 493 +1(unknown)

        # create entries
        self.entries = []
        if "RAD" in dataset_name:
            self.base_root = "datasets/VQA/VQA-RAD/VQA_RAD Image Folder/"
            for item in datasets:
                answer_text = str(item["answer"])
                if answer_text.lower() not in self.ans2label.keys():
                    label = None
                    score = None
                else:
                    label = self.ans2label[answer_text.lower()]
                    score = 1.0
                entry = {
                    "qid": item["qid"],
                    "image_name": item["image_name"],
                    "question": item["question"],
                    "question_type": item["question_type"],
                    "answer": {
                        "answer_text": item["answer"],
                        "labels": label,
                        "scores": score,
                    },
                    "answer_type": item["answer_type"]
                }
                self.entries.append(entry)
        elif "SLAKE" in dataset_name:
            self.base_root = "datasets/VQA/SLAKE/imgs/"
            for item in datasets:
                answer_text = str(item["answer"])
                if answer_text.lower() not in self.ans2label.keys():
                    label = None
                    score = None
                else:
                    label = self.ans2label[answer_text.lower()]
                    score = 1.0
                entry = {
                    "qid": item["qid"],
                    "image_name": item["img_name"],
                    "question": item["question"],
                    "question_type": item["base_type"],
                    "answer": {
                        "answer_text": item["answer"],
                        "labels": label,
                        "scores": score,
                    },
                    "answer_type": item["answer_type"]
                }
                self.entries.append(entry)
        
        # tokenization & tensorize
        for entry in self.entries:
            tokens = self.tokenizer(entry["question"])
            entry["q_token"] = tokens
            question = torch.from_numpy(np.array(entry["q_token"]))
            entry["q_token"] = question

            answer = entry["answer"]
            labels = answer["labels"]
            scores = answer["scores"]
            
            if labels is not None:
                labels = np.array(answer["labels"], dtype=np.int64)
                scores = np.array(answer["scores"], dtype=np.float32)
                labels = torch.from_numpy(labels)
                scores = torch.from_numpy(scores)
                entry["answer"]["labels"] = labels
                entry["answer"]["scores"] = scores
            else:
                entry["answer"]["labels"] = None
                entry["answer"]["scores"] = None
 

    def __len__(self):
        return len(self.entries)
    
    def __getitem__(self, idx):
        entry = self.entries[idx]
        image = Image.open(self.base_root+entry["image_name"])
        if self.augmentation:
            image = self._transforms(image.convert("RGB"))
        else:
            image = self.preprocess(image)
        entry["image"] = image

        answer = entry["answer"]
        if None != answer:
            labels = answer["labels"]
            scores = answer["scores"]
            target = torch.zeros(self.num_ans_candidates)
            if labels is not None:
                # insert values from scores to target
                target.scatter_(0, labels, scores)   # one-hot encoding
                # target[labels] = 1
            else:
                entry["answer"]["labels"] = []
                entry["answer"]["scores"] = []
                # if the key in testset is not included in ans2label, then it is assigned with "unknown"
                target[-1] = 1
            entry["target"] = target
        
        # return a Tuple instead of dictionary
        if self.mode == "train":
            sample = VqaSampler_train(
                q_token = entry["q_token"],
                image = entry["image"],
                target = entry["target"]
                )
        elif self.mode == "test":
            sample = VqaSampler_test(
                qid = entry["qid"],
                image_name = entry["image_name"],
                question = entry["question"],
                question_type = entry["question_type"],
                answer = entry["answer"]["answer_text"],
                answer_type = entry["answer_type"],
                q_token = entry["q_token"],
                image = entry["image"],
                target = entry["target"]
                )

        return sample


class BilinearAttention(nn.Module):
    def __init__(self, feature_extractor, finetuning, gamma=4, num_ans_candidates=100, ans2label=None, ans2vector=None):
        super(BilinearAttention, self).__init__()
        num_hid = 768
        self.feature_extractor = feature_extractor

        self.v_dim = self.feature_extractor.visual.head.proj.out_features
                 
        self.bilinear_attention = attention.BiAttention(self.v_dim, num_hid, num_hid, gamma)
        self.ans2label = ans2label

        self.b_net, self.q_prj, self.c_prj = [], [], []
        objects = 10
        for _ in range(gamma):
            self.b_net.append(BCNet(self.v_dim, num_hid, num_hid, None, k=1))
            self.q_prj.append(FCNet([num_hid, num_hid], "", .2))
            self.c_prj.append(FCNet([objects + 1, num_hid], "ReLU", .0))
        
        self.b_net = nn.ModuleList(self.b_net)
        self.q_prj = nn.ModuleList(self.q_prj)
        self.c_prj = nn.ModuleList(self.c_prj)

        self.classifier = SimpleClassifier(num_hid, num_hid * 2, num_ans_candidates, .5)

        self.glimpse = gamma

    def forward(self, image, question_token):
        # image feature input
        image_features = self.feature_extractor.encode_image(image).unsqueeze(1)   # [batch: 3, 1, output_dim/emb_dim: 512]
        # image_features = self.feature_extractor.visual.trunk(image).unsqueeze(1)    # [3, 1, 768]

        # textual information
        output = self.feature_extractor.text.transformer(question_token)   
        text_features = output.last_hidden_state   # [3, 256, 768]

        # attention mechanism
        att, logits = self.bilinear_attention(image_features, text_features)   # att: [3, 4, 1, 77] logits: [3, 4, 1, 77]

        b_emb = [0] * self.glimpse   # [0, 0, 0, 0]
        for g in range(self.glimpse):
            b_emb[g] = self.b_net[g].forward_with_weights(image_features, text_features, att[:, g, :, :])   # [3, 512]
            atten, _ = logits[:, g, :, :].max(2)

            text_features = self.q_prj[g](b_emb[g].unsqueeze(1)) + text_features   # [3, 77, 512]
        return text_features.sum(1)


def training(args, device, train_dataset, val_dataset):
    # initialize wandb for loggs
    wandb.login(key="c2f51df9b4bf2111388ccee68c2d41d301164d2c")
    wandb.init(
        project="VQA-Finetuning-Biomed-CLIP",
        name=f"{args.dataset}-{args.finetuning}",
        config={
            "task": "VQA-finetuning",
            "architecture": "Biomed-CLIP(ViT-B/16 & BiomedBert)",
            "dataset": args.dataset,
            "finetuning": args.finetuning,
            "learning rate": args.lr,
            "batch size": args.batch_size,
            "epochs": args.epochs,
            "optimizer":"Adamax",
            "lora_rank": args.lora_rank,
        }
        )

    # VQA-RAD: train/val/test: 1438/360/
    # define output dir
    weight_dir = Path("saved_models/biomedclip/vqa")
    weight_dir.mkdir(parents=True, exist_ok=True)
    weight_path_best = f"saved_models/biomedclip/vqa/{args.dataset}_{args.finetuning}_best.pth"
    weight_path_last = f"saved_models/biomedclip/vqa/{args.dataset}_{args.finetuning}_last.pth"


    # load pretrained model
    clip_model, preprocess = create_model_from_pretrained("hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224", jit=False)
    tokenizer = get_tokenizer("hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
    if args.checkpoint_path is not None:
        clip_model.load_state_dict(torch.load(args.checkpoing_path, map_location=device))
    
    clip_model.to(device)
    if "RAD" in args.dataset:
        loading_path = "datasets/VQA/VQA-RAD/ans2vector.pkl"
    else:
        loading_path = "datasets/VQA/SLAKE/ans2vector.pkl"
    ans2vector = load_ans_vector(loading=loading_path)

    # prepare DataLoader
    train_data_prepared = VQADatasets(args, train_dataset, preprocess, tokenizer=tokenizer, dataset_name=args.dataset, mode="train")
    val_data_prepared = VQADatasets(args, val_dataset, preprocess, tokenizer=tokenizer, dataset_name=args.dataset, mode="train")
    train_dataloader = DataLoader(train_data_prepared, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data_prepared, batch_size=args.batch_size, shuffle=False)

    # ans2vector = generate_ans_vector(clip_model, train_data_prepared.ans2label, tokenizer=tokenizer, output_path=f"datasets/VQA/{args.dataset}/ans2vector.pkl")

    # finetuning parameters
    if args.finetuning == "BAN":
        model = BilinearAttention(feature_extractor=clip_model, finetuning=args.finetuning, gamma=args.gamma, num_ans_candidates=train_data_prepared.num_ans_candidates, ans2label=train_data_prepared.ans2label, ans2vector=ans2vector)
        for param in model.feature_extractor.parameters():
            param.requires_grad = False
        sum_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"after finetuning: {sum_params}")
    elif args.finetuning == "lora-clip":
        # replace vision encoder with lora vision encoder
        lora_vit = BiomedCLIPViT_LoRA(clip_model, lora_rank=args.lora_rank)
        clip_model.visual.trunk = lora_vit.lora_vit
        # replace projection layer with lora projection
        clip_model = build_lora_projection(clip_model, lora_rank=args.lora_rank)
        model = BilinearAttention(feature_extractor=clip_model, finetuning=args.finetuning, gamma=args.gamma, num_ans_candidates=train_data_prepared.num_ans_candidates, ans2label=train_data_prepared.ans2label, ans2vector=ans2vector)
        sum_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"after finetuning: {sum_params}")
    elif args.finetuning == "normal-clip":
        model = BilinearAttention(feature_extractor=clip_model, finetuning=args.finetuning, gamma=args.gamma, num_ans_candidates=train_data_prepared.num_ans_candidates, ans2label=train_data_prepared.ans2label, ans2vector=ans2vector)
        for param in model.parameters():
            param.requires_grad = True
        sum_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"after finetuning: {sum_params}")
    else:
        model = BilinearAttention(feature_extractor=clip_model, finetuning=args.finetuning, gamma=args.gamma, num_ans_candidates=train_data_prepared.num_ans_candidates, ans2label=train_data_prepared.ans2label, ans2vector=ans2vector)
    
    model.to(device)
    # for name, param in model.named_parameters():
    #     print(name, param.requires_grad)

    sum_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    wandb.config.update({"trainable_parameters": sum_params})
    
    # group learning rates
    if args.finetuning == "normal-clip":
        if args.layer_wise:
            param_groups = [
                {"params": model.feature_extractor.parameters(), "lr": args.lr_clip, "weight_decay": args.weight_decay_clip},
                {"params": model.bilinear_attention.parameters(), "lr": args.lr_ban, "weight_decay": args.weight_decay_ban},
                {"params": model.b_net.parameters(), "lr": args.lr_ban, "weight_decay": args.weight_decay_ban},
                {"params": model.q_prj.parameters(), "lr": args.lr_ban, "weight_decay": args.weight_decay_ban},
                {"params": model.classifier.parameters(), "lr": args.lr_ban, "weight_decay": args.weight_decay_ban},
                {"params": model.c_prj.parameters(), "lr": args.lr_ban, "weight_decay": args.weight_decay_ban},
            ]
            optimizer = torch.optim.Adamax(param_groups, betas=(0.9, 0.999), eps=1.0e-6)
        else: 
            optimizer = torch.optim.Adamax(filter(lambda p: p.requires_grad, model.parameters()),
                                        lr=args.lr,
                                        betas=(0.9, 0.999),
                                        eps=1.0e-6,
                                        weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adamax(filter(lambda p: p.requires_grad, model.parameters()),
                                        lr=args.lr,
                                        betas=(0.9, 0.999),
                                        eps=1.0e-6,
                                        weight_decay=args.weight_decay)
    # criterion = torch.nn.BCEWithLogitsLoss(reduction="sum")
    criterion = torch.nn.CrossEntropyLoss()
    # criterion = torch.nn.CosineEmbeddingLoss()

    # trainer
    training_loss, validation_loss = [], []
    best_validation_loss = 8464
    evaluation = EvaluationMatrices()
    early_stop_counter = 0

    # start training
    with tqdm(desc=f"Epoch {1:4d}", total=args.epochs) as pbar:
        for epoch in range(1, args.epochs + 1):
            running_train_loss = 0.0
            running_val_loss = 0.0

            running_train_acc = 0.0
            running_val_acc = 0.0

            running_train_f1 = 0.0
            running_val_f1 = 0.0

            model.train()
            for case, batch in enumerate(train_dataloader):
                image = batch.image.to(device)
                question_tokens = batch.q_token.squeeze(1).to(device)
                target = batch.target.to(device)


                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    features = model(image, question_tokens)   # [3, 512]
                    preds = model.classifier(features)   # [Batch, 459]

                loss = criterion(preds, target)
                loss.backward()
                optimizer.step()
                running_train_loss += loss.item()
                
                output = torch.zeros_like(target)
                pred_values, pred_indices = torch.max(preds, dim=-1)
                output.scatter_(1, pred_indices.clone().detach().unsqueeze(1), 1)
                output = output.detach().cpu().numpy()
                target = target.detach().cpu().numpy()
                acc = evaluation.accuracy(output, target)
                f1 = evaluation.F1_score(output, target)
                running_train_acc += acc
                running_train_f1 += f1
            
            training_loss.append(running_train_loss / len(train_dataloader))
            wandb.log({"train_loss": running_train_loss / len(train_dataloader),
                       "train_acc": running_train_acc / len(train_dataloader),
                       "train_f1": running_train_f1 / len(train_dataloader)})

            model.eval()
            for case, batch in enumerate(val_dataloader):
                image = batch.image.to(device)
                question_tokens = batch.q_token.squeeze(1).to(device)
                target = batch.target.to(device)

                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        features = model(image, question_tokens)
                        preds = model.classifier(features)

                    loss = criterion(preds, target)
                    running_val_loss += loss.item()

                    output = torch.zeros_like(target)
                    pred_values, pred_indices = torch.max(preds, dim=-1)
                    output.scatter_(1, pred_indices.clone().detach().unsqueeze(1), 1)
                    output = output.detach().cpu().numpy()
                    target = target.detach().cpu().numpy()
                    acc = evaluation.accuracy(output, target)
                    f1 = evaluation.F1_score(output, target)
                    running_val_acc += acc
                    running_val_f1 += f1

            validation_loss.append(running_val_loss / len(val_dataloader))
            wandb.log({"val_loss": running_val_loss / len(val_dataloader),
                       "val_acc": running_val_acc / len(val_dataloader),
                       "val_f1": running_val_f1 / len(val_dataloader)})


            if validation_loss[-1] < best_validation_loss:
                best_validation_loss = validation_loss[-1]
                torch.save(model.state_dict(), weight_path_best)
                ans2vector = generate_ans_vector(model.feature_extractor, train_data_prepared.ans2label, tokenizer=tokenizer, output_path=f"datasets/VQA/{args.dataset}/ans2vector_{args.dataset}_{args.finetuning}_best.pkl")
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            if early_stop_counter >= 5:
                print(f"Early Stopping at epoch {epoch}")
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
        ans2vector = generate_ans_vector(model.feature_extractor, train_data_prepared.ans2label, tokenizer=tokenizer, output_path=f"datasets/VQA/{args.dataset}/ans2vector_{args.dataset}_{args.finetuning}_last.pkl")
        print("Last model saved.")
        wandb.finish()


def evaluation(args, device, test_dataset, checkpoint_path=None):
    wandb.login(key="c2f51df9b4bf2111388ccee68c2d41d301164d2c")
    wandb.init(
    project="VQA-Finetuning-Biomed-CLIP (Predictions)",
    name=args.wandbName,
    config={
        "task": "VQA-prediction",
        "architecture": "Biomed-CLIP(ViT-B/16 & BiomedBert)",
        "lora_rank": args.lora_rank,
    },
    )
    evaluation = EvaluationMatrices()
    clip_model, preprocess = create_model_from_pretrained("hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
    tokenizer = get_tokenizer("hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
    clip_model.to(device)

    if "RAD" in args.dataset:
        loading_path = f"datasets/VQA/VQA-RAD/{args.embedding_space}"
    elif "SLAKE" in args.dataset:
        loading_path = f"datasets/VQA/SLAKE/{args.embedding_space}"
    ans2vector = load_ans_vector(loading=loading_path)

    test_data_prepared = VQADatasets(args, test_dataset, preprocess, tokenizer=tokenizer, dataset_name=args.dataset, mode="test")
    test_dataloader = DataLoader(test_data_prepared, batch_size=1, shuffle=False)

    if checkpoint_path is not None:
        finetuning = checkpoint_path.split("_")[1]
        dataset = checkpoint_path.split("_")[0]
        checkpoint_path = f"saved_models/biomedclip/vqa/{checkpoint_path}"
        state_dict = torch.load(checkpoint_path, map_location=device)
        if "lora" in checkpoint_path:
            lora_vit = BiomedCLIPViT_LoRA(clip_model, lora_rank=args.lora_rank)
            clip_model.visual.trunk = lora_vit.lora_vit
            clip_model = build_lora_projection(clip_model, lora_rank=args.lora_rank)
            model = BilinearAttention(feature_extractor=clip_model, finetuning=finetuning, gamma=args.gamma, num_ans_candidates=test_data_prepared.num_ans_candidates, ans2label=test_data_prepared.ans2label, ans2vector=ans2vector)
        else:
            model = BilinearAttention(feature_extractor=clip_model, finetuning=finetuning, gamma=args.gamma, num_ans_candidates=test_data_prepared.num_ans_candidates, ans2label=test_data_prepared.ans2label, ans2vector=ans2vector)
        model.load_state_dict(state_dict)
    else:
        finetuning = None
        dataset = None
        model = BilinearAttention(feature_extractor=clip_model, finetuning=finetuning, gamma=args.gamma, num_ans_candidates=test_data_prepared.num_ans_candidates, ans2label=test_data_prepared.ans2label, ans2vector=ans2vector)
    model.to(device)

    model.eval()
    target_results = {
        "image_name": [],
        "question": [],
        "answer": [],
        "predicted_answer": []
    }
    keys = ['count', 'acc', 'f1-score']
    answer_types = ['CLOSED', 'OPEN', 'ALL']
    if "RAD" in args.dataset:
        question_types = ['COUNT', 'COLOR', 'ORGAN', 'PRES', 'PLANE', 'MODALITY', 'POS', 'ABN', 'SIZE', 'OTHER', 'ATTRIB']
    elif "SLAKE" in args.dataset:
        question_types = ["Modality", "Position", "Organ", "Size", "Abnormality", "Quantity", "Plane", "Shape", "Color", "KG"]
    question_types_result = dict((i, dict((j, dict((k, 0.0) for k in keys)) for j in question_types)) for i in answer_types)
    result = dict((i, dict((j, 0.0) for j in keys)) for i in answer_types)
    
    target_closed, output_closed = [], []
    target_open, output_open = [], []
    target_all, output_all = [], []
    with tqdm(desc=f"case {0:5d}", total=len(test_dataloader), unit="case") as pbar:
        for case, batch in enumerate(test_dataloader):
            image = batch.image.to(device)
            question_tokens = batch.q_token.squeeze(1).to(device)
            target = batch.target.to(device)
            answers = batch.answer

            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    features = model(image, question_tokens)
                    preds = model.classifier(features)

                pred_values, pred_indices = torch.max(preds, dim=-1)
                output = torch.zeros_like(target)
                output.scatter_(1, pred_indices.clone().detach().unsqueeze(1), 1)

            # summarize the information
            target_results["image_name"].append(batch.image_name)
            target_results["question"].append(batch.question)
            target_results["answer"].append(answers)
            target_results["predicted_answer"].append(test_data_prepared.label2ans[pred_indices.item()])   # predicted_answer: label

            # compute accuracy of each answer type

            result[batch.answer_type[0].replace(" ", "")]["count"] += 1.0   # calculate the number of samples for each answer type
            result["ALL"]["count"] += 1.0
            if batch.answer_type[0] == "OPEN":
                output_open.extend(output.detach().cpu().numpy())
                target_open.extend(target.detach().cpu().numpy())
            elif batch.answer_type[0] == "CLOSED":
                output_closed.extend(output.detach().cpu().numpy())
                target_closed.extend(target.detach().cpu().numpy())
            output_all.extend(output.detach().cpu().numpy())
            target_all.extend(target.detach().cpu().numpy())
            
            pbar.set_description(f"case {case:5d}")
            pbar.update()
    
    acc_closed = evaluation.accuracy(output_closed, target_closed) * 100
    f1_closed = evaluation.F1_score(output_closed, target_closed) * 100
    result["CLOSED"]["acc"] = acc_closed
    result["CLOSED"]["f1-score"] = f1_closed
    acc_open = evaluation.accuracy(output_open, target_open) * 100
    f1_open = evaluation.F1_score(output_open, target_open) * 100
    result["OPEN"]["acc"] = acc_open
    result["OPEN"]["f1-score"] = f1_open
    acc_all = evaluation.accuracy(output_all, target_all) * 100
    f1_all = evaluation.F1_score(output_all, target_all) * 100
    result["ALL"]["acc"] = acc_all
    result["ALL"]["f1-score"] = f1_all

    print("*******************************************")
    if checkpoint_path is None:
        print(f"Accuracy all type of answers: {acc_all:.2f}%")
        print(f"F1-Score all type of answers: {f1_all:.2f}%")
        print(f"Accuracy CLOSED-ended answers: {acc_closed:.2f}%")
        print(f"F1-Score CLOSED-ended answers: {f1_closed:.2f}%")
        print(f"Accuracy OPEN-ended answers: {acc_open:.2f}%")
        print(f"F1-Score OPEN-ended answers: {f1_open:.2f}%")
    else:
        print(f"DATASET: {dataset}, FINETUNE: {finetuning}, File: {args.wandbName}")
        print(f"Accuracy all type of answers: {acc_all:.2f}%")
        print(f"F1-Score all type of answers: {f1_all:.2f}%")
        print(f"Accuracy CLOSED-ended answers: {acc_closed:.2f}%")
        print(f"F1-Score CLOSED-ended answers: {f1_closed:.2f}%")
        print(f"Accuracy OPEN-ended answers: {acc_open:.2f}%")
        print(f"F1-Score OPEN-ended answers: {f1_open:.2f}%")
        wandb.log({"DATASET": dataset,
                   "Finetuning": finetuning,
                   "LORA": args.lora_rank,
                   "file_path": args.wandbName,
                   "Accuracy all type of answers": f"{acc_all:.2f}%",
                   "F1-Score all type of answers": f"{f1_all:.2f}%",
                   "Accuracy CLOSED-ended answers": f"{acc_closed:.2f}%",
                   "F1-Score CLOSED-ended answers": f"{f1_closed:.2f}%",
                   "Accuracy OPEN-ended answers": f"{acc_open:.2f}%",
                   "F1-Score OPEN-ended answers": f"{f1_open:.2f}%"})

    doc_dir = Path("saved_models/biomedclip/vqa/result_files")
    doc_dir.mkdir(parents=True, exist_ok=True)
    outfile = f"saved_models/biomedclip/vqa/result_files/results_{args.dataset}_{finetuning}.json"
    if not os.path.exists(os.path.dirname(outfile)):
        os.makedirs(os.path.dirname(outfile))
    json.dump(result, open(outfile, "w"))

    for i in question_types_result:
        pd.DataFrame(question_types_result[i]).transpose().to_csv(f"saved_models/biomedclip/vqa/result_files/{args.dataset}_{finetuning}_question_type_{i}.csv")

    df = pd.DataFrame(target_results)
    df.to_csv(f"saved_models/biomedclip/vqa/result_files/{args.dataset}_{finetuning}_targeted_predictions.csv", index=False)

    wandb.finish()


if __name__ == '__main__':
    args = parse_args()
    set_seed(8464)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.mode == "train":
        train_path = os.path.join(args.dataset_path, "train.json")
        val_path = os.path.join(args.dataset_path, "validation.json")
        if args.dataset == "VQA-RAD":
            if args.RAD_split == "image_split":
                train_path = os.path.join(args.dataset_path, "image_split", "train.json")
                val_path = os.path.join(args.dataset_path, "image_split", "validation.json")
            elif args.RAD_split == "question_split":
                train_path = os.path.join(args.dataset_path, "question_split", "train.json")
                val_path = os.path.join(args.dataset_path, "question_split", "validation.json")
            elif args.RAD_split == "anstype_split":
                train_path = os.path.join(args.dataset_path, "image_anstype_split", "train.json")
                val_path = os.path.join(args.dataset_path, "image_anstype_split", "validation.json")
        elif args.dataset == "SLAKE":
            if args.SLAKE_version == "english":
                train_path = os.path.join(args.dataset_path, "train_eng.json")
                val_path = os.path.join(args.dataset_path, "validation_eng.json")
            elif args.SLAKE_version == "chinese":
                train_path = os.path.join(args.dataset_path, "train_ch.json")
                val_path = os.path.join(args.dataset_path, "validation_ch.json")
            else:
                train_path = os.path.join(args.dataset_path, "train.json")
                val_path = os.path.join(args.dataset_path, "validation.json")
        with open(train_path, "rb") as f:
            train_dataset = json.load(f)
        with open(val_path, "rb") as f:
            val_dataset = json.load(f)
        training(args, device, train_dataset, val_dataset)
    if args.mode == "test":
        if args.dataset == "VQA-RAD":
            if args.RAD_split == "image_split":
                test_path = os.path.join(args.dataset_path, "image_split", "test.json")
            elif args.RAD_split == "question_split":
                test_path = os.path.join(args.dataset_path, "question_split", "test.json")
            elif args.RAD_split == "anstype_split":
                test_path = os.path.join(args.dataset_path, "image_anstype_split", "test.json")
        elif args.dataset == "SLAKE":
            if args.SLAKE_version == "english":
                test_path = os.path.join(args.dataset_path, "test_eng.json")
            elif args.SLAKE_version == "chinese":
                test_path = os.path.join(args.dataset_path, "test_ch.json")
            else:
                test_path = os.path.join(args.dataset_path, "test.json")
        with open(test_path, "rb") as f:
            test_dataset = json.load(f)
        evaluation(args, device, test_dataset, checkpoint_path=args.checkpoint_path)