from enum import Enum
import torch
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import numpy as np

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """
    Computes and stores the average and current value
    """
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0
        self.avg = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ''
        if self.summary_type == Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type == Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type == Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        elif self.summary_type == Summary.NONE:
            fmtstr = ''
        else:
            raise ValueError('invalid summary type %r:' % self.summary_type)

        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def display_summary(self):
        entries = [' *']
        entries += [meter.summary() for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

class EvaluationMatrices:
    def __init__(self):
        pass

    def top_accuracy(self, output, target, batch_size, topk=(1,)):
        """
        This function calculates the top predictions for the specified value of k
        :param output: the predictions
        :param target: the ground truth labels
        :param topk: topk to choose
        :return: topk accuracy
        """
        with torch.no_grad():
            maxk = max(topk)  # choose the maximum value of topk

            _, pred = output.topk(maxk, 1, True, True)  # return two tensors, values and indices
            pred = pred.t()  # transpose
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            # eq for output bool values, expand_as for extending to the size of tensor, view for reshaping

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    def F1_score(self, output, target):
        """
        This function calculates the f1 score
        :param output: ground truth, 1d array-like, or label indicator array / sparse matrix
        :param target: prediction, 1d array-like, or label indicator array / sparse matrix
        :return: f1 score
        """

        # f1 = f1_score(target.cpu().detach().numpy(), output.cpu().detach().numpy(), average='macro')
        f1 = f1_score(target, output, average='weighted', zero_division=1)
        return f1
    
    def accuracy(self, output, target):
        """
        This function calculates the accuracy
        :param output: ground truth, 1d array-like or label indicator array / sparse matrix
        :param target: prediction, 1d array-like or label indicator array / sparse matrix
        :return: accuracy
        """
        acc = accuracy_score(output, target)
        return acc
    
    def roc_auc(self, output, target, task="binary-class"):
        """
        This function calculates the roc_auc score
        :param output: ground truth, (n_samples,) or (n_samples, n_classes)
        :param target: predictions, (n_samples,) or (n_samples, n_classes)
        :return: roc_auc score
        """    
        if task == "multi-class":
            roc_auc = roc_auc_score(output, target, multi_class='ovr')
        else:
            roc_auc = roc_auc_score(output, target)
        return roc_auc
    
    def topk_metrics(self, ground_truths, candidates, similarity_scores, k=5):
        """
        This function computes Top@k and Recall@k metrics
        :param ground_truths: list of ground truth texts (each text per image)
        :param candidates: list of lists, where each sublist contains candidate texts for an image
        :param similarity_scores: list of lists, where each sublist contains similarity scores corresponding to candidates
        :param k: Top-k values

        returns:
        top_k: Top@k accuracy
        """
        top_k_hits = 0
        total_gt_count = len(ground_truths)

        for gt, candidate_list, score_list in zip(ground_truths, candidates, similarity_scores):
            # sort candiadtes by similarity score in descending order
            sorted_indices = np.argsort(score_list)[::-1]
            top_k_candidates = [candidate_list[i] for i in sorted_indices[:k]]

            # check if the ground-truth in the top-k
            if gt in top_k_candidates:
                top_k_hits += 1
        
        # compute metrics
        top_k = top_k_hits / total_gt_count

        return top_k * 100

    def recallk_metrics(self, ground_truths, candidates, similarity_scores, k=5):
        """
        This function computes Top@k and Recall@k metrics
        :param ground_truths: list of ground truth texts (each text per image)
        :param candidates: list of lists, where each sublist contains candidate texts for an image
        :param similarity_scores: list of lists, where each sublist contains similarity scores corresponding to candidates
        :param k: Top-k values

        returns:
        top_k: Top@k accuracy
        """
        recall_k_hits = 0
        total_gt_count = len(ground_truths)

        for gt, candidate_list, score_list in zip(ground_truths, candidates, similarity_scores):
            # sort candiadtes by similarity score in descending order
            sorted_indices = np.argsort(score_list)[::-1]
            top_k_candidates = [candidate_list[i] for i in sorted_indices[:k]]

            # check if the ground-truth in the top-k
            if gt in top_k_candidates:
                recall_k_hits += 1
        
        # compute metrics
        recall_k = recall_k_hits / total_gt_count

        return recall_k * 100
        