import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import datasets
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments, PreTrainedModel
from transformers.trainer_pt_utils import IterableDatasetShard
from transformers.utils import is_datasets_available
from transformers.trainer_utils import seed_worker


class CrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction="mean"):
        super().__init__(weight, size_average, ignore_index, reduce, "none")
        self._reduction = reduction

    def forward(self, input, target, mask=None):
        input = input.view(-1, input.size(-1))
        target = target.view(-1)

        if mask is not None:
            mask = mask.view(-1).bool()
            input = input[mask]
            target = target[mask]

        size = target.numel()

        loss = super().forward(input, target)

        if self._reduction == "none":
            return loss
        return loss.sum() / (size + 1e-8)


class PolyLoss(nn.Module):
    def __init__(self, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction="mean", epsilon=1.0):
        super().__init__()
        self.weight = torch.tensor(weight)
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.cross_entropy = CrossEntropyLoss(weight, size_average, ignore_index, reduce, "none")
        self.epsilon = epsilon

    def forward(self, input, target, mask=None):
        if mask is not None:
            mask = mask.view(-1).bool()
            input = input.view(-1, input.size(-1))
            target = target.view(-1)
            input = input[mask]
            target = target[mask]

        onehot_target = F.one_hot(target, num_classes=input.size(-1)).to(device=input.device, dtype=input.dtype)
        pt = torch.sum(onehot_target * F.softmax(input, -1), -1)
        CE = self.cross_entropy(input, target)
        poly1 = CE + self.epsilon * (1 - pt)
        if self.reduction == "mean":
            poly1 = poly1.mean()
        elif self.reduction == "sum":
            poly1 = poly1.sum()
        return poly1


class RMLoss(nn.Module):
    def __init__(self, reduction="mean", beta=0.001):
        super().__init__()
        self.reduction = reduction
        self.beta = beta

    def forward(self, logits, cu_lengths=None):
        # if cu_lengths is None, assume that all examples belong to the same conversation
        if cu_lengths is None:
            cu_lengths = [0, logits.size(0)]

        device = logits.device
        losses = []
        for start, end in zip(cu_lengths[:-1], cu_lengths[1:]):
            pairs = torch.combinations(torch.arange(end - start, device=device), 2)
            pos_ids, neg_ids = pairs[:, 0], pairs[:, 1]
            pos_logits = logits.take(start + pos_ids)
            neg_logits = logits.take(start + neg_ids)

            l2 = 0.5 * (pos_logits**2 + neg_logits**2)
            _loss = (-F.logsigmoid(pos_logits - neg_logits) + self.beta * l2).mean()
            losses.append(_loss)
        loss = torch.stack(losses)

        if self.reduction == "none":
            return loss
        return loss.mean()


class RMCLSLoss(nn.CrossEntropyLoss):
    def __init__(self, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction="mean"):
        super().__init__(weight, size_average, ignore_index, reduce, "none")
        self._reduction = reduction

    def forward(self, logits, cu_lengths=None):
        # if cu_lengths is None, assume that all examples belong to the same conversation
        if cu_lengths is None:
            cu_lengths = [0, logits.size(0)]

        device = logits.device
        logit_pairs = []
        # aggregate combination between ranks
        for start, end in zip(cu_lengths[:-1], cu_lengths[1:]):
            pairs = torch.combinations(torch.arange(end - start, device=device), 2)
            pos_ids, neg_ids = pairs[:, 0], pairs[:, 1]
            pos_logits = logits.take(start + pos_ids)
            neg_logits = logits.take(start + neg_ids)
            merged = torch.stack((pos_logits, neg_logits), dim=1)
            logit_pairs.append(merged)
        logit_pairs = torch.concat(logit_pairs, dim=0)
        labels = torch.zeros(logit_pairs.shape[0], dtype=torch.long, device=device)
        loss = super().forward(logit_pairs, labels)

        if self._reduction == "none":
            return loss
        return loss.mean()


def compute_metrics(eval_pred, preprocess_fns, metrics):
    out = {}
    for metric, preprocess_fn in zip(metrics, preprocess_fns):
        preds, labels = preprocess_fn(eval_pred)
        out = dict(**out, **metric.compute(predictions=preds, references=labels))

    return out


def preprocess_logits_for_metrics(logits, labels):
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids


def get_loss(loss, poly_eps: float = 1.0, score_l2_reg: float = 0.001):
    if loss == "CrossEntropyLoss":
        return CrossEntropyLoss()
    elif loss == "Poly":
        return PolyLoss(epsilon=poly_eps)
    elif loss == "RMLoss":
        return RMLoss(beta=score_l2_reg)
    elif loss == "RMCLSLoss":
        return RMCLSLoss()
    else:
        raise ValueError(f"Loss {loss} not supported")


class SFTTrainer(Trainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArguments = None,
        sampler: torch.utils.data.sampler.Sampler = None,
        loss_function: str = "CrossEntropyLoss",
        poly_eps: float = 1.0,
        train_collate_fn: Callable = None,
        **kwargs,
    ):
        super().__init__(model, args, **kwargs)
        self.train_collate_fn = train_collate_fn
        # By default CrossEntropyLoss ignores padding_index -100, but just in case use our own loss_fct
        self.loss_fct = get_loss(loss_function, poly_eps)
        self.sampler = sampler

    def compute_loss(self, model, inputs, return_outputs=False):
        labels_mask = inputs.pop("label_masks")
        targets = inputs.pop("targets")

        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask", None),
            use_cache=False,
        )

        loss = self.loss_fct(outputs.get("logits"), targets, mask=labels_mask)

        return (loss, outputs) if return_outputs else loss

    def _compute_loss(self, model, inputs):
        inputs = self._prepare_inputs(inputs)

        labels_mask = inputs.pop("label_masks")
        targets = inputs.pop("targets")

        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask", None),
            use_cache=False,
        )

        logits = outputs.get("logits")

        loss = self.loss_fct(outputs.get("logits"), targets, mask=labels_mask)

        return loss, logits, targets, labels_mask

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        with torch.no_grad():
            loss, logits, labels, labels_mask = self._compute_loss(model, inputs)
            labels[~labels_mask.bool()] = -100  # padding_index

        loss = loss.mean().detach()

        if self.args.prediction_loss_only:
            return (loss, None, None)

        return (loss, logits, labels)

    def get_train_dataloader(self):
        """
        Inject custom data sampling behaviour into training loop
        and use custom task mixing collate function : train_collate_fn

        rewrite from:
        https://github.com/huggingface/transformers/blob/67d074874d285e616393c65a0e670088e1b6b74a/src/transformers/trainer.py#L846
        """
        data_collator = self.train_collate_fn
        train_dataset = self.train_dataset
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")

        if isinstance(train_dataset, torch.utils.data.IterableDataset):
            # if we are using iterable dataset it means no weight sampling
            # added for backward compat
            if self.args.world_size > 1:
                train_dataset = IterableDatasetShard(
                    train_dataset,
                    batch_size=self._train_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )
            return DataLoader(
                train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                collate_fn=data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

        if self.sampler is None:
            train_sampler = self._get_train_sampler()
        else:
            train_sampler = self.sampler
            logging.warning("Custom sampler found!")

        dataloader = DataLoader(
            train_dataset,
            batch_size=self._train_batch_size,
            sampler=train_sampler,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            worker_init_fn=seed_worker,
        )
        return dataloader

