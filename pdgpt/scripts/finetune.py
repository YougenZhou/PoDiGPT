import argparse
import os

from accelerate import Accelerator
import wandb
from transformers import set_seed, TrainingArguments
from transformers.training_args import OptimizerNames

from pdgpt.utils import print_args, str2bool
from pdgpt.tokenization import get_tokenizer, tokenizer_sanity_check
from pdgpt.datasets import DialogueDataCollator


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--wandb_entity', type=str, default='youkenchaw')

    parser.add_argument('--output_dir', type=str, default='./output/bloomz')
    parser.add_argument('--log_dir', type=str, default='./log/bloomz')
    parser.add_argument('--cache_dir', type=str, default='./packages')
    parser.add_argument('--data_dir', type=str, default='./data/conversations')

    parser.add_argument('--model_name', type=str, default='bigscience/bloomz-7b1-mt')
    parser.add_argument('--n_epochs', type=int, default=5)
    parser.add_argument('--train_bsz_per_gpu', type=int, default=3)
    parser.add_argument('--eval_bsz_per_gpu', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--warmup_rates', type=float, default=0.25)
    parser.add_argument('--eval_step', type=int, default=50)
    parser.add_argument('--save_step', type=int, default=100)
    parser.add_argument('--log_step', type=int, default=5)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--resume_from_checkpoint', type=str2bool, default=False)

    args = parser.parse_args()
    return args


def pretrain(args):
    # accelerator = Accelerator(mixed_precision='fp16')

    # if accelerator.is_local_main_process:
    print_args(args)

    optimizer = OptimizerNames.ADAMW_HF

    train_conf = TrainingArguments(
        output_dir=args.output_dir
    )

    tokenizer = get_tokenizer(args)

    # if accelerator.is_local_main_process:
    tokenizer_sanity_check(tokenizer)

    train_collate_fn = DialogueDataCollator(
        tokenizer,
        max_length=args.max_length,
        random_offset_probability=args.random_offset_probability,
        label_masking=args.label_masking,
        samples_mixing=args.samples_mixing,
        pad_to_multiple_of=16,
        use_system_prefix=args.use_system_prefix,
        system_prefix=args.system_prefix,
        use_system_tag=args.use_system_tag,
        system_property_dropout=args.system_property_dropout,
        system_add_length=args.system_add_length,
    )

    if args.val_max_length is None:
        args.val_max_length = args.max_length

    eval_collate_fn = DialogueDataCollator(
        tokenizer,
        max_length=args.val_max_length,
        random_offset_probability=args.random_offset_probability,
        label_masking=args.label_masking,
        samples_mixing=False,
        use_system_prefix=args.use_system_prefix,
        system_prefix=args.system_prefix,
        use_system_tag=args.use_system_tag,
        system_property_dropout=args.system_property_dropout,
        system_add_length=args.system_add_length,
    )

    train, evals = get_dataset(args)

    # wandb_name = args.model_name.replace(os.getenv("HOME", "/home/ubuntu"), "")
    # wandb.init(
    #     project="pdgpt",
    #     entity=args.wandb_entity,
    #     resume=args.resume_from_checkpoint,
    #     name=f"{wandb_name}-finetuned",
    #     config=args,
    # )

    trainer = SFTTrainer(
        model=model,
        args=args,
        sampler=sampler,
        train_collate_fn=train_collate_fn,
        loss_function=training_conf.loss_fn,
        poly_eps=training_conf.poly_eps,
        train_dataset=train,
        eval_dataset=evals,
        data_collator=eval_collate_fn,
        tokenizer=tokenizer,
        compute_metrics=partial(compute_metrics, metrics=metrics, preprocess_fns=preprocess_fns),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
    trainer.train(resume_from_checkpoint=training_conf.resume_from_checkpoint)
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)


def main():
    args = setup_args()
    set_seed(args.random_seed)
    pretrain(args)


if __name__ == '__main__':
    main()
