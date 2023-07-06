import argparse
import os
from tqdm import tqdm

from torch.utils.data import Subset
from transformers import set_seed, TrainingArguments
from transformers.training_args import OptimizerNames

from pdgpt.utils import print_args, str2bool
from pdgpt.tokenization import get_tokenizer, tokenizer_sanity_check
from pdgpt.datasets import DialogueDataCollator, get_dataset


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--deepspeed", type=str2bool, default=False)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--wandb_entity', type=str, default='youkenchaw')
    parser.add_argument("--show_dataset_stats", type=str2bool, default=False)
    args = parser.parse_args()

    conf = {}
    
    return args


def pretrain(args):
    if not args.deepspeed or args.local_rank == 0:
        print_args(args)

    # needs to happen before model loading in case of stage 3 training
    training_conf = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        warmup_steps=args.warmup_steps,
        learning_rate=float(args.learning_rate),
        deepspeed=args.deepspeed_config if args.deepspeed else None,
        optim=OptimizerNames.ADAMW_HF,
        fp16=args.dtype in ["fp16", "float16"],
        bf16=args.dtype in ["bf16", "bfloat16"],
        local_rank=args.local_rank,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_epsilon=float(args.adam_epsilon),
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        eval_accumulation_steps=args.eval_accumulation_steps,
        resume_from_checkpoint=args.resume_from_checkpoint,
        report_to="wandb" if args.log_wandb else None,
    )

    tokenizer = get_tokenizer(args)

    if not args.deepspeed or args.local_rank == 0:
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

    train, evals = get_dataset(training_conf)

    show_dataset_stats = (args.verbose or args.show_dataset_stats) and (
            not args.deepspeed or args.local_rank == 0
    )
    if show_dataset_stats:
        print("Training dataset sizes (before sampling):")
        total = len(train)
        for d in train.datasets:
            if isinstance(d, Subset):
                name = f"Subset of {type(d.dataset).__name__}"
                if hasattr(d.dataset, "name"):
                    name += f" ({d.dataset.name})"
            else:
                name = type(d).__name__
                if hasattr(d, "name"):
                    name += f" ({d.name})"
            print(f"{name}: {len(d)} ({len(d) / total:.2%})")

        print(f"\nTotal train: {total}")
        print("-" * 80)
        print("Evaluation set sizes:")
        total_eval = sum(len(x) for x in evals.values())
        for k, d in evals.items():
            print(f"{k}: {len(d)} ({len(d) / total_eval:.2%})")
        print(f"\nTotal eval: {total_eval}")
        print("-" * 80)

    # if args.use_custom_sampler:
    #     samples_length = None
    #     if args.sort_by_length:
    #         samples_length = list(
    #             map(
    #                 lambda x: train_collate_fn.process_one(x, return_length=True),
    #                 tqdm(train, desc="Calculating lengths per sample"),
    #             )
    #         )
    #
    #     sampler = PerDatasetSampler.build_sampler_from_config(
    #         training_conf,
    #         train.datasets,
    #         rank=training_conf.local_rank,
    #         world_size=training_conf.world_size,
    #         samples_length=samples_length,
    #         verbose=show_dataset_stats,
    #     )
    # else:
    #     sampler = None

    metrics, preprocess_fns = get_metrics(training_conf, tokenizer)

    model = get_model(training_conf, tokenizer)

    if training_conf.peft_model:
        print("Using PEFT model")
        model = peft_model(
            model, peft_type=training_conf.peft_type, gradient_checkpointing=training_conf.gradient_checkpointing
        )

    if not args.log_wandb:
        os.environ["WANDB_MODE"] = "offline"

    if args.log_wandb and (not args.deepspeed or args.local_rank == 0):
        import wandb

        wandb_name = args.model_name.replace(os.getenv("HOME", "/home/ubuntu"), "")
        wandb.init(
            project="supervised-finetuning",
            entity=args.wandb_entity,
            resume=args.resume_from_checkpoint,
            name=f"{wandb_name}-{args.log_dir}-finetuned",
            config=args,
        )
        wandb.config["_max_length"] = args.max_length
        wandb.config["_val_max_length"] = args.val_max_length

    trainer = SFTTrainer(
        model=model,
        args=training_conf,
        sampler=sampler,
        train_collate_fn=train_collate_fn,
        loss_function=args.loss_fn,
        poly_eps=args.poly_eps,
        train_dataset=train,
        eval_dataset=evals,
        data_collator=eval_collate_fn,
        tokenizer=tokenizer,
        compute_metrics=partial(compute_metrics, metrics=metrics, preprocess_fns=preprocess_fns),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
    trainer.train(resume_from_checkpoint=training_conf.resume_from_checkpoint)
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)


def main():
    args = setup_args()
    set_seed(args.random_seed)
    pretrain(args)


if __name__ == '__main__':
    main()
