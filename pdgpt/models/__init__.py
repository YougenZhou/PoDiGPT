import math

import torch
import transformers
from transformers import LlamaForCausalLM
from peft import LoraConfig, PeftModel, PrefixTuningConfig, get_peft_model, prepare_model_for_int8_training

from pdgpt.utils import patch_model

SUPPORTED_MODELS = ["galactica", "gpt-j"]


def freeze_top_n_layers(model, target_layers):
    # its possible we can simply detect which module is a ModuleList
    # and simply freeze the module without doing string parsing
    for name, param in model.named_parameters():
        if "embed" in name:
            param.requires_grad = False
        elif ".layer" in name or ".h." in name:
            tokens = name.split(".")
            layer_ = None
            for token in tokens:
                if token.isdigit():
                    layer_ = int(token)
                    break
            if layer_ is not None and layer_ < target_layers:
                # print('freeze ', layer_, name)
                param.requires_grad = False
    return model


def get_specific_model(
    model_name, seq2seqmodel=False, without_head=False, cache_dir=".cache", quantization=False, **kwargs
):
    # encoder-decoder support for Flan-T5 like models
    # for now, we can use an argument but in the future,
    # we can automate this
    if without_head:
        model = transformers.AutoModel.from_pretrained(model_name, cache_dir=cache_dir, **kwargs)
    elif seq2seqmodel:
        model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=cache_dir, **kwargs)
    else:
        if "falcon" in model_name:
            kwargs["trust_remote_code"] = True
        model = transformers.AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, **kwargs)
    return model


def get_model(conf, tokenizer, pad_vocab_size_to_multiple_of=16, check_freeze_layer=True):
    dtype = torch.float32
    if conf.dtype in ["fp16", "float16"]:
        dtype = torch.float16
    elif conf.dtype in ["bf16", "bfloat16"]:
        dtype = torch.bfloat16

    if conf.is_reward_model:
        if "pythia" in conf.model_name:
            model = GPTNeoXRewardModel.from_pretrained(conf.model_name, cache_dir=conf.cache_dir, torch_dtype=dtype)

            if conf.pooling:
                assert conf.pooling in ("mean", "last"), f"invalid pooling configuration '{conf.pooling}'"
                model.config.pooling = conf.pooling
        else:
            model = transformers.AutoModelForSequenceClassification.from_pretrained(
                conf.model_name, cache_dir=conf.cache_dir, num_labels=1, torch_dtype=dtype
            )
    if not conf.is_reward_model:
        if conf.peft_type is not None and conf.peft_type == "prefix-tuning" and "llama" in conf.model_name:
            model = LlamaForCausalLM.from_pretrained(conf.model_name, cache_dir=conf.cache_dir, torch_dtype=dtype)
        else:
            model = get_specific_model(
                conf.model_name,
                cache_dir=conf.cache_dir,
                quantization=conf.quantization,
                seq2seqmodel=conf.seq2seqmodel,
                without_head=conf.is_reward_model,
                torch_dtype=dtype,
            )

        n_embs = model.get_input_embeddings().num_embeddings
        if len(tokenizer) != n_embs and check_freeze_layer:
            assert not conf.freeze_layer, "Cannot change the number of embeddings if the model is frozen."

        if len(tokenizer) != n_embs or pad_vocab_size_to_multiple_of:
            p = pad_vocab_size_to_multiple_of
            target_size = len(tokenizer) if not p else math.ceil(len(tokenizer) / p) * p
            print("Resizing embeddings to", target_size)
            model.resize_token_embeddings(target_size)

        if conf.freeze_layer:
            model = freeze_top_n_layers(model, conf.freeze_layer)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([p.numel() for p in model_parameters])
    print("Number of trainable parameters: {}M".format(int(params / 1e6)))

    patch_model(
        model,
        resid_pdrop=conf.residual_dropout,
        flash_attention=conf.use_flash_attention,
        residual_dropout_lima=conf.residual_dropout_lima,
    )

    return model


def prepare_model_for_gradient_checkpointing(model):
    r"""
    Prepares the model for gradient checkpointing if necessary
    """
    if not getattr(model, "is_loaded_in_8bit", False):
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    return model


def peft_model(model, peft_type="lora", int8_training=False, gradient_checkpointing=False):
    if peft_type == "lora":
        config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
    elif peft_type == "prefix-tuning":
        config = PrefixTuningConfig(
            num_virtual_tokens=30, prefix_projection=True, encoder_hidden_size=1024, task_type="CAUSAL_LM"
        )
    else:
        raise ValueError("peft_method config is lora or prefix-tuning")
    model = get_peft_model(model, config)
    if int8_training:
        model = prepare_model_for_int8_training(model)

    if gradient_checkpointing:
        model = prepare_model_for_gradient_checkpointing(model)
    model.print_trainable_parameters()
    return model
