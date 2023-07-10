import math

import torch
import transformers
from transformers import LlamaForCausalLM

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
