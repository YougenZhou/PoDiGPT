from typing import NamedTuple

import transformers

from pdgpt.utils import QA_SPECIAL_TOKENS, create_dataset_entry_qa


class SpecialTokens(NamedTuple):
    pad_token: str = ""
    eos_token: str = ""
    sep_token: str = ""


class TokenizerConfig(NamedTuple):
    special_tokens: SpecialTokens = {}


TOKENIZER_CONFIGS = {
    "galactica": TokenizerConfig(special_tokens=SpecialTokens("<pad>", "</s>")),
    "GPT-JT": TokenizerConfig(special_tokens=SpecialTokens(sep_token="<|extratoken_100|>")),
    "codegen": TokenizerConfig(special_tokens=SpecialTokens("<|endoftext|>", sep_token="<|endoftext|>")),
    "pythia": TokenizerConfig(special_tokens=SpecialTokens("<|padding|>", "<|endoftext|>", "<|endoftext|>")),
    "gpt-neox": TokenizerConfig(special_tokens=SpecialTokens("<|padding|>", "<|endoftext|>", "<|endoftext|>")),
    "llama": TokenizerConfig(special_tokens=SpecialTokens("</s>", "</s>", sep_token="<s>")),
    "cerebras": TokenizerConfig(special_tokens=SpecialTokens("<|endoftext|>", "<|endoftext|>", "<|endoftext|>")),
    "deberta-v3": TokenizerConfig(special_tokens=SpecialTokens("[PAD]", "[SEP]", sep_token="[CLS]")),
    "bloom": TokenizerConfig(special_tokens=SpecialTokens("<pad>", "</s>", "<s>")),
    "electra": TokenizerConfig(special_tokens=SpecialTokens("[PAD]", "[SEP]", sep_token="[CLS]")),
    "falcon": TokenizerConfig(
        special_tokens=SpecialTokens("<|endoftext|>", "<|endoftext|>", sep_token="<|endoftext|>")
    ),
}


def match_tokenizer_name(model_name: str) -> TokenizerConfig:
    """
    Match a partial model name to a tokenizer configuration
    i.e. model_name `Salesforce/codegen-2B-multi` has config name `codegen`
    """
    tokenizer_config_matches = [config for name, config in TOKENIZER_CONFIGS.items() if name in model_name]
    if not tokenizer_config_matches:
        raise ValueError(f"Cannot find any tokeniser configuration to match {model_name=}")
    elif 1 < len(tokenizer_config_matches):
        raise ValueError(f"Found multiple tokeniser configuration matches for {model_name=}")
    else:
        return tokenizer_config_matches[0]


def get_tokenizer(args):
    tokenizer_name = args.model_name

    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=args.cache_dir)

    tokenizer_config = match_tokenizer_name(args.model_name)

    if tokenizer_config.special_tokens:
        if "GPT-JT" in args.model_name:
            tokenizer_config.special_tokens.pad_token = tokenizer.eos_token
        tokenizer.add_special_tokens(
            {
                "pad_token": tokenizer_config.special_tokens.pad_token,
                "eos_token": tokenizer_config.special_tokens.eos_token,
                "sep_token": tokenizer_config.special_tokens.sep_token,
            }
        )

    additional_special_tokens = (
        []
        if "additional_special_tokens" not in tokenizer.special_tokens_map
        else tokenizer.special_tokens_map["additional_special_tokens"]
    )

    additional_special_tokens = list(set(additional_special_tokens + list(QA_SPECIAL_TOKENS.values())))

    tokenizer.add_special_tokens({"additional_special_tokens": additional_special_tokens})

    return tokenizer


def tokenizer_sanity_check(tokenizer):
    print("Tokenizer sanity check:")
    print(f"Type: {type(tokenizer).__name__}")

    print("special_tokens_map:", tokenizer.special_tokens_map)

    print(f"bos_token='{tokenizer.bos_token}', bos_token_id={tokenizer.bos_token_id}")
    print(f"eos_token='{tokenizer.eos_token}', eos_token_id={tokenizer.eos_token_id}")

    ds_entry = create_dataset_entry_qa(
        mode="sft", questions=["Q1", "Q2"], answers=["A1", "A2"], lang="en", context="ctx"
    )
    in_text = ds_entry.get_formatted(
        tokenizer.eos_token,
        use_system_tag=True,
        system_property_dropout=0,
        system_add_length=True,
    )
    in_text = "".join(in_text)

    prompter_token_id = tokenizer.convert_tokens_to_ids(QA_SPECIAL_TOKENS["Question"])
    assistant_token_id = tokenizer.convert_tokens_to_ids(QA_SPECIAL_TOKENS["Answer"])
    print(f"{prompter_token_id=}, {assistant_token_id=}")

    tr = tokenizer(in_text, max_length=1024, pad_to_max_length=False, truncation=True)

    message_indices = []
    i = -1
    for id in tr.input_ids:
        if id in (prompter_token_id, assistant_token_id):
            i += 1
        message_indices.append(i)

    print("encoding result:", tr)
    for i, xs in enumerate(tr.input_ids):
        decoded = tokenizer.decode(xs)
        print(f'{i}: {xs} -> "{decoded}"')

    print("message_indices:", message_indices)
