import evaluate


def default_preprocess(eval_pred, ignore_negative_labels=True):
    preds, labels = eval_pred.predictions, eval_pred.label_ids
    if not ignore_negative_labels:
        return preds, labels
    mask = labels > 0
    return preds[mask], labels[mask]


def get_metrics(conf, tokenizer):
    metrics, preprocess_fns = [evaluate.load('accuracy'), default_preprocess]
    return metrics, preprocess_fns
