import re

import numpy as np
import pandas as pd


def compute_intermediate_metrics(correct_nodes: dict[str, list[int]], predicted_nodes: dict[str, np.ndarray]):
    precisions = []
    recalls = []
    hits_at_1 = []
    hits_at_5 = []
    recalls_at_20 = []
    reciprocal_ranks = []
    num_guesses = []
    f1s = []

    ids = correct_nodes.keys()
    all_correct_nodes = [correct_nodes[id] for id in ids]
    all_predicted_nodes = [predicted_nodes.get(id, []) for id in ids]

    for predicted_nodes, correct_nodes in zip(all_predicted_nodes, all_correct_nodes):

        num_correct_predictions = len(set(predicted_nodes).intersection(set(correct_nodes)))

        if num_correct_predictions == 0:
            precisions.append(0)
            recalls.append(0)
            hits_at_1.append(0)
            hits_at_5.append(0)
            recalls_at_20.append(0)
            reciprocal_ranks.append(0)
            num_guesses.append(len(predicted_nodes))
            f1s.append(0)
        else:
            prec = num_correct_predictions / len(set(predicted_nodes))
            precisions.append(prec)
            rec = num_correct_predictions / len(set(correct_nodes))
            recalls.append(rec)
            num_correct_predictions_at_20 = len(set(predicted_nodes[:20]).intersection(set(correct_nodes)))
            recalls_at_20.append(num_correct_predictions_at_20 / len(set(correct_nodes)))
            f1s.append(2 / (1 / prec + 1 / rec))
            hits_at_1.append(1 * (predicted_nodes[
                                     0] in correct_nodes))  # only makes sense if ordered (compare hist@first vs hits@last)
            hits_at_5.append(1 * (len(set(predicted_nodes[:5]).intersection(set(correct_nodes))) > 0))
            for i, node in enumerate(predicted_nodes):
                if node in correct_nodes:
                    rr = 1 / (i + 1)
                    break
            else:
                rr = 0
            reciprocal_ranks.append(rr)
            num_guesses.append(len(predicted_nodes))
    print(f"Average scoring for all questions in this split:")
    print(f"F1:              {np.mean(f1s):.4f}")
    print(f"Precision:       {np.mean(precisions):.4f}")
    print(f"Recall:          {np.mean(recalls):.4f}")
    print(f"Exact hit@1:     {np.mean(hits_at_1):.4f}")
    print(f"Exact hit@5:     {np.mean(hits_at_5):.4f}")
    print(f"Exact hit@any:   {np.mean(np.array(precisions) != 0):.4f}")
    print(f"Recall@20:       {np.mean(recalls_at_20):.4f}")
    print(f"MRR:             {np.mean(reciprocal_ranks):.4f}")
    print(f"Num predictions: {np.mean(num_guesses):.2f}")

def compute_metrics(eval_output, skip_invalid_hit=True):
    df = pd.concat([pd.DataFrame(d) for d in eval_output])
    all_hit = []
    all_exact_hit_at_1 = []
    all_exact_hit_at_5 = []
    all_exact_hit_at_any = []
    all_precision = []
    all_recall = []
    all_recall_at_20 = []
    all_rr = []
    all_f1 = []
    all_num_preds = []
    for pred, label in zip(df.pred.tolist(), df.label.tolist()):
        pred = pred.split('[/s]')[0].strip().split('|')
        try:
            hit = re.findall(pred[0], label)
        except Exception as e:
            print(f'Label: {label}')
            print(f'Pred: {pred}')
            print(f'Exception: {e}')
            print('------------------')
            if skip_invalid_hit:
                continue
            else:
                hit = []

        all_hit.append(len(hit) > 0)

        label = label.split('|')
        exact_hit_at_1 = 1 * (pred[0] in label)
        exact_hit_at_5 = 1 * (len(set(pred[:5]).intersection(set(label))) > 0)
        matches = set(pred).intersection(set(label))
        matches_at_20 = len(set(pred[:20]).intersection(set(label)))
        precision = len(matches) / len(set(pred))
        recall = len(matches) / len(set(label))
        recall_at_20 = matches_at_20 / len(set(label))
        if recall + precision == 0:
            f1 = 0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        for i, node in enumerate(pred):
            if node in label:
                rr = 1 / (i + 1)
                break
        else:
            rr = 0

        all_exact_hit_at_1.append(exact_hit_at_1)
        all_exact_hit_at_5.append(exact_hit_at_5)
        all_exact_hit_at_any.append(1 * (precision > 0))
        all_precision.append(precision)
        all_recall.append(recall)
        all_recall_at_20.append(recall_at_20)
        all_rr.append(rr)
        all_f1.append(f1)
        all_num_preds.append(len(pred))

    dataset_len = len(df.label.tolist())
    hit = sum(all_hit) / dataset_len
    exact_hit_at_1 = sum(all_exact_hit_at_1) / dataset_len
    exact_hit_at_5 = sum(all_exact_hit_at_5) / dataset_len
    exact_hit_at_any = sum(all_exact_hit_at_any) / dataset_len
    precision = sum(all_precision) / dataset_len
    recall = sum(all_recall) / dataset_len
    recall_at_20 = sum(all_recall_at_20) / dataset_len
    mrr = sum(all_rr) / dataset_len
    f1 = sum(all_f1) / dataset_len
    num_preds = sum(all_num_preds) / dataset_len

    print(f'F1:              {f1:.4f}')
    print(f'Precision:       {precision:.4f}')
    print(f'Recall:          {recall:.4f}')
    print(f'Substring hit@1: {hit:.4f}')
    print(f'Exact hit@1:     {exact_hit_at_1:.4f}')
    print(f'Exact hit@5:     {exact_hit_at_5:.4f}')
    print(f'Exact hit@any:   {exact_hit_at_any:.4f}')
    print(f'Recall@20:       {recall_at_20:.4f}')
    print(f'MRR:             {mrr:.4f}')
    print(f'Num predictions: {num_preds:.2f}')

