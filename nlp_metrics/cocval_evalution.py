from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.spice.spice import Spice
import subprocess


def eval_nlp_scores(pred, gt, verbose=False):
    """
    evaluates the nlp scores bleu1-bleu4, meteor, rouge-l, cider, spice
    Also logs the corpus values as scalars and the individual scores as histograms!
    Args:
        pred (List): List of predictions
        gt (List): List of ground truths
    """
    gts = {}
    res = {}
    for imgId in range(len(pred)):
        gts[imgId] = gt[imgId]
        res[imgId] = pred[imgId]
    # Set up scorers
    if verbose: print('Setting up scorers...')
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), ["METEOR"]),
        (Rouge(), ["ROUGE_L"]),
        (Cider(), ["CIDEr"])
    ]

    # Compute scores
    results = {}
    for scorer, method in scorers:
        #print('Computing %s score...' % (scorer.method()))

        corpus_score, sentence_scores = scorer.compute_score(gts, res)
        for ind in range(len(method)):
            cs, ss = corpus_score, sentence_scores
            if isinstance(corpus_score, list):
                cs, ss = corpus_score[ind], sentence_scores[ind]
            results[method[ind]] = cs, ss
            if verbose:
                print("%s: %0.3f" % (method[ind], cs))

    return results
