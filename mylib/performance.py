import math
#y correct class
#cls predicted class
def get_performance_measure(y, cls, perform_measure):
    tp, tn, fp, fn = 0.0, 0.0, 0.0, 0.0
    for label, pred in zip(y,cls):
        assert label == 1 or label == 0
        assert pred == 1 or pred == 0
        if label and pred:
            tp += 1
        elif label:
            fn += 1
        if not label and pred:
            fp += 1
        elif not label:
            tn += 1

    try:
        prec  = tp / (tp + fp)
    except:
        prec = 0
    try:
        rec =   tp / (tp + fn)
    except:
        rec = 0
    try:
        f1 = 2*(prec*rec)/(prec+rec)
    except:
        f1 = 0
    perf = {
    'add' : tp + fp,
    'acc' : (tp + tn) / (tp + tn + fp + fn),
    'accuracy' : (tp + tn) / (tp + tn + fp + fn),
    'prec' : prec,
    'precision' : prec,
    'rec' : rec,
    'recall' : rec,
    'f1': f1,
    'g' : math.sqrt(prec * rec)
    }

    return perf[perform_measure]

def get_contingency_table(y, x, wps, thr):
    tp, tn, fp, fn = 0.0, 0.0, 0.0, 0.0

    con_table = defaultdict(list)
    for label, pred, words in zip(y, x, wps):
        if label and pred > thr:
            tp += 1
            cat = 'tp'
        elif label:
            fn += 1
            cat = 'fn'
        if not label and pred > thr:
            fp += 1
            cat = 'fp'
        elif not label:
            tn += 1
            cat = 'tn'

        con_table[cat].append((words, pred))
    return con_table

if __name__== '__main__':
    main()
    
