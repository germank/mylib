import operator
from itertools import repeat
from pprint import pprint
import pylab as pl
from scipy import stats
from sklearn import svm
from sklearn import cross_validation
from sklearn import metrics

def to_dataset(positive_cases, negative_cases, 
               positive_rows = None, negative_rows = None):
    
    return list_to_dataset([positive_cases[r] for r in positive_rows],
                           [negative_cases[r] for r in negative_rows])
    
def list_to_dataset(positive_cases, negative_cases):
    ds = []
    for r in positive_cases:
        ds.append((r, 1))
    for r in negative_cases:    
        ds.append((r, 0))
        
    x,y = zip(*ds)
    x=pl.vstack(x)
    y = pl.array(y)
    return (x,y)
    
def to_balanced_dataset(positive_cases, negative_cases, rows=None): 
    
    if not rows:
        rows = set(positive_cases.keys()) & set(negative_cases.keys())
    return to_dataset(positive_cases, negative_cases, rows, rows)

def to_keylist(*key_vector_dicts):
    return list(set.intersection(*map(lambda d: set(d.keys()), key_vector_dicts)))

def concatenate_vectors(*vectors):
    res = {}
    ks = set.intersection(*[set(vd.keys()) for vd in vectors])
    for k in ks:
        res[k] = pl.concatenate([vd[k] for vd in vectors], axis=1)
    return res


def test_classification(clf, ds):
    test_x,test_y = ds
    preds = pl.array([clf.predict(x_i) for x_i in test_x])
    N_errors = sum(abs(preds-pl.array([test_y]).T)) 
    N_cases = len(test_y)
    print 'Errors:', N_errors
    print 'Cases:',  N_cases
    print 'Error rate: {0:.2f}%'.format(N_errors/float(N_cases)*100)
    print  'p-value: {0:.2g}'.format(stats.binom(N_cases, 0.5).cdf(N_errors))

def report_accuracy(scores, p):
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2))
    print 'p-value: {0:.2g}'.format(p)
    return (scores, p)

def scores_pvalues(scores, N_samples, K_FOLDS):
    fold_size = N_samples / float(K_FOLDS)
    test_size = (K_FOLDS-1)*fold_size
    error_size = (1-scores.mean()) * test_size
    p = stats.binom(round(test_size), 0.5).cdf(round(error_size))
    return (scores, p)

def is_significative(p, epsilon=1.0e-3):
    return p < epsilon
     
    
def train_test_classification(clf, ds, K_FOLDS):
    test_x,test_y = ds
    scores = cross_validation.cross_val_score(clf, test_x, test_y, cv=K_FOLDS)
    return scores_pvalues(scores, len(test_y), K_FOLDS)

def cross_train_test_classification(clf, tr_ds, test_ds, K_FOLDS):
    tr_x, tr_y = tr_ds
    test_x, test_y = test_ds
    assert len(tr_y) == len(test_y)
    skf = cross_validation.StratifiedKFold(tr_y, K_FOLDS)
    scores = pl.array([])
    for train_indexes, test_indexes in skf:
        clf.fit(tr_x[train_indexes], tr_y[train_indexes])
        scores =pl.append(scores, metrics.zero_one_score(test_y[test_indexes], clf.predict(test_x[test_indexes])))
    return report_accuracy(*scores_pvalues(scores, len(tr_y), K_FOLDS))


import sys

def print30(*args, **kargs):
    sep  = kargs.get('sep', ' ')             # Keyword arg defaults
    end  = kargs.get('end', '\n')
    file = kargs.get('file', sys.stdout)
    output = ''
    first  = True
    for arg in args:
        output += ('' if first else sep) + str(arg)
        first = False
    file.write(output + end)
    
class ProgressBar:
    def __init__(self, iterations, title=None):
        self.title = title
        self.status_desc = None
        self.iterations = iterations
        self.prog_bar = '[]'
        self.fill_char = '*'
        self.width = 50
        self.__update_amount(0)

    def animate(self, iter, status_desc = None):
        self.status_desc = status_desc
        print30('\r', self, end='')
        sys.stdout.flush()
        self.update_iteration(iter + 1)

    def update_iteration(self, elapsed_iter):
        self.__update_amount((elapsed_iter / float(self.iterations)) * 100.0)
        self.prog_bar += '  %d of %s complete' % (elapsed_iter, self.iterations)

    def __update_amount(self, new_amount):
        percent_done = int(round((new_amount / 100.0) * 100.0))
        all_full = self.width - 2
        num_hashes = int(round((percent_done / 100.0) * all_full))
        self.prog_bar = '[' + self.fill_char * num_hashes + ' ' * (all_full - num_hashes) + ']'
        pct_place = (len(self.prog_bar) // 2) - len(str(percent_done))
        pct_string = '%d%%' % percent_done
        self.prog_bar = self.prog_bar[0:pct_place] + \
            (pct_string + self.prog_bar[pct_place + len(pct_string):])

    def __str__(self):
        return (self.title + ': ' if self.title else '') + str(self.prog_bar) + \
            (' ' + self.status_desc if self.status_desc else '')

import uuid
import time
from IPython.display import HTML, Javascript, display

class HTMLProgressBar:
    def __init__(self, iterations, title=None):
        self.iterations = iterations
        self.divid = str(uuid.uuid4())
        html = ""
        if title:
            html += """<p style="font-weight:bold"><strong>{0}</strong></p>""".format(title)
        html += """
        <div style="border: 1px solid black; width:500px; float:left;">
          <div id="{0}" style="background-color:blue; width:0%%">&nbsp;</div>
        </div> <span id="{0}_t" style="float:left; margin-left:10px;">0%</span>
        """.format(self.divid)
        self.pb = HTML(html)
        display(self.pb)
    
    def animate(self, i, msg=None):

        if msg:
            html = '{0:.2f}% ({1})'.format(i*100.0/self.iterations, msg)
        else:
            html = '{0:.2f}%'.format(i*100.0/self.iterations)

        display(Javascript("$('div#{0}').width('{1}%'); $('span#{0}_t').html(\"{2}\"); ".format(self.divid,i*100.0/self.iterations, html)))


from pylab import *

def max_contribution(m, axis, K):
    import operator
    from itertools import islice
    ms=m**2
    row_sd = sum(ms, axis=axis)
    top_cols = sorted(enumerate(row_sd), key=operator.itemgetter(1), reverse=True)
    top_cols_k = islice(top_cols, K)
    return map(operator.itemgetter(0), top_cols_k)

def smooth_matrix(m, rank):
    return m
    u, s, v = linalg.svd(m, full_matrices=False)
    s[rank:]=0
    return dot(u, dot(diag(s), v))
        
def plot_lf(lex_func, modifier, core=None, descfunc=lambda s, j: s._id2coldescr[j], rank=1, K=5):
    m = array(reshape(lex_func.function_space.get_row(modifier)._mat, 
        lex_func.function_space.get_element_shape()))
    (M,N) = m.shape
    sm = smooth_matrix(m, rank)
    fig = figure(modifier, [10, 16])
    imshow(sm, interpolation='nearest', vmin=-0.3, vmax=0.3, cmap=get_cmap('RdGy'))
    #cbaxes = fig.add_axes([0.8, 0.1, 0.03, 0.8]) 
    _=colorbar(orientation='horizontal')
    if core:
        #mean = sum(m, axis=0)/lex_func.function_space.get_element_shape()[0]
        #var = zeros(lex_func.function_space.get_element_shape()[0])
        #for ro
        
        top_cols_k_idx = max_contribution(m, 0, K)
        #xticks(top_cols_k_idx, [core._id2coldescr[i] for i in top_cols_k_idx], rotation=90)
        for i, x_i in enumerate(reversed(top_cols_k_idx)):
            annotate(descfunc(core,x_i), xy=(float(x_i)/M,float(i)/K+0.01), xytext=(1.05,float(i)/K), 
                     xycoords='axes fraction',
                     textcoords='axes fraction', 
                     arrowprops=dict(arrowstyle="wedge", connectionstyle="arc3,rad=0", fc=(1,1,1)),
                     bbox=dict(facecolor='white', edgecolor='None', alpha=1 ),
                     horizontalalignment='left',
                     fontsize=14)
        top_rows_k_idx = max_contribution(m, 1, K)
        #xticks(top_cols_k_idx, [core._id2coldescr[i] for i in top_cols_k_idx], rotation=90)
        yticks(top_rows_k_idx, [str(i+1) + ") " + descfunc(core,x_i) for i,x_i in enumerate(top_rows_k_idx)], fontsize=14)
        #yticks(top_rows_k_idx, [core._id2coldescr[i] for i in top_rows_k_idx], rotation=0)
        
        
        
    title(modifier, fontsize=16)
    

from collections import *
from itertools import *

same = lambda x:x  # identity function
add = lambda a,b:a+b
_tuple = lambda x:(x,)  # python actually has coercion, avoid it like so

def flattenDict(dictionary, keyReducer=add, keyLift=_tuple, init=()):

    # semi-lazy: goes through all dicts but lazy over all keys
    # reduction is done in a fold-left manner, i.e. final key will be
    #     r((...r((r((r((init,k1)),k2)),k3))...kn))

    def _flattenIter(pairs, _keyAccum=init):
        atoms = ((k,v) for k,v in pairs if not isinstance(v, Mapping))
        submaps = ((k,v) for k,v in pairs if isinstance(v, Mapping))
        def compress(k):
            return keyReducer(_keyAccum, keyLift(k))
        return chain(
            (
                (compress(k),v) for k,v in atoms
            ),
            *[
                _flattenIter(submap.items(), compress(k))
                for k,submap in submaps
            ]
        )
    return dict(_flattenIter(dictionary.items()))