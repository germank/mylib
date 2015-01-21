import operator

from collections import *
from itertools import *

same = lambda x:x  # identity function
add = lambda a,b:a+b
_tuple = lambda x:(x,)  # python actually has coercion, avoid it like so

def dict_flatten(dictionary, keyReducer=add, keyLift=_tuple, init=()):

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

def dict_union(*dicts):
    return dict(chain(*map(lambda dct: list(dct.items()), dicts)))

def dict_map(func, d, level=0):
    '''
    map a function into the values of a dictionary
    '''
    assert level >= 0
    if level > 0:
        return {k: dict_map(func, sd, level-1) for k,sd in d.iteritems()}
    else:
        return {k: func(v) for k,v in d.iteritems()}

def dict_agg(d, new_keys, level=0):
    '''
    Joins keys of a dictionary according to new_keys. The resulting values
    are aggregated into lists
    new_keys: a map from the original key names into new keynames 
    '''
    
    assert level >= 0
    if level > 0:
        return {k: dict_agg(sd, new_keys, level-1) for k,sd in d.iteritems()}
    else:
        nd = {}
        for k,v in d.iteritems():
            nk = new_keys[k]
            if nk not in nd:
                nd[nk] = list()
            nd[nk].append(v)
        return nd

def dict_zip(*ds, **kwargs):
    '''
    Keyword arguments:
    level -- the level at which dictionaries are merged (default: 0)
    '''
    level = kwargs.get('level', 0)
    assert all(set(ds[0].keys())==set(di.keys()) for di in ds[1:])
    assert level >= 0
    if level > 0:
        return {k: dict_zip(map(operator.itemgetter(k), ds),level=level-1) for k in ds[0].keys()}
    else:
        return {k: tuple(map(operator.itemgetter(k), ds)) for k in ds[0].keys()}
    
    
def dict_intersect(*ds, **kwargs):
    level = kwargs.get('level', 0)
    assert level >= 0
    if level > 0:
        return {k: dict_intersect(map(operator.itemgetter(k), ds),level=level-1) for k in ds[0].keys()}
    else:
        common_keys = set.intersection(*map(set, (d.keys() for d in ds)))
        return {k: dict_union(*map(operator.itemgetter(k), ds)) for k in common_keys}
    
def dict_unflatten(dictionary):
    resultDict = dict()
    for key, value in dictionary.iteritems():
        d = resultDict
        for part in key[:-1]:
            if part not in d:
                d[part] = dict()
            d = d[part]
        d[key[-1]] = value
    return resultDict
    
def dict_swap(dictionary, level1, level2):
    flat_dict = dict_flatten(dictionary)
    swapped_dict = {}
    for k,v in flat_dict.iteritems():
        k2 = tuple(k[level2] if i==level1 else k[level1] if i==level2 else k[i] for i in range(len(k)))
        swapped_dict[k2] = v
    return dict_unflatten(swapped_dict)

def dictlist_pk(ds, pk):
    return { tuple(map(d.__getitem__, pk)): {k: v for k,v in d.iteritems() if k not in pk} for d in ds}