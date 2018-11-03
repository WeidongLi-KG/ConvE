from itertools import count
from collections import namedtuple

KBIndex = namedtuple('KBIndex', ['ent_list', 'rel_list', 'rel_reverse_list', 'ent_id', 'rel_id', 'rel_reverse_id'])

def index_ent_rel(*filenames):
    ent_set = set()
    rel_set = set()
    rel_reverse = set()
    for filename in filenames:
        with open(filename) as f:
            for ln in f:
                s, r, t = ln.strip().split('\t')[:3]
                r_reverse = r + '_reverse'
                ent_set.add(s)
                ent_set.add(t)
                rel_set.add(r)
                rel_reverse.add(r_reverse)
    ent_list = sorted(list(ent_set))
    rel_list = sorted(list(rel_set))
    rel_reverse_list = sorted(list(rel_reverse))
    ent_id = dict(zip(ent_list, count()))
    rel_id = dict(zip(rel_list, count()))
    rel_size = len(rel_id)
    rel_reverse_id = dict(zip(rel_reverse_list, count(rel_size)))
    return KBIndex(ent_list, rel_list, rel_reverse_list, ent_id, rel_id, rel_reverse_id)


def graph_size(kb_index):
    return len(kb_index.ent_id), len(kb_index.rel_id)*2


def read_data(filename, kb_index):
    src = []
    rel = []
    dst = []
    with open(filename) as f:
        for ln in f:
            s, r, t = ln.strip().split('\t')
            src.append(kb_index.ent_id[s])
            rel.append(kb_index.rel_id[r])
            dst.append(kb_index.ent_id[t])
    return src, rel, dst

def read_reverse_data(filename, kb_index):
    src = []
    rel = []
    dst = []
    with open(filename) as f:
        for ln in f:
            s, r, t = ln.strip().split('\t')
            r_revsers = r + '_reverse'
            src.append(kb_index.ent_id[t])
            rel.append(kb_index.rel_id[r_revsers])
            dst.append(kb_index.ent_id[s])
    return src, rel, dst

def read_data_with_rel_reverse(filename, kb_index):
    src = []
    rel = []
    dst = []
    with open(filename) as f:
        for ln in f:
            s, r, t = ln.strip().split('\t')
            r_reverse = r + '_reverse'
            src.append(kb_index.ent_id[s])
            rel.append(kb_index.rel_id[r])
            dst.append(kb_index.ent_id[t])
            src.append(kb_index.ent_id[t])
            rel.append(kb_index.rel_reverse_id[r_reverse])
            dst.append(kb_index.ent_id[s])
    return src, rel, dst