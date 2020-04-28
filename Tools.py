import re
import numpy as np
import pandas as pd
import time



def takeKBInput():
    kb_input = input()
    if kb_input in ['y', 'n']:
        assert kb_input == 'y'
        return True
    else:
        print("Input must be y/n.")
        takeKBInput()

# get index of first occurrence meeting a condition defined by func in each column of a 2d DataFrame
def getFirst(grp:pd.DataFrame,func):
    ridx=[]
    for (col_idx,col) in enumerate(grp.values.T):
        miss=True
        last=0
        for (row_idx,x) in enumerate(col):
            last+=1
            if func(x):
                ridx.append(row_idx)
                miss=False
                break
        if miss:
            ridx.append(last)
    return ridx


def getDf_roiLabels(rois=None,labels=None,toShow=None):
    if toShow is not None:
        print('WARNING: ROI masking being applied.')
        toMask = set(labels) - set(toShow)
        roi_lbs = pd.Series(index=rois, data=labels).replace(list(toMask), 0)
    else:
        roi_lbs = pd.Series(index=rois, data=labels)
    roi_lbs_r = pd.Series(index=roi_lbs.values, data=roi_lbs.index).sort_index()
    return roi_lbs,roi_lbs_r


def ifAllEqual(lst):
    return lst[1:] == lst[:-1]


from scipy.sparse import csc_matrix
def concatenate_csc_matrices_by_cols(matrix1, matrix2):
    new_data = np.concatenate((matrix1.data, matrix2.data))
    new_indices = np.concatenate((matrix1.indices, matrix2.indices))
    new_ind_ptr = matrix2.indptr + len(matrix1.data)
    new_ind_ptr = new_ind_ptr[1:]
    new_ind_ptr = np.concatenate((matrix1.indptr, new_ind_ptr))

    return csc_matrix((new_data, new_indices, new_ind_ptr))
"""
An example for concatenating multiple matrices
matrices = [matrix1,matrix2,matrix3]

cmat=matrix1
for mat in matrices[1:]:
    cmat = concatenate_csc_matrices_by_cols(cmat,mat)
"""


def square_to_condensed(i, j, n):
    assert i < n
    assert j < n
    assert i != j, "no diagonal elements in condensed matrix"
    if i < j:
        i, j = j, i
    return int(n * j - j * (j + 1) / 2 + i - 1 - j)


def datas_pdist(datas,metric='correlation'):
    # datas_pdidx_to_trlpair = pd.Series()
    datas_pdist_trl = pd.Series()
    datas_pdist_stim = pd.Series()
    for data in datas:
        datas_pdist_trl[data.name], datas_pdist_stim[data.name] = resp_pdist(data.getMiDff(),data,metric=metric)
    """ Check if stim pair orders are the same """
    for i in range(0, datas_pdist_stim.shape[0] - 1):
        assert datas_pdist_stim.iloc[i].columns.to_list() == datas_pdist_stim.iloc[i + 1].columns.to_list()

    return datas_pdist_stim


def resp_pdist(resp,data,metric='correlation'):
    from itertools import combinations
    from scipy.spatial.distance import pdist
    ids = pd.IndexSlice  # for multi-indexing
    resp = resp.swaplevel().sort_index(level='on_idx', sort_remaining=False)

    on_idxs = resp.index.get_level_values('on_idx').unique().astype(int)

    """ CHECK: if each on_idx has the same trl order """
    trl_template = resp.loc[on_idxs[0]].index.to_list()
    for on_idx in on_idxs:
        assert resp.loc[on_idx].index.to_list() == trl_template

    row_to_trl = dict(zip([resp.loc[on_idxs[0]].index.get_loc(trl_idx) for trl_idx in trl_template], trl_template))
    """Map indices from pdist to trl pairs"""
    pdidx_to_trlpair = {}
    for row1, row2 in combinations(row_to_trl.keys(), 2):
        pdidx = square_to_condensed(row1, row2, len(trl_template))
        pdidx_to_trlpair[pdidx] = (row_to_trl[row1], row_to_trl[row2])
    # datas_pdidx_to_trlpair[data.name] = pdidx_to_trlpair
    """Calculate pdist for each time point"""
    dists = []
    for on_idx in on_idxs:
        dists.append(pdist(resp.loc[ids[on_idx, :], :], metric=metric))
    if metric == 'correlation' or metric == 'cosine':
        dists = 1 - np.array(dists)

    trl_pairs = [pdidx_to_trlpair[x] for x in range(0, len(pdidx_to_trlpair))]
    stim_pairs = [tuple(data.trl_to_stim(x)) for x in trl_pairs]
    """
    pdist_trl, pdist_stim
    """
    return pd.DataFrame(data=dists, index=on_idxs, columns=trl_pairs),pd.DataFrame(data=dists, index=on_idxs, columns=stim_pairs)


def get_df_stats(df):
    stats = pd.DataFrame(index=df.columns, columns=['na_count', 'n_unique', 'type', 'memory_usage'])
    for col in df.columns:
        stats.loc[col] = [df[col].isna().sum(), df[col].nunique(dropna=False), df[col].dtypes, df[col].memory_usage(deep=True, index=False) / 1024**2]
    stats.loc['Overall'] = [stats['na_count'].sum(), stats['n_unique'].sum(), None, df.memory_usage(deep=True).sum() / 1024**2]
    return stats


def get_dissimilarity(x, keep_index=True, metric='cosine'):
    from scipy.spatial.distance import pdist, squareform
    dists = squareform(pdist(x.T, metric=metric))
    if keep_index:
        return pd.DataFrame(dists,index=x.columns,columns=x.columns)
    else:
        return dists


""" 
Helper functions for making equal aspects for 3D plots
from https://stackoverflow.com/a/50664367/6622237
"""
def set_axes_radius(ax, origin, radius):
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])

    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    set_axes_radius(ax, origin, radius)


# start_time = time.time()
# print("--- %s seconds ---" % (time.time() - start_time))

# plt.rcParams["axes.grid"] = False

# raise ValueError('To stop running')

# plt.close('all')























