from StdImports import *
from scipy.optimize import curve_fit
from scipy import signal
import random
from Data2 import Loader2,get_kind,get_neuro,Datas,rootDir2
loader2 = Loader2()
# datas = loader.get('((gh|mz)_.*_lh)|((gh|oc)_.*_al)',correct_baseline=False)
# datas = loader2.get('gh_(54|56|58|59)_.*',correct_baseline=False, do_rescale = False)
datas = loader2.get('gh_58_.*',correct_baseline=False, do_rescale = False)


crct_path = rootDir2/ 'corrected'
fig_path = crct_path / 'correction plots'

print('Save results? (y/n)')
kb = input()
assert kb == 'y' or kb == 'n'
if kb == 'y':
    save = True
    print('Saving results...')
elif kb == 'n':
    save = False
    print('Don\'t save results.')


def exp_func(x, a, b, c):
    return a * np.exp(b * x) + c

def lin_func(x,a,b):
    return a*x+b



def get_bsl(data):
    u_rawF = data.get_unit_rawF()-100

    """ drop ROI's NA trls"""
    u_rawF2 = u_rawF.dropna(how='all')
    pre = u_rawF2.loc[:, :0].iloc[:, 10:]
    assert not pre.isnull().values.any()

    last_valid_on_idx = u_rawF2.apply(lambda row: row.last_valid_index(), axis=1)

    assert u_rawF2.shape[0] == last_valid_on_idx.shape[0]
    tail_df = pd.DataFrame(index=u_rawF2.index, columns=u_rawF2.columns)

    tail_len = 4
    for row_iloc in range(u_rawF2.shape[0]):
        tail_onrg = slice(last_valid_on_idx.iloc[row_iloc] - tail_len, last_valid_on_idx.iloc[row_iloc])
        tail_df.iloc[row_iloc, :].loc[tail_onrg] = u_rawF2.iloc[row_iloc, :].loc[tail_onrg]

    tail_sr = tail_df.stack()

    rois = pre.index.get_level_values(level='roi').unique()
    std_coef = 1.5

    popts = pd.DataFrame(index=rois, columns=['a', 'b', 'c'])
    bsls = pd.DataFrame().reindex_like(u_rawF)
    bsl_para = pd.DataFrame(index=u_rawF.index, columns=['a', 'b', 'c'])

    for roi in rois:
        # tail_sr.loc[roi]
        # roi_pre = pre.loc[roi]
        roi_pre_tail = pre.loc[roi].stack().append(tail_sr.loc[roi]).sort_index()
        roi_pre_tail = roi_pre_tail.unstack()
        low_thresh = roi_pre_tail.mean(axis=0) - std_coef * roi_pre_tail.std(axis=0)
        up_thresh = roi_pre_tail.mean(axis=0) + std_coef * roi_pre_tail.std(axis=0)
        msk_pre = roi_pre_tail.mask((roi_pre_tail < low_thresh) | (roi_pre_tail > up_thresh))

        ydata = msk_pre.stack()
        xdata = ydata.index.get_level_values(level='on_idx')
        popt, _ = curve_fit(exp_func, xdata, ydata, p0=(0.01, -0.05, ydata.loc[ids[:, 0]].mean()),
                            bounds=([0, -np.inf, 0], [np.inf, 0, ydata.loc[ids[:, 0]].mean() + 100]),
                            max_nfev=xdata.shape[0] * 300)
        popts.loc[roi] = popt

        def fit_c(x, c):
            return exp_func(x, popt[0], popt[1], c)

        for trl_idx,trl_pre in roi_pre_tail.iterrows():
            rm_y = trl_pre.loc[:0].rolling(6).min().dropna()
            rm_x = rm_y.index
            rm_c, _ = curve_fit(fit_c, rm_x, rm_y, p0=trl_pre.loc[0])
            # rm_bsl = pd.Series(fit_c(u_rawF.columns, *rm_c), index=u_rawF.columns)
            bsls.loc[roi,trl_idx] = fit_c(u_rawF.columns, *rm_c)
            bsl_para.loc[roi, trl_idx] = [*popt[:2], rm_c.item()]
    return u_rawF, bsls



def dff_mean_median(u_rawF,bsls):
    dff = (u_rawF - bsls).div(bsls.loc[:, 0], axis=0)
    trim_dff = dff.iloc[:, 10:]
    crct_dff = dff - trim_dff.loc[:, :0].mean().median()
    return crct_dff

def dff_median(u_rawF,bsls):
    # dff0 = ((u_rawF - bsls) / bsls).iloc[:, 10:]
    dff0 = (u_rawF - bsls).div(bsls.loc[:, 0], axis=0)
    trim_dff0 = dff0.iloc[:, 10:]
    crct_dff0 = dff0 - trim_dff0.loc[:, :0].stack().median()
    return crct_dff0

for data in datas:
    u_rawF, bsls = get_bsl(data)
    crct_dff = dff_median(u_rawF, bsls)

    cdff_save = crct_dff.unstack().T.swaplevel().sort_index() # convert to default data.dff format
    cdff_save.dropna(how='all', inplace=True)
    assert cdff_save.shape == data.dff.shape
    # to do check index using data.getUnitDff and data.dff

    crct_dict = {'bsls':bsls.to_dict(), 'crct_dff':cdff_save.to_dict(), 'version':data.version}

    # raise ValueError('To stop running')

    if save:
        import pickle
        filehandler = open(crct_path / (data.name+'.file'), 'wb')
        pickle.dump(crct_dict, filehandler)

    # pickle.load(crct_path / 'test.file')
    # with open(crct_path / 'test.file', 'rb') as pickle_file:
    #     content = pickle.load(pickle_file)
    #
    # pd.DataFrame(content['bsls'])

    """
    1. Heatmaps after correction
    2. pre-stim dff distribution
    """
    plt.ioff()
    # Heatmaps after correction
    crct_dff2 = dff_mean_median(u_rawF,bsls)
    plt.rcParams["axes.grid"] = False
    fig, axes = plt.subplots(2,2,figsize=(40,20))
    axes[0,0].imshow(crct_dff, cmap='RdBu_r',vmax=0.2,vmin=-0.2,aspect='auto')
    axes[0,0].set_title('dff_median')
    axes[0,1].imshow(crct_dff2, cmap='RdBu_r',vmax=0.2,vmin=-0.2,aspect='auto')
    axes[0,1].set_title('dff_mean_median')
    # pre-stim dff distribution
    on_rg = slice(-3,0)
    ax3 = axes[1,0]
    s_crct_dff = crct_dff.loc[:,on_rg].stack()
    s_crct_dff.plot.hist(bins=100,ax=ax3)
    ax3.vlines(0,*ax3.get_ylim(),colors='r')
    ax3.set_title('{},on_rg: {} \n ratio >0: {:.2%}'.format('pre-stim dff_median distribution',
                                                  on_rg, np.sum(s_crct_dff>0)/s_crct_dff.shape[0]))

    ax4 = axes[1,1]
    s_crct_dff2 = crct_dff2.loc[:,on_rg].stack()
    s_crct_dff2.plot.hist(bins=100,ax=ax4)
    ax4.vlines(0,*ax4.get_ylim(),colors='r')
    ax4.set_title('{},on_rg: {} \n ratio >0: {:.2%}'.format('pre-stim dff_mean_median distribution',
                                        on_rg, np.sum(s_crct_dff2>0)/s_crct_dff2.shape[0]))
    plt.suptitle(data.name)


    mng = plt.get_current_fig_manager()
    # mng.window.state('zoomed')
    mng.resize(*mng.window.maxsize())

    if save:
        plt.savefig(fig_path / (data.name+'_fig1.png'),dpi=500,)

    """
    1. Mean dff after correction
    2. Correlation with original rawF
    """
    plt.rcParams["axes.grid"] = True
    fig, axes = plt.subplots(1,2)
    # Mean dff after correction
    ax1 = axes[0]
    crct_dff.mean().plot(ax=ax1)
    ax1.hlines(0,*ax1.get_xlim(),colors='k')
    ax1.set_title('Mean dff after correction')

    # Correlation with original rawF
    on_rg = slice(-5,64)
    corr_new_old = u_rawF.loc[:,on_rg].corrwith(crct_dff.loc[:,on_rg],axis=1)
    ax2 = axes[1]
    corr_new_old.plot(ax=ax2)
    ax2.set_title('{},on_rg: {}'.format('Corr with original rawF',on_rg))

    plt.suptitle(data.name)
    #
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')

    if save:
        plt.savefig(fig_path / (data.name+'_fig2.png'),dpi=500,)


    """
    Lowest correlation trials
    """
    n_small = 3
    small_corr = corr_new_old.nsmallest(n_small) # small correlation

    plt.rcParams["axes.grid"] = False

    fig, axes = plt.subplots(n_small,1)
    for (idx,scorr),ax1 in zip(small_corr.iteritems(),axes):
        u_rawF.loc[idx,crct_dff.columns].plot(ax=ax1,label='rawF')
        bsls.loc[idx,crct_dff.columns].plot(ax=ax1,label='baseline')
        # ax1.legend()
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        crct_dff.loc[idx].plot(ax=ax2,label='crct_dff',c='r')
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1+h2, l1+l2, loc=2)
        plt.title('{},corr: {}'.format(idx,scorr))

    plt.suptitle('{},{}'.format('Lowest correlation trials',data.name))

    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')

    if save:
        plt.savefig(fig_path / (data.name+'_fig3.png'),dpi=500,)