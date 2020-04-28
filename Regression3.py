from StdImports import *

raise ValueError('To stop running')

dirPath = Path(r'F:\analysis\regression\yoyo')
# datas = loader.get('gh_(54|61)_.*')
datas = loader.get('gh_.*')

datas_info = datas.index.str.extract(pat=r'(?P<line_exp>gh_\d{2})_(?P<region>\w{2})')
data_gps = datas.groupby(datas_info.line_exp.values) # TODO WRITE DONW


onrg_trains = [slice(1, 16), slice(24, 40), slice(0,48), slice(-38,0)]

start_time = time.time()
for onrg_train in onrg_trains:
    for gp_name in data_gps.groups.keys():
        data_gp = data_gps.get_group(gp_name)



        """  Construct input X and target Y  """
        train_data_idx = data_gp.index.str.contains('al')
        target_data_idx = data_gp.index.str.contains('cl|lh')

        train_data = data_gp[train_data_idx]
        target_data = data_gp[target_data_idx]

        X = data_gp[train_data_idx][0].filter('.*(02|04)').getMiDff(use_stim_level=True).loc[ids[:,onrg_train], :]
        y_sr = data_gp[target_data_idx].apply(lambda x:x.filter('.*(02|04)').getMiDff(use_stim_level=True).loc[ids[:,onrg_train], :])
        if np.sum(target_data_idx) == 2:
            Y = pd.concat(y_sr.tolist(),axis=1,keys=y_sr.index)
        elif np.sum(target_data_idx) == 1:
            Y = y_sr[0]
        else:
            raise ValueError('Wrong target number')

        Y.columns.rename('target', level=0, inplace=True)
        assert all(X.index == Y.index)  # check index consistency


        """  Construct CV iterator  """
        from sklearn import model_selection
        odor_set = set(X.index.unique(level='stim1').str[:-2])
        odor_group_dict = dict(zip(odor_set,range(len(odor_set))))
        stim_group = X.index.get_level_values(level='stim1').str[:-2]
        stim_group = stim_group.map(lambda x: odor_group_dict[x]).tolist()
        group_kfold = model_selection.GroupKFold(n_splits=len(odor_set))
        # stim_kfold = group_kfold.split(X, Y, groups=stim_group)

        """  Build regression model with CV  """
        reg_param = dict(
            max_iter=10000,
            selection='random'
        )

        from sklearn import linear_model
        mtlcv = linear_model.MultiTaskLassoCV(copy_X=True,
                                              alphas=10**np.linspace(-4,0,num=20),
                                              cv=group_kfold.split(X, Y, groups=stim_group), n_jobs=12,#group_kfold.n_splits,
                                              verbose=1,
                                              **reg_param)

        """ Fit model """
        # mtlcv.fit(X, syn_Y)
        mtlcv.fit(X, Y)
        from sklearn.linear_model import lasso_path
        path_result = mtlcv.path(X, Y, coef_init=np.random.rand(Y.shape[1],X.shape[1]))
        #https://stackoverflow.com/questions/50410037/multiple-linear-regression-with-specific-constraint-on-each-coefficients-on-pyth

        #
        # """  Random simulation  """
        # syn_coef0 = np.random.rand(Y.shape[1], X.shape[1])  # random coefs for simulation
        # row_sum = np.sum(syn_coef0, axis=1)
        # syn_coef = pd.DataFrame(syn_coef0 / row_sum[:, None], index=Y.columns, columns=X.columns)
        # syn_coef = pd.DataFrame(syn_coef0, index=Y.columns, columns=X.columns)
        # syn_Y = X.dot(syn_coef.T)
        # assert syn_Y.shape == Y.shape
        #
        # fig, axes = plt.subplots(1,2, sharey=True)
        # sns.heatmap(X.dot(syn_coef.T), ax=axes[0])
        # sns.heatmap(Y, ax=axes[1])


        assert mtlcv.n_iter_ < reg_param['max_iter']

        """ Retrieve best model """
        best_alpha = mtlcv.alpha_
        coef_pd = pd.DataFrame(mtlcv.coef_, index=Y.columns, columns=X.columns)



        """
        #####################################################################################################################
        save results
        """

        print("Saving results...")
        model_name = type(mtlcv).__name__


        from datetime import datetime
        time_str = datetime.now().strftime("%y%m%d_%H")  # format: YearMonthDay_HourMinute_
        base_name = '_'.join([gp_name,str(onrg_train.start),str(onrg_train.stop),model_name,time_str])

        """ Save model information"""
        ds_status = {}
        ds_status['dnames'] = datas.index.tolist()
        ds_status['gp_name'] = gp_name
        ds_status['on_rg'] = str([onrg_train.start, onrg_train.stop])
        ds_status['model'] = model_name
        ds_status['datas'] = datas.apply(lambda d:d.status).to_dict()
        ds_status['loader'] = loader.status
        ds_status['train_data_idx'] = train_data_idx.tolist()
        ds_status['target_data_idx'] = target_data_idx.tolist()

        if type(ds_status['loader']['regex']) is re.Pattern:
            ds_status['loader']['regex'] = ds_status['loader']['regex'].pattern

        import json
        with open(dirPath / (base_name+'.json'), 'w') as fp:
            json.dump(ds_status, fp, indent=4)


        """ Save coefs """
        coef_pd.to_csv(dirPath / (base_name+'_coef.csv'))

        """ Save model"""
        def convert_generator_for_pickle(obj):
            import types
            for attr_name in dir(obj):
                attr = getattr(obj, attr_name)
                if isinstance(attr, types.GeneratorType):
                    print('=================')
                    print(f'Converted generator: {attr_name} to list')
                    setattr(obj, attr_name,[a for a in attr])



        import joblib
        convert_generator_for_pickle(mtlcv)
        joblib.dump(mtlcv, dirPath / (base_name+'.joblib'))


print("--- %s seconds ---" % (time.time() - start_time))

print('Finished')



raise ValueError('To stop running')




"""
Save meta info to excel
"""
print('Saving meta...')
rc_path = dir_path / 'reg_records.xlsx'
records = pd.read_excel(rc_path,keep_default_na=False,dtype=str)

rc_backup_path = dir_path / 'reg_records_backup.xlsx'
records_backup = pd.read_excel(rc_backup_path,keep_default_na=False,dtype=str)


updated_records = records.append({'Name': base_name,
                                  'data_regex': data_regex,
                                  'dfilter_regex': dfilter_regex,
                                  'on_rg': str(on_rg),
                                  'reduce_by': reduce_by,
                                  'metric': sim_metric,
                                  'time_groups':onrg_gps},
                                 ignore_index=True)

""" Make sure the previous records are unchanged"""
assert updated_records.iloc[:records_backup.shape[0]].equals(records_backup)
if updated_records.iloc[:records_backup.shape[0]].equals(records_backup):
    updated_records.to_excel(rc_path, index=False)
    updated_records.to_excel(rc_backup_path, index=False)




"""
######################################################
Read saved results
New version, works with multiple datasets at a time
######################################################
"""

from os import listdir
import fnmatch


dirPath = Path(r'F:\analysis\regression\200226')
fns = []
for file in listdir(dirPath):
    if fnmatch.fnmatch(file, '*.joblib'):
        fns.append(file.split('.')[0])


def parse_fn(fn):
    name_parts = np.array(fn.split('_'))
    exp_name = ('_').join(name_parts[:2])
    train_onrg = ('_').join(name_parts[2:4])

    return dict(exp=exp_name, onrg_train=train_onrg)


datas = None  # a non elegant fix

def load_reg_result(fn):
    global datas
    import joblib
    import json
    name_parts = parse_fn(fn)
    """ Load model """
    mtlcv = joblib.load(dirPath / (fn + '.joblib'))

    """ Load data status"""
    with open(dirPath / (fn + '.json')) as f:
        ds_status = json.load(f)

    """ Load coef df """
    coef_pd = pd.read_csv(dirPath / (fn + '_coef.csv'), index_col=[0, 1, ], header=[0]).rename_axis(
        index={'roi': 'target_roi'},columns='input_roi')
    # reformat
    coef_pd['exp'] = name_parts['exp']
    coef_pd['on_rg'] = name_parts['onrg_train']

    coef_pd.set_index(['exp', 'on_rg'], append=True, inplace=True)
    coef_pd = coef_pd.reorder_levels(['exp', 'on_rg', 'target', 'target_roi'])


    if datas is None:
        """ Load datas from loader.status"""
        datas = loader.get(**ds_status['loader'])
        assert set(datas.index) == set(ds_status['dnames'])

        print("==================================================")
        print("Model: {}".format(ds_status['model']))
        print()
        print('Regression on {}'.format(ds_status['gp_name']))
        print('on_rg: ' + ds_status['on_rg'])

    # datas_info = datas.index.str.extract(pat=r'(?P<line_exp>gh_\d{2})_(?P<region>\w{2})')
    # data_gps = datas.groupby(datas_info.line_exp.values)  # TODO WRITE DONW
    #
    # fn_parts = parse_fn(fn)
    # data_gp = data_gps.get_group(gp_name)
    #
    # train_data_idx = ds_status['train_data_idx']
    # target_data_idx = ds_status['target_data_idx']
    res = dict(model=mtlcv, coef_pd=coef_pd, )
    res.update(name_parts)
    return res


coef_list = []
for fn in fns:
    res = load_reg_result(fn)
    coef_list.append(res['coef_pd'].stack())

mega_coef = pd.concat(coef_list, axis=0).rename('coef')

datas_info = datas.index.str.extract(pat=r'(?P<line_exp>gh_\d{2})_(?P<region>\w{2})')
data_gps = datas.groupby(datas_info.line_exp.values) # TODO WRITE DONW



def set_pane_brightness(ax, pane_brightness):
    ax.w_xaxis.set_pane_color((*(pane_brightness,) * 3, 1))
    ax.w_yaxis.set_pane_color((*(pane_brightness,) * 3, 1))
    ax.w_zaxis.set_pane_color((*(pane_brightness,) * 3, 1))


def set_gridline_color(ax, color):
    ax.w_xaxis._axinfo['grid']['color'] = color
    ax.w_yaxis._axinfo['grid']['color'] = color
    ax.w_zaxis._axinfo['grid']['color'] = color

    ax.w_xaxis._axinfo['tick']['color'] = (0, 0, 0, 0)
    ax.w_yaxis._axinfo['tick']['color'] = (0, 0, 0, 0)
    ax.w_zaxis._axinfo['tick']['color'] = (0, 0, 0, 0)


""" Rerun till this point"""



"""
Plot coef matrices
"""
plt.ioff()
plt.ion()
def strip_unused_input_col(coef_pd):
    non_na_cols = ~(coef_pd == 0).all(axis=0)
    nonna_coef_pd = coef_pd.loc[:, non_na_cols]

    # nonna_coef_lh = nonna_coef_pd.filter(regex='.*_lh', axis=0)
    # nonna_coef_cl = nonna_coef_pd.drop(nonna_coef_lh.index)
    return nonna_coef_pd

cnorm = colors.DivergingNorm(vmin=-0.6, vcenter=0, vmax=1)

for onrg_plot in mega_coef.index.unique(level='on_rg'):
    # onrg_plot = '0_48'
    onrg_coef = mega_coef.loc[ids[:,onrg_plot]]

    for exp_plot in onrg_coef.index.unique(level='exp'):
        # exp_plot = 'gh_54'
        coef_plot = onrg_coef.loc[exp_plot].unstack()

        nonna_coef_pd = strip_unused_input_col(coef_plot)
        clustergrid = sns.clustermap(nonna_coef_pd,metric='cosine',
                                     cmap='RdBu_r', norm=cnorm, vmax=1,vmin=-0.6,
                                     xticklabels=True, yticklabels=False, rasterized=True)

        plt.suptitle(exp_plot+','+onrg_plot)

        plt.rcParams['svg.fonttype'] = 'none'
        plt.savefig(Path(r'my_path\Imaging Paper\0308') / (f'coef_{exp_plot}_{onrg_plot}.svg'),transparent=True,
                    bbox_inches='tight', dpi=1200, pad_inches=0
                    )

        # break
    # break



"""
Plot order realigned coef matrices
"""

crange = (-0.5, 0.5)
cnorm = colors.DivergingNorm(vmin=min(crange), vcenter=0, vmax=max(crange))

uonrgs = mega_coef.index.unique(level='on_rg').tolist()

onrg_order = ('-38_0', '1_16', '24_40', '0_48')
onrg_order_dict = {onrg: i for i, onrg in enumerate(onrg_order)}
assert set(onrg_order) == set(uonrgs)
ref_onrg = '1_16'  # align others to this
other_onrg = mega_coef.index.unique(level='on_rg').tolist()
other_onrg.remove(ref_onrg)


for exp_plot in mega_coef.index.unique(level='exp'):
    ref_coef = mega_coef.loc[exp_plot,ref_onrg].unstack()
    nonna_coef_pd = strip_unused_input_col(ref_coef)
    na_coef_cols = ref_coef.columns.difference(nonna_coef_pd.columns)

    clustergrid = sns.clustermap(nonna_coef_pd,metric='cosine',
                                         cmap='RdBu_r', norm=cnorm, vmax=max(crange),vmin=min(crange),
                                         xticklabels=True, yticklabels=False, rasterized=True)

    ref_col_order = clustergrid.dendrogram_col.reordered_ind
    row_order = ref_coef.index[clustergrid.dendrogram_row.reordered_ind]
    col_order = nonna_coef_pd.columns[ref_col_order].append(na_coef_cols)

    fig, axes = plt.subplots(1,len(uonrgs), figsize=(5.5*len(uonrgs),5))
    for onrg_plot in uonrgs:
        coef_plot = mega_coef.loc[exp_plot,onrg_plot].unstack()
        i = onrg_order_dict[onrg_plot]
        ax = axes[i]
        im = ax.imshow(coef_plot.loc[row_order, col_order],
                  aspect='auto',
                  cmap='RdBu_r', norm=cnorm, vmax=max(crange), vmin=min(crange),
                  rasterized=True, )
        ax.set_title(onrg_plot)
        ax.set_xlabel('Input ROIs')
        if i == 0:
            ax.set_ylabel('Target ROIs')

    plt.suptitle(exp_plot)

    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    plt.rcParams['svg.fonttype'] = 'none'
    plt.savefig(Path(r'my_path\Imaging Paper\0308') / (f'coef_{exp_plot}_temporal.svg'),
                transparent=True,
                bbox_inches='tight', dpi=1200, pad_inches=0
                )






"""  Plot coef spatial projections for multiple trial periods  """
mega_coef = mega_coef.loc[ids[:,['-38_0','1_16','24_40']]]


gp_name = 'gh_54'
data_gp = data_gps.get_group(gp_name)
cents = data_gp.apply(lambda x: x.getCentroids())


input_cents = cents.filter(regex='.*_al').item()
target1_cents = cents.filter(regex='.*_cl').item()
target2_cents = cents.filter(regex='.*_lh').item()

select_roi = 'slc15_roi011'
input_colors = pd.Series(0,index=input_cents.index)
input_colors = input_colors.map({0: (0.3,)*3})
input_colors[select_roi] = (1,0,0)


fig = plt.figure(figsize=(6,6))
ax1 = fig.add_subplot(111,  projection='3d')
ax1.view_init(elev=20, azim=70)
ax1.scatter3D(input_cents.x,input_cents.y,input_cents.z, c=input_colors, s=70)
ax1.set_title('AL')

codf = mega_coef[gp_name].unstack(level='input_roi')
onrg_trains = codf.index.unique(level='on_rg')
targets = codf.index.unique(level='target')

n_ax = len(onrg_trains)*len(targets)
n_col = (n_ax+1)//2
n_row = 2
fig = plt.figure(figsize=(6*n_col,6*n_row))
icol = 0
for onrg_train in onrg_trains:
# for onrg_train in ['-38_0', '1_16', '24_40', '0_48']:

    icol += 1
    irow = 0
    for target in targets:
        i = icol+irow*n_col
        ax = fig.add_subplot(n_row,n_col, i, projection='3d')
        region = target.split('_')[-1]
        if region == 'lh':
            ax.view_init(elev=20, azim=70)
        elif region == 'cl':
            ax.view_init(elev=30, azim=-70)

        if irow == 0:
            ax.set_title(onrg_train)

        coef_on_target1 = codf.loc[ids[onrg_train,target,:],select_roi]
        target1_cents = cents[target].reindex(index=coef_on_target1.index.get_level_values('target_roi'))
        assert all(coef_on_target1.index.get_level_values('target_roi') == target1_cents.index)

        ax.scatter3D(target1_cents.x, target1_cents.y, target1_cents.z,
                      c=coef_on_target1, s=25,
                      cmap='RdBu_r',
                      vmax=0.2,vmin=-0.2)
        # ax2.set_title('Lateral Horn')

        irow += 1
plt.suptitle(select_roi)



"""
Plot multiple target projections for one trial period
"""
oneon_coef = mega_coef.loc[ids[:,['1_16']]]  # coefs for one on-range

# plt.ioff()
# plt.ion()

gp_name = 'gh_61'

gp_coef = oneon_coef.loc[gp_name]
gp_coef_pd = gp_coef.unstack()
gp_coef = gp_coef_pd.loc[:,~(gp_coef_pd == 0).all()].stack()  # remove input rois that got ignored in L1 regression
ninput = gp_coef.index.unique(level='input_roi').shape[0]  # number of non-zero inputs (preserved after L1 regression)


region_ord = dict(cl=1,lh=2)
targets = gp_coef.index.unique(level='target').tolist()
targets.sort(key=lambda s: region_ord[s.split('_')[-1]])




data_gp = data_gps.get_group(gp_name)
cents = data_gp.apply(lambda x: x.getCentroids())


input_cents = cents.filter(regex='.*_al').item()
target1_cents = cents.filter(regex='.*_cl').item()
target2_cents = cents.filter(regex='.*_lh').item()

nsample = 15
select_rois = pd.Series(gp_coef.index.unique(level='input_roi')).sample(nsample)

nrow = 3
ncol = nsample
subwidth = 2  # width of subplots
pane_brightness = 0.95



icol = 0
fig = plt.figure(figsize=(subwidth*ncol, 0.9*subwidth*nrow))
for select_roi in select_rois:
    icol += 1
    irow = 0
    input_colors = pd.Series(0,index=input_cents.index)
    input_colors = input_colors.map({0: (0.3,)*3})
    input_colors[select_roi] = (1,0,0)

    i = icol + irow * ncol
    ax1 = fig.add_subplot(nrow,ncol,i, projection='3d')
    ax1.view_init(elev=20, azim=70)
    ax1.scatter3D(input_cents.x,input_cents.y,input_cents.z, c=input_colors, s=20,depthshade=True)
    set_pane_brightness(ax1, pane_brightness)
    set_gridline_color(ax1, (0.7,) * 3)
    # ax1.set_title('AL')

    ax1.set_title(select_roi)
    irow += 1

    for target in targets:
        i = icol + irow * ncol
        ax = fig.add_subplot(nrow, ncol, i, projection='3d')
        region = target.split('_')[-1]
        if region == 'lh':
            ax.view_init(elev=20, azim=70)
            s = 5
        elif region == 'cl':
            ax.view_init(elev=30, azim=-70)
            s = 10

        if irow == 0:
            ax.set_title(onrg_train)

        coef_on_target1 = gp_coef.loc[:, target, :, select_roi]
        target1_cents = cents[target].reindex(index=coef_on_target1.index.get_level_values('target_roi'))
        assert all(coef_on_target1.index.get_level_values('target_roi') == target1_cents.index)

        ax.scatter3D(target1_cents.x, target1_cents.y, target1_cents.z,
                     c=coef_on_target1, s=s,edgecolors='k',linewidths=0.1,
                     cmap='RdBu_r',
                     vmax=0.2, vmin=-0.2, depthshade=False)

        set_pane_brightness(ax, pane_brightness)
        set_gridline_color(ax, (0.7,)*3)

        # ax.set_title(region)

        irow += 1
plt.subplots_adjust(left=0.01, bottom=0.05, right=0.99, top=0.95, wspace=0.05, hspace=0.05)
plt.suptitle(f'{gp_name}, {nsample} / {ninput}')


plt.rcParams['svg.fonttype'] = 'none'
plt.savefig(Path(r'F:\analysis\figures\0324') / (f'coef_proj_{gp_name}.svg'),
            transparent=True,
            bbox_inches='tight', pad_inches=0
            )




"""
pos vs neg (percent) coef scatter
"""

thresh = 0.03
use_onrg = '1_16'
line_exps = mega_coef.index.unique(level='exp')
subwidth = 5
fig, axes = plt.subplots(2,len(line_exps), figsize=(len(line_exps)*subwidth,2*subwidth))
for col, exp in enumerate(line_exps):
    for row, target_region in enumerate(['cl','lh']):
        this_coef = mega_coef[exp][use_onrg][exp + '_' + target_region].unstack()
        pos_pct = (this_coef > thresh).sum() / this_coef.shape[0]
        neg_pct = (this_coef < -thresh).sum() / this_coef.shape[0]
        assert neg_pct.index.tolist() == pos_pct.index.tolist()
        net_effect = pos_pct - neg_pct
        ax = axes[row,col]
        ax.scatter(neg_pct, pos_pct, c=net_effect,
                   cmap='RdBu_r',vmin=-1,vmax=1,
                   edgecolors='k',lw=0.5, s=100,
                   alpha=0.9)
        # ax.set_aspect('equal')
        if row == 0:
            ax.set_title(exp)
        else:
            ax.set_xlabel('Negative Coefficients (%)')
        if col == 0:
            if row == 0:
                ax.set_ylabel('CL, Positive Coefficients (%)')
            else:
                ax.set_ylabel('LH, Positive Coefficients (%)')

plt.setp(axes, xlim=(0,1), ylim=(0,1), aspect='equal')

plt.subplots_adjust(left=0.05, bottom=0.05, right=0.99, top=0.95, wspace=0.15, hspace=0.05)
plt.suptitle(f'Positive vs Negative Coefficients')


plt.rcParams['svg.fonttype'] = 'none'
plt.savefig(Path(r'F:\analysis\figures\0402') / (f'pos vs neg_coefs_v2.svg'),
            # transparent=True,
            bbox_inches='tight', pad_inches=0
            )



""" Project net effect back to AL input"""
fig, axes = plt.subplots(2,len(line_exps), figsize=(len(line_exps)*subwidth,2*subwidth*0.8),
                         subplot_kw={'projection':'3d'})
for col, exp in enumerate(line_exps):
    for row, target_region in enumerate(['cl','lh']):
        this_coef = mega_coef[exp][use_onrg][exp + '_' + target_region].unstack()
        pos_pct = (this_coef > thresh).sum() / this_coef.shape[0]
        neg_pct = (this_coef < -thresh).sum() / this_coef.shape[0]
        assert neg_pct.index.tolist() == pos_pct.index.tolist()
        net_effect = pos_pct - neg_pct
        net_effect.name = 'net'

        cents_al = datas[exp + '_al'].getCentroids()
        cents_al = cents_al.join(net_effect, how='left')

        ax = axes[row,col]
        ax.view_init(elev=20, azim=70)
        ax.scatter3D(cents_al.x,cents_al.y,cents_al.z,c=cents_al.net,
                     cmap='RdBu_r',vmin=-1,vmax=1,
                     edgecolors='k',lw=0.5, s=120,
                     depthshade=True,alpha=0.8)

        if row == 0:
            ax.set_title(exp)
        set_pane_brightness(ax, pane_brightness=0.97)
        set_gridline_color(ax, (0.7,) * 3)

plt.subplots_adjust(left=0.01, bottom=0.05, right=0.99, top=0.95, wspace=0.05, hspace=0.05)

plt.rcParams['svg.fonttype'] = 'none'
plt.savefig(Path(r'F:\analysis\figures\0402') / (f'net_effect_al.svg'),
            transparent=True,
            bbox_inches='tight', pad_inches=0
            )



"""
Distances to pos/neg core
"""
plt.ioff()
plt.ion()

use_onrg = '1_16'
line_exps = mega_coef.index.unique(level='exp')
subwidth = 5
# fig, axes = plt.subplots(2,len(line_exps), figsize=(len(line_exps)*subwidth,2*subwidth))
for col, exp in enumerate(line_exps):
    cents = data_gps.get_group(exp).apply(lambda x: x.getCentroids())
    for row, target_region in enumerate(['cl','lh']):
        dname = exp+'_'+target_region
        this_coef = mega_coef[exp][use_onrg][dname].unstack()
        quant = 0.75
        pos_thresh = this_coef.where(this_coef > thresh).quantile(quant)
        neg_thresh = this_coef.where(this_coef < -thresh).quantile(1-quant)

        this_cents = cents[dname]
        this_cents.index.name = 'target_roi'

        pos_rois = this_coef.stack()[(this_coef > pos_thresh).stack()].swaplevel().sort_index()
        pos_rois.name = 'coef'
        pos_cents = pos_rois.to_frame().join(this_cents, on='target_roi', how='left').drop(columns=['coef'])
        pos_mean_center = pos_cents.groupby('input_roi').mean()
        pos_mean_dist = np.sqrt((pos_cents.subtract(pos_mean_center, level='input_roi')**2).sum(axis=1)).mean(level='input_roi')
        pos_mean_dist.name = 'pos_mean_dist'

        neg_rois = this_coef.stack()[(this_coef < neg_thresh).stack()].swaplevel().sort_index()
        neg_rois.name = 'coef'
        neg_cents = neg_rois.to_frame().join(this_cents, on='target_roi', how='left').drop(columns=['coef'])
        neg_mean_center = neg_cents.groupby('input_roi').mean()
        neg_mean_dist = np.sqrt((neg_cents.subtract(neg_mean_center, level='input_roi')**2).sum(axis=1)).mean(level='input_roi')
        neg_mean_dist.name = 'neg_mean_dist'

        pos_neg_dist = pos_mean_dist.to_frame().join(neg_mean_dist, how='inner')

        ax = axes[row, col]

        sns.jointplot(x="neg_mean_dist", y="pos_mean_dist", data=pos_neg_dist, marginal_kws=dict(bins=15))
        plt.suptitle(dname)

        # if row == 0:
        #     ax.set_title(exp)
        # else:
        #     ax.set_xlabel('Negative ROIs\' Mean Distance to Centroid')
        # if col == 0:
        #     if row == 0:
        #         ax.set_ylabel('CL, Positive ROIs\' Mean Distance to Centroid')
        #     else:
        #         ax.set_ylabel('LH, Positive ROIs\' Mean Distance to Centroid')
        plt.rcParams['svg.fonttype'] = 'none'
        plt.savefig(Path(r'F:\analysis\figures\0402') / (f'mean_dist_{dname}.svg'),
                    # transparent=True,
                    bbox_inches='tight', pad_inches=0
                    )
    # break
    # plt.setp(axes, aspect='equal')
    #
    # plt.subplots_adjust(left=0.05, bottom=0.05, right=0.99, top=0.95, wspace=0.15, hspace=0.05)
    # plt.suptitle(f'Positive vs Negative ROIs\' Mean Distance to Centroid')





































