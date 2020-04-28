import numpy as np
import pandas as pd
import glob
import re
from collections import OrderedDict
import copy
from Tools import *
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
import os

rootDir = Path(r"my_path\exp_data")
pix_size = 0.325

line_region_order = ('oc_al','gh_al','gh_cl','gh_lh','mz_lh')
line_region_order_dict = {line_region:i for i, line_region in enumerate(line_region_order)}


class Loader:
    #gh-GH146, oc-Orco, mz-Mz699

    metaDir = rootDir / 'meta'
    exps={}

    expMap = {
        'gh_61': 'GH146_e61_1',
        'gh_59': 'GH146_e59_1',
        'gh_58': 'GH146_e58_1',
        'gh_57': 'GH146_e57_1',
        'gh_56': 'GH146_e56_1',
        'gh_54': 'GH146_e54_1',

        'mz_61': 'Mz699_e61_1',
        'mz_58': 'Mz699_e58_1',
        'mz_57': 'Mz699_e57_1',
        'mz_56': 'Mz699_e56_1',
        'mz_55': 'Mz699_e55_1',
        # 'mz_54': 'Mz699_e54_1', # remove, too much flashing

        'oc_61': 'Orco_e61_2',
        'oc_59': 'Orco_e59_1',
        'oc_58': 'Orco_e58_1',
        'oc_57': 'Orco_e57_2',
        'oc_56': 'Orco_e56_2',
        'oc_54': 'Orco_e54_1',
    }

    def __init__(self):

        for line_exp, expName in self.expMap.items():
            self.exps[line_exp] = self.__auto_read(expName)

        """
        Manual overwritting expMap items is allowed.
        """
        self.dnames = [] # data names
        for line_exp, subdict in self.exps.items():
            for region in subdict['data'].keys():
                self.dnames.append('_'.join([line_exp, region]))
        self.loaded_origin = {}
        self.loaded = {}
        self.rescale = {}


    def _load_rescale(self,do_rescale):
        import json
        target = rootDir / 'post curve fitting' / 'post.json'
        try:
            with open(target) as f:
                post_dict = json.load(f)

            # self.rescale = {}
            for k, v in post_dict.items():
                vv = pd.DataFrame(v)
                if bool(v):
                    vv.index = vv.index.str.split(',').map(lambda x: (x[0], int(x[1])))
                self.rescale[k] = vv
        except Exception as e:
            import traceback
            traceback.print_exc()
            print('Continue?')
            takeKBInput()
            # do_rscale = False

    def get(self,regex,filter_regex= None, roi_type={'gh_al':'glom','gh_cl':'bouton'}, remove_dup = True,
            do_rescale = True, correct_baseline = True, wrap_sr=False):
        if do_rescale and (not bool(self.rescale)):
            self._load_rescale(do_rescale)
            print('Finished loading rescalers...')
        if isinstance(regex,str):
            regex = re.compile(regex)
        select_dnames = list(filter(regex.match,self.dnames))
        load_datas = dict()
        for slct_dname in select_dnames:
            if filter_regex is None:
                load_datas[slct_dname] = self.get_one(slct_dname,roi_type, remove_dup, do_rescale,correct_baseline)
            else:
                load_datas[slct_dname] = self.get_one(slct_dname,roi_type, remove_dup, do_rescale,correct_baseline).filter(filter_regex)
        assert len(load_datas) >= 1
        self.status = {'regex': regex,
                       'filter_regex': filter_regex,
                       'roi_type': roi_type,
                       'remove_dup': remove_dup,
                       'do_rescale': do_rescale,
                       'correct_baseline': correct_baseline,
                       'wrap_sr': wrap_sr
                       }
        if len(load_datas) == 1 and (not wrap_sr):
            return next(iter(load_datas.values()))
        else:
            # return pd.Series(data=load_datas)
            return Datas(data=load_datas)

    def get_one(self, line_exp_region:str, roi_type, remove_dup, do_rescale, correct_baseline) ->pd.DataFrame:
        query = {'roi_type': roi_type, 'remove_dup': remove_dup, 'do_rescale': do_rescale, 'filter_regex': None}
        if line_exp_region in self.loaded:
            print('in get_one')
            print('roi_type :{}'.format(roi_type))
            print('dup_removed :{}'.format(self.loaded[line_exp_region].status['dup_removed']))
            print('do_rescale :{}'.format(do_rescale))
            print('rescaled :{}'.format(self.loaded[line_exp_region].status['rescaled']))
        if (line_exp_region in self.loaded) and (query == self.loaded[line_exp_region].query):
            print("Finished retrieving {}.".format(line_exp_region))
            return copy.deepcopy(self.loaded[line_exp_region])
        else:
            print("Loading {}.".format(line_exp_region))
            rawFPath, maskPath, metaPath, crctPath = self.__getPaths(line_exp_region)

            resp = dict(np.load(rawFPath))
            resp_ver=self.__getVersion(resp,"resp") # get version of time series
            try:
                print('*****************************')
                print("NOTES:"+str(resp.pop('note')))
                print('*****************************')
            except KeyError:
                print('No notes.')

            if maskPath is not None:
                print(maskPath)
                roimask = dict(np.load(maskPath))
                try:
                    print('*****************************')
                    print("NOTES:" + str(roimask.pop('note')))
                    print('*****************************')
                except KeyError:
                    print('No notes.')
                mask_ver = self.__getVersion(roimask, "masks")  # get version of ROI masks
            else:
                roimask = None
                mask_ver = None
            version=None
            #version check
            if None in [mask_ver,resp_ver]:
                print("WARNING:Skipping version check...")
            else:
                if mask_ver==resp_ver:
                    print("Passed version check...")
                    print("Version: "+resp_ver)
                    version=resp_ver
                else:
                    print("WARNING: Versions are inconsistent.")
                    print('Time series: {}, ROI masks: {}'.format(resp_ver,mask_ver))
                    print('Proceed?')
                    takeKBInput()
                    print('WARNING: Using inconsistent file versions.')
                    version=[resp_ver,mask_ver]

            fnParse=self.__fnParser(rawFPath)
            blks=fnParse['blks']
            meta = pd.read_csv(metaPath, keep_default_na=False)  # the complete meta file
            meta.rename({'trl_idx':'trl_num'},axis='columns',inplace=True)
            cmeta = meta.loc[meta['blk_idx'].isin(blks)]  # current meta, get only the processed blocks
            # get first and last stk_idx of each block
            cmeta_byGP = cmeta.groupby('blk_idx')

            # construct a dataframe which stores the start&end stk idx of each current block
            blk_info = pd.concat([cmeta_byGP.trl_startStk.min(), cmeta_byGP.trl_endStk.max()], axis=1)
            blk_info.rename({'trl_startStk': 'blk_startStk', 'trl_endStk': 'blk_endStk'}, axis='columns', inplace=True)

            rgs = []
            for index, row in blk_info.iterrows():
                # rgs.append(range(row.blk_startStk,row.blk_endStk))
                rgs.extend((range(row.blk_startStk, row.blk_endStk + 1)))
            # fixme is it possible to make indexing more efficient?
            rawF = pd.DataFrame(data=resp, index=rgs)

            # rawF2 = rawF - 100  # correct camera bias

            # 1st version, simple,reliable but slow
            # dff = pd.DataFrame(index=rawF.index, columns=rawF.columns, dtype='float64')
            # for idx, trl in cmeta.iterrows():
            #     f = rawF2.loc[range(trl.trl_startStk + 20, trl.stim1_startStk), :].median()
            #     dff.loc[range(trl.trl_startStk, trl.trl_endStk + 1), :] = (rawF2.loc[range(trl.trl_startStk, trl.trl_endStk + 1),:] - f) / f

            # # 2nd version, ~100x faster than 1st version
            stk_rgs = []  # store all stk indices
            preStim_rgs = []  # stk indices for preStim period used for baseline F calculation
            trllist_preStim = []  # trl_idx for each preStim stk
            trllist_whole = []  # trl_idx for all stk
            trl_lens = []
            on_rgs = []  # on based indices
            stim_list = []  # list of stim
            for idx, trl in cmeta.iterrows():
                trl_range = range(trl.trl_startStk, trl.trl_endStk + 1)
                trl_lens.append(len(trl_range))
                stk_rgs.extend(trl_range)
                preStim_range = range(trl.trl_startStk + 20, trl.stim1_startStk)
                preStim_rgs.extend(preStim_range)
                trllist_preStim.extend([trl.trl_num] * len(preStim_range))
                trllist_whole.extend([trl.trl_num] * (trl.trl_endStk - trl.trl_startStk + 1))
                on_rgs.extend(np.arange(trl.trl_startStk, trl.trl_endStk + 1) - trl.stim1_startStk)
                stim_list.extend([trl.stim1] * (trl.trl_endStk - trl.trl_startStk + 1))

            trl_idx = pd.Series(data=trllist_whole, index=stk_rgs,
                                name='trl_idx')  # stk - trl_idx mapping for all stk

            on_idx = pd.Series(data=on_rgs, index=stk_rgs, name='on_idx')
            stim_idx = pd.Series(data=stim_list, index=stk_rgs, name='stim_idx')
            if not correct_baseline:
                rawF2 = rawF - 100  # correct camera bias
                f_preStim = rawF2.loc[preStim_rgs].assign(trl_idx=trllist_preStim).groupby(
                    by='trl_idx').median()  # take median of each trl's preStim period
                if len(set(trl_lens)) == 1:  # if all trials have equal lengths, this is slightly faster
                    trl_len = trl_idx.shape[0] / f_preStim.shape[0]
                    f_preStimMat = np.repeat(f_preStim.to_numpy(), trl_len, axis=0)
                    dff = (rawF2 - f_preStimMat) / f_preStimMat
                else:  # if trials have varying lengths
                    f_preStimMat2 = np.empty_like(rawF2)
                    row_counter = 0
                    for (this_trl_len, idx) in zip(trl_lens, f_preStim.index):
                        f_preStimMat2[row_counter:row_counter + this_trl_len, :] = np.repeat(
                            f_preStim.loc[[idx], :].to_numpy(), this_trl_len, axis=0)
                        row_counter += this_trl_len

                    dff = (rawF2 - f_preStimMat2) / f_preStimMat2
                data = Data(line_exp_region, pd.Index(stk_rgs), dff, rawF, roimask, cmeta, on_idx, stim_idx,
                                trl_idx, version, None)

            else:
                """
                Load baseline corrected version
                """
                with open(crctPath, 'rb') as pickle_file:
                    crct = pickle.load(pickle_file)
                assert crct['version'] == resp_ver

                mi_crct_dff = pd.DataFrame(crct['crct_dff'])

                """
                Rescale
                """
                if do_rescale and (not self.rescale[line_exp_region].empty):
                    u_crct_dff = mi_crct_dff.swaplevel().unstack().T
                    u_crct_dff.loc[self.rescale[line_exp_region].index] = self.rescale[line_exp_region]
                    mi_crct_dff = u_crct_dff.unstack().T.swaplevel().sort_index()
                    print('Finished post rescaling...')
                    rescaled = True
                else:
                    rescaled = False

                """
                Reformat for Data object
                """
                crct_dff = mi_crct_dff.reset_index(drop=True)
                crct_dff.index += 1
                assert crct_dff.index[0] == 1

                bsl = pd.DataFrame(crct['bsls'])
                print('Finished loading {}.'.format(line_exp_region))
                print('=========================================================')
                print()
                # return {'dff':dff,'rawF':rawF,'mask':roimask},cmeta
                assert not crct_dff.isna().any().any() #check if any NaN
                data = Data(line_exp_region,pd.Index(stk_rgs),crct_dff,rawF,roimask,cmeta,on_idx,stim_idx,trl_idx,version,bsl)
                data.query = query
                data.status['rescaled'] = rescaled
                data.status['bsl_corrected'] = True

            """Select roi type or remove duplication"""
            if roi_type or remove_dup:
                data = data.get_roi(roi_type, remove_dup,self.expMap)

            self.loaded.update({line_exp_region: data})

            return copy.deepcopy(data)


    def __fetchFiles(self,dal=None, dlh=None, dcl=None, mal=None, mlh=None, mcl=None):
        exp = {}
        exp['data'] = {}
        exp['mask'] = {}
        if dal is not None:
            exp['data']['al'] = dal
        if dlh is not None:
            exp['data']['lh'] = dlh
        if dcl is not None:
            exp['data']['cl'] = dcl
        if mal is not None:
            exp['mask']['al'] = mal
        if mlh is not None:
            exp['mask']['lh'] = mlh
        if mcl is not None:
            exp['mask']['cl'] = mcl
        return exp

    def __auto_read(self, expName):
        fpaths = [x for x in (rootDir / expName).glob('**/*') if x.is_file()]
        fns = [x.name for x in fpaths]
        # print(fns)
        # fn_re = re.compile('_'.join([expName, '(lh|calyx|al)', 'blk\d+', '(mask|rawF)',
        #                              '(up|down).npz']))  # regular expression for filtering file names
        # fns = list(filter(fn_re.match, fns))
        # fns = [fn.split('.')[0] for fn in fns]
        # assert fns # CHECK: if fns is empty

        fn_re1 = re.compile('_'.join([expName, '(lh|calyx)', 'blk\d+', '(mask|rawF)',
                                      '(up|down).npz']))  # regular expression for filtering file names
        fns1 = list(filter(fn_re1.match, fns))
        # print(f'fns1: {fns1}')
        fns1 = [fn.split('.')[0] for fn in fns1]

        fn_re2 = re.compile('_'.join([expName, '(al)', 'blk\d+', '(mask|rawF)',
                                      '(up|down)_small.npz']))  # regular expression for filtering file names
        fns2 = list(filter(fn_re2.match, fns))
        # print(f'fns2: {fns2}')
        fns2 = [fn.split('.')[0] for fn in fns2]
        cat_fns = fns1 + fns2
        assert cat_fns

        dal = None
        dlh = None
        dcl = None
        mal = None
        mlh = None
        mcl = None
        for fn in cat_fns:
            try:
                _, _, _, region, _, ftype, side = fn.split('.')[0].split('_')
            except:
                _, _, _, region, _, ftype, side , appendix = fn.split('.')[0].split('_')
            if region == 'al':
                if ftype == 'rawF':
                    dal = fn
                if ftype == 'mask':
                    mal = fn
            if region == 'calyx':
                if ftype == 'rawF':
                    dcl = fn
                if ftype == 'mask':
                    mcl = fn
            if region == 'lh':
                if ftype == 'rawF':
                    dlh = fn
                if ftype == 'mask':
                    mlh = fn
        return self.__fetchFiles(dal=dal, dlh=dlh, dcl=dcl, mal=mal, mlh=mlh, mcl=mcl)

    def __getPaths(self,line_exp_region, ):
        maskPath = None

        nameElm = line_exp_region.split('_')  # name elements
        assert len(nameElm) == 3
        region = nameElm[-1]
        line_exp = '_'.join(nameElm[:2])

        fn = self.exps[line_exp]['data'][region] + '.npz'
        expName = '_'.join(fn.split('_')[:3])
        rawFPath = rootDir / expName / fn
        metaPath = self.metaDir / (expName + '_ExpInfo.csv')
        try:
            maskn = self.exps[line_exp]['mask'][region] + '.npz'
            maskPath = rootDir / expName / maskn
        except KeyError:
            print("ROI masks not found. Proceed?")
            takeKBInput()

        crctPath = rootDir / 'corrected' / (line_exp_region + '.file')
        return rawFPath, maskPath, metaPath, crctPath


    def __getVersion(self, file: dict, name:str):
        try:
            version = str(file.pop('version'))
        except KeyError:
            print("No version found for {}. Proceed?".format(name))
            takeKBInput()
            version = None
            print("WARNING:Proceeding without version check.")
        return version

    def __fnParser(self,fpath):
        output = {}
        # get file name
        fn_parts = fpath.name.split('.')
        assert len(fn_parts) == 2
        fn = fn_parts[0]

        # get block numbers
        blk_str = re.findall('_blk\d+_', fn)
        assert len(blk_str) == 1, "Found multiple or no matching for block numbers."  # can have only one match
        blk_str = blk_str[0]
        # extract blk numbers: take string out of type(blk_str)=list, index to get number substring, convert to a list of single digits
        # blks = list(map(int, filter(str.isdigit, blk_str))) #this mapping works by itself, but for unknown resone, doesn't work here
        blks = [int(x) for x in list(blk_str[4:-1])]
        output['blks'] = blks
        return output
# todo add a file dict checker function?


class Data:

    def __init__(self,name,index,dff,rawF,mask,meta,on_idx,stim_idx,trl_idx,version,bsl):
        assert dff.shape == rawF.shape
        self.name = name
        self.index = index
        self.dff = dff
        self.rawF = rawF
        self.mask = mask
        self.meta = meta
        self.on_idx = on_idx
        self.stim_idx = stim_idx
        self.trl_idx = trl_idx
        self.version = version
        self.shape = dff.shape
        self.bsl = bsl
        self.query = {'roi_type': None, 'remove_dup': None, 'do_rescale': None, 'filter_regex': None}
        self.status = {'bsl_corrected': False, 'rescaled': False,'roi_type': None, 'dup_removed': False,
                       'filtered_trls': False}

    def __repr__(self):
        return self.name

    def __getitem__(self, item):
        """ NOTE!!! This behavior is different from a direct indexing on a data frame.
        NOTE It's actually .loc[]."""
        if isinstance(item,slice):
            idx = item
        else:
            idx = list(item)

        new_dff = self.dff.loc[idx]
        new_rawF = self.rawF.loc[idx]
        new_on_idx = self.on_idx.loc[idx]
        new_stim_idx = self.stim_idx.loc[idx]
        new_trl_idx = self.trl_idx.loc[idx]
        new_index = new_dff.index
        data = Data(self.name,new_index,new_dff,new_rawF,self.mask,self.meta,new_on_idx,new_stim_idx,new_trl_idx,self.version,self.bsl)
        data.query = copy.deepcopy(self.query)
        data.status = copy.deepcopy(self.status)
        data.status['filtered_trls'] = True
        return data

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def reorderByTrl(self, reorder_trl:list):
        """
        reorder data by given trl sequence
        """
        trl_idc = self.trl_idx.groupby(self.trl_idx).indices
        reord_idc = []
        for trl in reorder_trl:
            reord_idc.extend(self.index[trl_idc[trl]])  # groupby removes original indices, must remap
        return self[reord_idc]

    def reorderByData(self,data1):
        """
        reorder data2 based on data1 stim1 sequences
        Matching rules:
        1. for duplicate stims, match by monotonic order of trl_num
        2. for unique stims, direct match
        """
        data2=self
        ts1 = data1.get_trl_stim().reset_index(drop=True)  # ts meaning: t - trl_num, s - stim1
        ts2 = data2.get_trl_stim().reset_index(drop=True)
        # make sure have same set of stim, including duplicates
        assert np.array_equal(ts1.stim1.sort_values().values, ts2.stim1.sort_values().values)
        stim_counts = ts1.stim1.value_counts()
        dup_stims = stim_counts.index[stim_counts > 1]  # list of duplicate stims

        idx_map = {}  # keys: ts2-index, values: ts1-index, after matching
        # match duplicate stims
        for dup_stim in dup_stims:
            dup_idx1 = ts1.loc[ts1.stim1 == dup_stim].sort_values(by='trl_num').index.tolist()
            dup_idx2 = ts2.loc[ts2.stim1 == dup_stim].sort_values(by='trl_num').index.tolist()
            idx_map.update(dict(zip(dup_idx2, dup_idx1)))
        # match unique stims
        uni_ts1 = ts1.loc[set(ts1.index) - set(idx_map.values())]
        uni_ts2 = ts2.loc[set(ts2.index) - set(idx_map.keys())]
        dic1 = dict(zip(uni_ts1.stim1, uni_ts1.index))
        dic2 = dict(zip(uni_ts2.stim1, uni_ts2.index))
        idx_map.update({idx2: dic1[stim] for stim, idx2 in dic2.items()})

        # list of trl_num for reodering data2 to match data1
        reord_trl = ts2.rename(mapper=idx_map, axis=0).sort_index().trl_num.tolist()
        # apply reodering
        return data2.reorderByTrl(reord_trl)

    def reset_index(self,inplace=False):
        if inplace:
            data = self
        else:
            data = copy.deepcopy(self)
        for attr, var in self.__getPdAttrs().items():
            setattr(data, attr, var.reset_index(drop=True))
        setattr(data, "index", data.dff.index)
        return data
    
    def filter(self, regex: "regex", flt_blks: "list of ints" = None,sort=True):
        return self.filter_fromMeta(self.meta, regex= regex, flt_blks= flt_blks, sort= sort)
    
    def filter_fromMeta(self, meta, regex: "regex", flt_blks: "list of ints" = None,sort=True):
        if type(regex) == str:
            regex = re.compile(regex)
        idxCon = meta.stim1.apply(lambda x: bool(regex.match(x)))
        sub_info = meta.loc[idxCon, :]
        if flt_blks is not None:
            sub_info = meta.loc[meta.blk_idx.isin(flt_blks), :]
        if sort:
            sub_info = sub_info.sort_values('stim1')

        # construct condition based sub resp dFrame
        subrgs = []
        for index, row in sub_info.iterrows():
            subrgs.extend((range(row.trl_startStk, row.trl_endStk + 1)))
        sub_data = self[subrgs]
        setattr(sub_data, "meta", sub_info)
        sub_data.query['filter_regex'] = regex
        # sub_data.status
        return sub_data

    def roi_type(self,to_keep, query_identity, dup_removed):
        sub_data = copy.deepcopy(self)
        setattr(sub_data,"dff",self.dff.loc[:,to_keep])
        setattr(sub_data,"rawF",self.rawF.loc[:,to_keep])
        setattr(sub_data,"mask",{k: self.mask[k] for k in to_keep})
        setattr(sub_data,"shape",sub_data.dff.shape)
        sub_data.status['roi_type'] = query_identity
        sub_data.status['dup_removed'] = dup_removed
        return sub_data


    def get_uroi(self):
        rm_dict = self.get_roi_meta()
        if rm_dict is None:
            return None
        else:
            return sorted(list(rm_dict['merge'].keys()))

    def get_id(self,expMap,keys=None):
        if self.name.split('_')[-1] == 'cl':
            rm_dict = self.load_identity(expMap)
            id_dict = {k: [] for k in ['bouton', 'stem', 'mix', 'other']}
            for roi, id in rm_dict['identity'].items():
                id_dict[id].append(roi)
        else:
            rm_dict = self.get_roi_meta()
            id_dict = {k: [] for k in ['neuron', 'glom', 'stem', 'mix', 'other']}
            for roi, id in rm_dict['identity'].items():
                id_dict[id].append(roi)

        if keys is None:
            return id_dict
        else:
            if type(keys) is str:
                keys = [keys]

            a = [id_dict[k] for k in keys]
            b = []
            for sub in a:
                b.extend(sub)
            return sorted(b)

    def get_roi(self, query_identities, remove_dup, expMap):
        line, _, region = self.name.split('_')
        dkind = '_'.join([line,region])
        if line == 'oc':
            # skip id
            can_do_id = False
            can_do_merge = True
        elif line == 'gh':
            if region == 'al':
                # can do both
                can_do_id = True
                can_do_merge = True
            if region == 'cl':
                # can do id in the future
                can_do_id = True
                can_do_merge = False
            if region == 'lh':
                # skip both for now
                can_do_id = False
                can_do_merge = False
        elif line == 'mz':
            # skip both for now
            can_do_id = False
            can_do_merge = False
        query_identity = query_identities.get(dkind,None)
        do_id = (query_identity is not None) and can_do_id
        do_merge = can_do_merge and remove_dup
        dup_removed = False
        roi_sets = []
        if do_id:
            roi_sets.append(set(self.get_id(expMap, keys=query_identity)))
        if do_merge:
            roi_sets.append(set(self.get_uroi()))
            dup_removed = True
        if len(roi_sets) == 2:
            to_keep = sorted(list(roi_sets[0].intersection(roi_sets[1])))
        elif len(roi_sets) == 1:
            to_keep = roi_sets[0]
        else:
            return self
        return self.roi_type(to_keep, query_identity, dup_removed)

    def get_roi_meta(self):
        rmDir = rootDir / 'roi_meta'
        rmPath = rmDir / (self.name + '_meta.json')
        import os.path
        if os.path.isfile(rmPath):
            import json
            with open(rmPath) as f:
                return json.load(f)
        else:
            return None

    def load_identity(self,expMap):
        idPath = rootDir / 'Labels' / (expMap[self.name[:-3]] + '_calyx_label.npy')
        id_dict = np.load(idPath).item()
        return id_dict

    def recover_trl_order_for_mi(self,mi_data):
        assert len(mi_data.index.names) == 2  # only two index levels allowed
        level_names = [name for name in mi_data.index.names]
        level_of_trlidx = mi_data.index.names.index('trl_idx')
        level_names.remove('trl_idx')
        name_of_other = level_names[0]
        level_of_other = mi_data.index.names.index(name_of_other)

        trl_order = self.trl_idx.unique()

        trl_to_mi = {trl_idx: [] for trl_idx in trl_order}
        for mi in mi_data.index.values:
            for trl_idx in trl_order:
                if mi[level_of_trlidx] == trl_idx:
                    trl_to_mi[trl_idx].append(mi)
        ori_idx = []
        for trl_idx in trl_order:
            ori_idx.extend(trl_to_mi[trl_idx])
        if level_of_trlidx == 0:
            ori_mi_data = mi_data.reindex(ori_idx)
        if level_of_trlidx == 1:
            ori_mi_data = mi_data.reindex(ori_idx).sort_index(level=level_of_other, sort_remaining=False)
        return ori_mi_data

    def getUnitDff(self,flat=False, trim_onidx=True):
        """
        row: roi,trl_idx, col: on_idx in a trl
        Note the output is actually a transpose of standard dff format.
        Thus it allows easy creation of the standard X input for sklearn APIs"""
        mi_dff = self.dff.assign(trl_idx=self.trl_idx, on_idx=self.on_idx).set_index(
            ['trl_idx', 'on_idx'])  # row multi-indexed by trl_idx, on_idx
        unit_dff = mi_dff.unstack(level=0)  # reshaped s.t. each col has only one trl, col MI by roi, trl_idx
        unit_dff_T = unit_dff.T
        unit_dff_T.index.rename('roi', level=0, inplace=True)
        if flat:
            unit_dff_T.index = unit_dff_T.index.to_flat_index()  # row: flat index (roi,trl_idx), col: on_idx in a trl
        udff = self.recover_trl_order_for_mi(unit_dff_T)
        if trim_onidx:
            udff = udff.iloc[:,10:]
        return udff

    def getMiDff(self, sort=False, trim_onidx=True, use_stim_level=False):
        """
        :return:
        index: Muliti-index: trl_idx, on_idx
        columns: rois
        """

        def replace_trl_level_with_stim(self,miDff):
            miDff['stim1'] = self.trl_to_stim(miDff.index.get_level_values('trl_idx'))
            return miDff.set_index('stim1', append=True).droplevel('trl_idx').swaplevel()

        miDff = self.recover_trl_order_for_mi(self.getUnitDff(trim_onidx=trim_onidx).stack().unstack(level='roi'))
        if use_stim_level:
            miDff = replace_trl_level_with_stim(self,miDff)
        if sort:
            return miDff.sort_index()
        else:
            return miDff


    def get_unit_rawF(self,flat=False):
        """
        row: roi,trl_idx, col: on_idx in a trl
        Note the output is actually a transpose of standard rawF format.
        Thus it allows easy creation of the standard X input for sklearn APIs"""
        mi_rawF = self.rawF.assign(trl_idx=self.trl_idx, on_idx=self.on_idx).set_index(
            ['trl_idx', 'on_idx'])  # row multi-indexed by trl_idx, on_idx
        unit_rawF = mi_rawF.unstack(level=0)  # reshaped s.t. each col has only one trl, col MI by roi, trl_idx
        unit_rawF_T = unit_rawF.T
        unit_rawF_T.index.rename('roi', level=0, inplace=True)
        if flat:
            unit_rawF_T.index = unit_rawF_T.index.to_flat_index()  # row: flat index (roi,trl_idx), col: on_idx in a trl
        return self.recover_trl_order_for_mi(unit_rawF_T)

    def get_mi_rawF(self):
        return  self.recover_trl_order_for_mi(self.get_unit_rawF().stack().unstack(level='roi'))

    def plotMasks(self, mode= None):
        if mode is None:
            mask_sr = pd.Series(data=self.mask).apply(lambda x: x.item())
        elif mode == bool:
            mask_sr = pd.Series(data=self.mask).apply(lambda x: x.item().astype(bool))
        elif mode == float:
            mask_sr = pd.Series(data=self.mask).apply(lambda x: x.item().astype(bool).astype(float))
        else:
            print('Wrong mode input.')
        slcs = list(set(mask_sr.index.map(lambda x: x.split('_')[0])))
        slcs.sort()
        slcs_mask = {slc: mask_sr.filter(regex=slc + '_.+', axis=0) for slc in slcs}

        plt.rcParams["axes.grid"] = False
        fig, _ = plt.subplots(1, len(slcs))
        for sp_idx, slc in enumerate(slcs):
            ax = fig.add_subplot(1, len(slcs), sp_idx + 1)
            ax.imshow(slcs_mask[slc].sum().todense())
        plt.title(self.name)

    def getCentroids(self):
        """ To prevent behavior like data1.getCentroids(data2.mask),
         the getCentroids_fromMask API is needed for the subclass gpData"""
        return self.getCentroids_fromMask(self.mask)

    def getCentroids_fromMask(self,roimask):

        z_gap = 8
        allkeys = list(roimask.keys())
        d1, d2 = roimask[allkeys[0]].item().todense().shape
        Coor = np.matrix([np.outer(np.ones(d2), np.arange(d1)).ravel(),
                          np.outer(np.arange(d2), np.ones(d1)).ravel()], dtype='float64')
        cm_dict = {}
        for keyv in allkeys:
            a = roimask[keyv].item().todense().T.flatten()
            cm = (Coor * a.T / a.T.sum(axis=0)).T
            cm_dict[keyv] = tuple(np.ravel(cm))

        col_names = []
        xs = []
        ys = []
        zs = []
        for key, value in cm_dict.items():
            # col_names.append(key)
            slc, roi = key.split('_')
            roi_num = "{0:0=3d}".format(int(re.findall('\d+', roi)[0]))  # not original version
            col_names.append(slc + '_roi' + roi_num)  # not original version
            xs.append(value[0])
            ys.append(value[1])
            # zs.append(int(key[3:5]))
            zs.append(int(re.findall('\d+', key.split('_')[0])[0]))  # in case slice1, slice2

        xs = np.asarray(xs)
        ys = np.asarray(ys)
        zs = np.asarray(zs)
        zs = np.abs(zs - np.amax(zs))
        centroids = pd.DataFrame(index=col_names,
                                 data={'x': xs * pix_size, 'y': ys * pix_size, 'z': zs * z_gap})  # .reindex_like()
        return centroids

    def get_centroid_colors(self):
        cents = self.getCentroids()
        max_coor = cents.max(axis=0)
        min_coor = cents.min(axis=0)
        rgb_coor = (cents - min_coor) / (max_coor - min_coor)
        colors = rgb_coor.apply(lambda r: (r.x, r.y, r.z), axis=1)
        return colors

    def plot_centroid_colors(self):
        cents = self.getCentroids()
        colors = self.get_centroid_colors()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter3D(cents.x, cents.y, cents.z, c=colors, s=10, depthshade=False)
        ax.set_xlabel('x (um)')
        ax.set_ylabel('y (um)')
        ax.set_zlabel('z (um)')
        plt.suptitle(self.name)

    def plot_color_model(self):
        coors = np.linspace(0, 1, 10)
        x, y, z = np.meshgrid(coors, coors, coors)
        xf = x.flatten()
        yf = y.flatten()
        zf = z.flatten()
        colors = [(x, y, z) for x, y, z in zip(xf, yf, zf)]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter3D(xf, yf, zf, c=colors, s=10, depthshade=False)

    def get_roi_radius(self):
        mask_radius = {}
        for roi, roimask in self.mask.items():
            area = roimask.item().count_nonzero() * pix_size ** 2
            radius = np.sqrt(area / np.pi)
            mask_radius[roi] = radius

        return pd.Series(mask_radius, name='radius')
    # def plotHeatmap(self,data='dff',vmin=-0.5, vmax=1.5):
    #     if data == 'dff':
    #         df = self.dff
    #     if data == 'rawF':
    #         df = self.rawF
    #     sub_info = self.meta
    #     mpl.rcParams["axes.grid"] = False
    #     f, axs = plt.subplots(1, sub_info.shape[0], sharey=True, figsize=(16, 12))
    #     axCounter = 0
    #     images = []
    #     for idx, trl in sub_info.iterrows():
    #         # images.append(axs[axCounter].imshow(df.loc[range(trl.stim1_startStk, trl.stim1_startStk+4), :].T,vmin=vmin, vmax=vmax,cmap='bwr'))
    #         images.append(
    #             axs[axCounter].imshow(df.loc[range(trl.trl_startStk, trl.trl_endStk + 1), :].T, vmin=vmin, vmax=vmax))
    #         axs[axCounter].axvline(x=trl.stim1_startStk - trl.trl_startStk, color='w', linewidth=1)
    #         axs[axCounter].axvline(x=trl.stim1_endStk - trl.trl_startStk, color='w', linewidth=1)
    #         axs[axCounter].set_xlabel(trl.stim1)
    #         axs[axCounter].set_aspect('auto')
    #         axCounter += 1
    #
    #     f.colorbar(images[0], ax=axs, orientation='vertical', fraction=.1)
    #     f.suptitle(self.name)

    def get_odor_conc(self,):
        trl_stim = self.get_trl_stim()
        odor_conc = trl_stim.stim1.str.extract('(?P<odor>\d?\w+\d?\w+)(?P<conc>\d{2})')
        return odor_conc

    def get_trl_info(self):
        odor_conc = self.get_odor_conc()
        trl_stim = self.get_trl_stim()
        trl_info = pd.concat([trl_stim, odor_conc], axis=1).set_index('trl_num')
        return trl_info

    def trl_sorter(self, order = 'odor_low-high'):
        odor_conc = self.get_odor_conc()
        trl_stim = self.get_trl_stim()
        if order == 'odor_high-low':
            reorder_idx = odor_conc.sort_values('conc', ascending=True).sort_values('odor')
        elif order == 'odor_low-high':
            reorder_idx = odor_conc.sort_values('conc', ascending=False).sort_values('odor')
        elif order == 'low-high_odor':
            reorder_idx = odor_conc.sort_values('odor').sort_values('conc', ascending=False)
        elif order == 'high-low_odor':
            reorder_idx = odor_conc.sort_values('odor').sort_values('conc', ascending=False)
        else:
            raise KeyError(f'Keyword: {order} does not exist for kwarg: order.')

        trl_order = trl_stim.reindex(reorder_idx.index).trl_num
        return trl_order

    def plot_heatmap(self,rg_on=None,vmin=-1.5, vmax=1.5,colorbar=False,
                     sup_title= True, time_mode=False, cmap='RdBu_r',
                 order='default', subplot_grid=None, figsize=(20,10), stim_line_kwargs = dict(color='w', linewidth=1.5),
                     tick_font_size=14,label_font_size = 18   ):

        to_plot = self.getMiDff().sort_index()
        # trls = to_plot.index.unique(level='trl_idx')
        trl_stim = self.get_trl_stim()
        trls = trl_stim.trl_num
        odor_conc = trl_stim.stim1.str.extract('(?P<odor>\d?\w+\d?\w+)(?P<conc>\d{2})')

        if order == 'default':
            trl_order = trls.sort_values()
        elif order == 'high-low':
            reorder_idx = odor_conc.sort_values('conc', ascending=True).sort_values('odor')
            trl_order = trl_stim.reindex(reorder_idx.index).trl_num
        elif order == 'low-high':
            reorder_idx = odor_conc.sort_values('conc', ascending=False).sort_values('odor')
            trl_order = trl_stim.reindex(reorder_idx.index).trl_num
        else:
            raise KeyError(f'Keyword: {order} does not exist for kwarg: order.')

        if subplot_grid is None:
            subplot_grid = (1, len(trl_order))

        nrows, ncols = subplot_grid

        mpl.rcParams["axes.grid"] = False
        f, axs = plt.subplots(nrows, ncols, sharey=True, sharex=True, squeeze=False,
                              figsize=figsize)
        axCounter = 0
        images = []
        for idx, trl in enumerate(trl_order):
            if rg_on is None:
                this_trl = to_plot.loc[pd.IndexSlice[trl, :], :]
            else:
                this_trl = to_plot.loc[pd.IndexSlice[trl, rg_on], :]

            on_inds = this_trl.index.get_level_values(level='on_idx')
            row_idx, col_idx = divmod(axCounter, ncols)
            if time_mode:
                images.append(
                    axs[row_idx, col_idx].imshow(this_trl.T, vmin=vmin, vmax=vmax,
                                                 cmap=cmap,
                                                 extent=[on_inds.min() / 4, on_inds.max() / 4, this_trl.shape[1], 0]))
                axs[row_idx, col_idx].axvline(x=0, **stim_line_kwargs)
                axs[row_idx, col_idx].axvline(x=4, **stim_line_kwargs)

            else:
                images.append(
                    axs[row_idx, col_idx].imshow(this_trl.T, vmin=vmin, vmax=vmax,
                                                 cmap=cmap,
                                                 extent=[on_inds.min(), on_inds.max(), this_trl.shape[1], 0]))
                axs[row_idx, col_idx].axvline(x=0 - 0.5, **stim_line_kwargs)
                axs[row_idx, col_idx].axvline(x=16 + 0.5, **stim_line_kwargs)

            axs[row_idx, col_idx].set_title(self.trl_to_stim(trl), fontname="Arial", fontsize=label_font_size)
            axs[row_idx, col_idx].set_aspect('auto')
            axCounter += 1

        [ax.set_ylabel('ROIs', fontname="Arial", fontsize=label_font_size) for ax in axs[:, 0]]
        [ax.set_yticklabels(ax.get_yticks(), fontname="Arial", fontsize=tick_font_size) for ax in axs[:, 0]]
        [ax.set_xlabel('Time (s)', fontname="Arial", fontsize=label_font_size) for ax in axs[-1, :]]
        if sup_title:
            plt.suptitle(self.name)
        if colorbar:
            f.subplots_adjust(right=0.9)
            cbar_ax = f.add_axes([0.92, 0.15, 0.02, 0.7])
            f.colorbar(images[-1], cax=cbar_ax)
            cbar_ax.tick_params(labelsize=tick_font_size)
            cbar_ax.set_title(r'$\Delta F / F$', fontname="Arial", fontsize=label_font_size)

        stk_rate = 4  # unit: stks/s
        stim_dur = stim_off = 4  # unit: s

        xtick_t_interval = 4  # unit: s
        xtick_stepsize = stk_rate * xtick_t_interval  # in seconds

        xtick_loc_post_on = np.arange(stk_rate * stim_off, on_inds.max() + 0.01, xtick_stepsize)
        xtick_loc_pre_on = np.arange(0, on_inds.min() - 0.01, -xtick_stepsize)
        xtick_loc = np.concatenate([xtick_loc_pre_on, xtick_loc_post_on])
        xtick_labels = xtick_loc / stk_rate
        xtick_labels.astype(int)
        all_int = all(map(lambda x: x.is_integer(), xtick_labels))

        for ax in axs[-1, :]:
            ax.set_xticks(xtick_labels)
            if all_int:
                ax.set_xticklabels(xtick_labels.astype(int), fontname="Arial", fontsize=tick_font_size)
            else:
                ax.set_xticklabels(xtick_labels, fontname="Arial", fontsize=tick_font_size)

    """ Utilities """
    def __getPdAttrs(self):
        """
        get attributes that are pd.Series or pd.DataFrame
        EXCEPT meta
        """
        pdAttrs={}
        for attr, var in vars(self).items():
            if isinstance(var, pd.Series) or isinstance(var, pd.DataFrame):
                if attr!='meta':
                    pdAttrs[attr]=var
        return pdAttrs

    def unwrap(self):
        """
        FOR DEBUGGING ONLY
        a function for passing out private function"""
        return self.__getPdAttrs()

    def getStim(self):
        return self.stim_idx.loc[self.getTrlStarts().first_index]

    def getTrlStarts(self): # NOTE: returns a df.
        return self.trl_idx.reset_index().groupby('trl_idx',sort = False).first().rename({'index':'first_index'},axis=1)

    def get_trl_stim(self):
        """ Index is the first index of the corresponding trl """
        trl_ = self.getTrlStarts()
        _trl = pd.Series(trl_.index, index=trl_.first_index)
        _stim = self.getStim()
        return pd.concat([_trl, _stim], axis=1).rename({'trl_idx':'trl_num','stim_idx':'stim1'},axis='columns')

    def trl_to_stim(self,trls):
        if isinstance(trls,str):
            trls = int(trls)
        if isinstance(trls,int):
            trls = [trls]
        trl_stim = self.get_trl_stim().set_index('trl_num')

        stims = trl_stim.loc[[*trls]].values.tolist()
        if len(stims) == 1:
            return stims[0].pop()
        else:
            stims_unwrap = []
            for stim in stims:
                stims_unwrap.extend(stim[:])
            return stims_unwrap
    def stim_to_trl(self,stims):
        if isinstance(stims,str):
            stims=[stims]
        trl_stim = self.get_trl_stim().set_index('trl_num')
        return trl_stim[trl_stim.isin(stims)].dropna()

    def checkIndex(self):
        return [self.index.equals(x.index) for x in [self.dff, self.rawF, self.on_idx, self.stim_idx, self.trl_idx]]

    def get_kind(self):
        name_parts = self.name.split('_')
        return '_'.join([name_parts[0],name_parts[2]])


def get_kind(dname):
    name_parts = dname.split('_')
    return '_'.join([name_parts[0],name_parts[2]])

def get_neuro(dname):
    table = {'gh':'ePN','mz':'iPN','oc':'ORN'}
    name_parts = dname.split('_')
    return '_'.join([table[name_parts[0]],name_parts[2]])

# todo check correctness
# todo cope with varying trl lengths
class gpData(Data):
    def __init__(self,datas):
        is_original = False
        data1=datas[0]
        assert all([data1.name != data.name for data in datas[1:]]) # check if datas being concatenated are all distinct
        assert all([data1.index.shape[0] == data.index.shape[0] for data in datas[1:]])  # check if all datas have same time length
        sameExp = ifAllEqual([data.name.split('_')[1] for data in datas])
        if sameExp:
            new_data1 = data1
        else:
            new_data1 = data1.reset_index()

        new_datas = [new_data1]
        if sameExp:
            for data in datas[1:]:
                # new_datas.append(data.reorderByData(data1))
                new_datas.append(data[data1.index]) # for safer indexing behavior
            is_original = True
        else:
            for data in datas[1:]:
                new_datas.append(data.reorderByData(data1).reset_index())

        for new_data in new_datas:
            new_data.dff.rename(lambda x: new_data.name + '.' + x, axis='columns', inplace=True)
            new_data.rawF.rename(lambda x: new_data.name + '.' + x, axis='columns', inplace=True)

        new_dff = pd.concat([new_data.dff for new_data in new_datas], axis=1)
        new_rawF = pd.concat([new_data.rawF for new_data in new_datas], axis=1)

        new_name = tuple([x.name for x in datas]) # element order matters for these attributes, set to immutable
        new_index = new_data1.index
        new_meta = tuple([x.meta for x in datas])
        new_mask = tuple([x.mask for x in datas])
        new_on_idx = new_data1.on_idx
        new_stim_idx = new_data1.stim_idx
        new_trl_idx = new_data1.trl_idx
        new_version = tuple([x.version for x in datas])
        Data.__init__(self,new_name,new_index,new_dff,new_rawF,new_mask,new_meta,new_on_idx,new_stim_idx,new_trl_idx,new_version)
        # super().__init__(new_name,new_index,new_dff,new_rawF,new_mask,new_meta,new_on_idx,new_stim_idx,new_trl_idx,new_version)
        self.is_original = is_original  # if no reset_index

    def __repr__(self):
        names = ''
        for name in self.name:
            names = names+name+','
        return names

    def filter(self, regex: "regex", flt_blks: "list of ints" = None,sort=True):
        if self.is_original:
            return super().filter_fromMeta(self.meta[0], regex = regex, flt_blks = flt_blks, sort=sort)
        else:
            # todo can define using on_idx, stim_idx, trl_idx
            print("Undifined for reindexed gpData.")

    # todo column name inconsistency exists
    def getCentroids(self,name_or_idx):
        if isinstance(name_or_idx, int):
            idx=name_or_idx
        if isinstance(name_or_idx, str):
            idx=self.name.index(name_or_idx)
        return super().getCentroids_fromMask(self.mask[idx])

    def splitData(self,return_type='list'):
        # list form, labels only, for np correspondence
        # dict form
        # direct return
        if return_type == "list":
            data_lbs = np.empty(self.dff.shape[1], dtype=int)
            for idx, name in enumerate(self.name):
                reg = re.compile(name + '\..+')
                col_locs = list(map(lambda x: self.dff.columns.get_loc(x), self.dff.filter(regex=reg, axis=1).columns))
                data_lbs[col_locs] = idx

            data_lbs = data_lbs.tolist()
            return data_lbs

        if return_type == "dict":
            data_cols=OrderedDict()
            for idx, name in enumerate(self.name):
                reg = re.compile(name + '\..+')
                data_cols[name] = self.dff.filter(regex=reg, axis=1).columns
            return data_cols


#
#
# def filter(meta:pd.DataFrame,regex:"regex",dff:pd.DataFrame=None,rawF:pd.DataFrame=None,
#            flt_blks:"list of ints"=None):
#     idxCon=meta.stim1.apply(lambda x: bool(regex.match(x)))
#     sub_info=meta.loc[idxCon,:]
#     if flt_blks is not None:
#         sub_info=meta.loc[meta.blk_idx.isin(flt_blks),:]
#     sub_info=sub_info.sort_values('stim1')
#     # construct condition based sub resp dFrame
#     subrgs=[]
#     on_rgs=[]
#     stim_list=[]
#     trl_list=[]
#     for index,row in sub_info.iterrows():
#         subrgs.extend((range(row.trl_startStk,row.trl_endStk+1)))
#         on_rgs.extend(np.arange(row.trl_startStk,row.trl_endStk+1)-row.stim1_startStk)
#         stim_list.extend([row.stim1]*(row.trl_endStk-row.trl_startStk+1))
#         trl_list.extend([row.trl_num]*(row.trl_endStk-row.trl_startStk+1))
#
#     on_idx=pd.Series(data=on_rgs,index=subrgs,name='on_idx')
#     stim_idx=pd.Series(data=stim_list,index=subrgs,name='stim_idx')
#     trl_idx=pd.Series(data=trl_list,index=subrgs,name='stim_idx')
#
#     sub_dff = None
#     sub_rawF = None
#     if dff is not None:
#         sub_dff=dff.loc[subrgs,:]
#     elif rawF is not None:
#         sub_rawF=rawF.loc[subrgs,:]
#     else:
#         print('Neither dff nor rawF is received.')
#
#     return {'on_idx':on_idx,'stim_idx':stim_idx,'trl_idx':trl_idx},{'sub_dff':sub_dff,'sub_rawF':sub_rawF}
#
#
# def getCentroids(roimask):
#     pix_size = 0.325
#     z_gap = 8
#     allkeys=list(roimask.keys())
#     d1,d2=roimask[allkeys[0]].item().todense().shape
#     Coor = np.matrix([np.outer(np.ones(d2), np.arange(d1)).ravel(),
#                               np.outer(np.arange(d2), np.ones(d1)).ravel()], dtype='float64')
#     cm_dict={}
#     for keyv in allkeys:
#         a=roimask[keyv].item().todense().T.flatten()
#         cm = (Coor * a.T / a.T.sum(axis=0)).T
#         cm_dict[keyv]=tuple(np.ravel(cm))
#
#     col_names=[]
#     xs=[]
#     ys=[]
#     zs=[]
#     for key,value in cm_dict.items():
#         # col_names.append(key)
#         slc,roi=key.split('_')
#         roi_num="{0:0=3d}".format(int(re.findall('\d+', roi)[0])) #not original version
#         col_names.append(slc+'_roi'+roi_num) #not original version
#         xs.append(value[0])
#         ys.append(value[1])
#         # zs.append(int(key[3:5]))
#         zs.append(int(re.findall('\d+', key.split('_')[0])[0])) # in case slice1, slice2
#
#     xs=np.asarray(xs)
#     ys=np.asarray(ys)
#     zs=np.asarray(zs)
#     zs=np.abs(zs-np.amax(zs))
#     centroids=pd.DataFrame(index=col_names,data={'x':xs*pix_size,'y':ys*pix_size,'z':zs*z_gap})#.reindex_like()
#     return centroids
#
#
class Datas(pd.Series):
    # def __init__(self,datas):
    #     # self.datas = datas
    #     # pd.Series.__init__(self,datas)
    #     pd.Series(datas.tolist(),index=datas.index)
    #     super().__init__(self)

    @property
    def _constructor(self):
        return Datas



    def align_dstim(self, on_rg):
        ids = pd.IndexSlice  # for multi-indexing
        mi_dffs = self.map(lambda d: d.getMiDff(use_stim_level=True).loc[ids[:, on_rg], :])

        mi_dff_list = []
        for dname, mi_dff in mi_dffs.iteritems():
            mi_dff_list.append(pd.concat([mi_dff], keys=[dname], names=['dname'], axis=1))

        x = pd.concat(mi_dff_list, axis=1)
        return x

    def get_info(self):
        datas_info = self.index.str.extract(r'(?P<line>\w{2})_(?P<exp>\d{2})_(?P<region>\w{2})').set_index(self.index)
        datas_info['line_region'] = datas_info['line'] + '_' + datas_info['region']
        datas_info['line_exp'] = datas_info['line'] + '_' + datas_info['exp']
        return datas_info.sort_values(by=['line_region','exp'])

    def as_series(self):
        return pd.Series(self)


def parse_dname(dname):
    line, exp, region = dname.split('_')
    return dict(line=line, exp=exp, region=region,
         line_region='_'.join([line, region]),
         line_exp='_'.join([line, exp]))


def get_line_region(dname):
    line, exp, region = dname.split('_')
    return '_'.join([line,region])


def line_region_to_int(dname):
    return line_region_order_dict[get_line_region(dname)]















