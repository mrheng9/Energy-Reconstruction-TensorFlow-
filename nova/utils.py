import pandas as pd
import numpy as np
from functools import reduce
import time

mkeys = ['run','subrun','cycle','evt','subevt']
def get_dataframe(file,keys,group, indices=[]):
    _dflist = []
    t1 = time.time()
    _dict = {}
    for key in keys:
        _dict[key] = file[group].get(key)[...].flatten()
    #print ("---> ", time.time()-t1)
    return pd.DataFrame(_dict)

def get_prongdata_index(file):
    png_keys   = ['isprimary','label3d','rece']
    prim_keys  = ['p.E','pdg','rec.mc.nu.prim_idx'] 
    train_keys = ['interaction']
    cont_keys = ['distallpngtop', 'distallpngbottom', 'distallpngwest','distallpngeast','distallpngback','distallpngfront']
    veto_keys = ['keep']

    t1 = time.time()
    df_png   = get_dataframe(file,mkeys+png_keys,'rec.vtx.elastic.fuzzyk.png.prongtrainingdata')
    t2 = time.time()
    df_prim  = get_dataframe(file,mkeys+prim_keys,'rec.mc.nu.prim')
    t3 = time.time()
    df_train = get_dataframe(file,mkeys+train_keys,'rec.training.trainingdata')
    t4 = time.time()
    df_png['pngtrainidx'] = df_png.index
    t5 = time.time()
    df_cont = get_dataframe(file,mkeys+cont_keys,'rec.sel.nuecosrej')
    df_veto = get_dataframe(file,mkeys+veto_keys,'rec.sel.veto')

    
    df_png   = df_png[(df_png['isprimary']==1)&(df_png['label3d']==0)]
    t6 = time.time()
    df_prim  = df_prim[(np.abs(df_prim['pdg'])==11)&(df_prim['rec.mc.nu.prim_idx']==0)]
    t7 = time.time()
    df_train = df_train[(df_train['interaction']>=4)&(df_train['interaction']<=7)]
    t8 = time.time()
    df_cont = df_cont[(df_cont['distallpngtop']>63)&\
                      (df_cont['distallpngbottom']>12)&\
                      (df_cont['distallpngeast']>12)&\
                      (df_cont['distallpngwest']>12)&\
                      (df_cont['distallpngfront']>18)&\
                      (df_cont['distallpngback']>18)]

    df_veto = df_veto[df_veto['keep']==1]
    
    df_list = [df_png, df_prim, df_train, df_cont, df_veto]
    #df_list = [df_png, df_prim, df_train]
    df = reduce(lambda left,right: pd.merge(left,right,on=mkeys), df_list)
    t9 = time.time()

    dd = df[df['rece'] == df.groupby(mkeys)['rece'].transform(max)]
    t10 = time.time()
    #print ('{:.2} {:.2} {:.2} {:.2} {:.2} {:.2} {:.2} {:.2} {:.2} : {:.2}'.format(t2-t1, t3-t2, t4-t3, t5-t4, t6-t5, t7-t6, t8-t7, t9-t8, t10-t9, t10-t1))
    return dd
