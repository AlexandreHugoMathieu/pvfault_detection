# SPDX-FileCopyrightText: Copyright © Heliocity SAS 2020. All rights reserved. Tous droits réservés
# SPDX-License-Identifier: UNLICENSED


"""

1) read in a real source installation that has previously been analysed
2) define desired output configuration
3) insert fault signals
4) generate noise on copied data to obfuscate fake arrays
5) export to pvdata Excel file
6) by hand: generate fake configuration file for generated dataset

"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mpc

import numpy as np
import pandas as pd

mpl.use('qt5agg')

from app.load_io import load_io


def insert_frozen_data(io, plot=False):
    # données figées
    # 1) add flag figées : random start time. end time defined as random propagation of incident
    # 2) set data to nan during frozen state
    # 3) fill forward
    nsample = int(0.001*io.df.shape[0])
    fake_frozen = io.df.sample(n=nsample)
    if plot:
        plt.plot(io.df[f"Gh"],label="initial dataset")
        plt.bar(fake_frozen.index-pd.DateOffset(minutes=60*io.df.tstep.hour/2),height=1000,width=io.df.tstep.hour/24*4,color=[1,0,0,0.75],label="start of outage")
    while np.floor(nsample)>0:
        df_shift = io.df.shift(1)
        io.df.loc[io.df.index.isin(fake_frozen.index.to_list())] = df_shift.loc[io.df.index.isin(fake_frozen.index.to_list())]
        fake_frozen.index = fake_frozen.index + pd.DateOffset(minutes=60*io.df.tstep.hour)
        nsample*=0.9
        fake_frozen = fake_frozen.sample(n=int(np.floor(nsample)))
        print(nsample,fake_frozen.shape)
        # plt.plot(io.df[f"Gh"])
    
    if plot:
        plt.plot(io.df[f"Gh"],color='0',linewidth=1,label="frozen states added")
        plt.legend()
        print(fake_frozen)
        plt.show()
    
    return io

def insert_duplicate_data_point(io,src_pt = "mpp-1-1",delta_Npv=7, snratio=0, plot=False):
    # add new point derived from mpp-1-1, and apply noise function
    pt = src_pt
    io.stc[f'Npv_per_str@mpp-3-1'] = io.stc[f'Npv_per_str@{pt}'] + delta_Npv
    snratio = 0.00  # do not apply noise now to avoid breaking ratio rules. we can do so at end
    noise = (snratio) * np.random.normal(-1, 1, io.df.shape[0])
    print(io.stc[f'Npv_per_str@{pt}'])
    io.df[f"Idc@mpp-3-1"] = io.df[f"Idc@mpp-1-1"]
    io.df[f"Idc@mpp-3-1"] += io.df[f"Idc@mpp-3-1"] * noise
    io.df[f"Vdc@mpp-3-1"] = io.df[f"Vdc@mpp-1-1"] / io.stc[f'Npv_per_str@{pt}'] * io.stc[f'Npv_per_str@mpp-3-1']
    io.df[f"Vdc@mpp-3-1"] += io.df[f"Vdc@mpp-3-1"] * noise
    io.df[f"Pdc@mpp-3-1"] = io.df[f"Vdc@mpp-3-1"] * io.df[f"Idc@mpp-3-1"]
    io.df[f"Pac@mpp-3-1"] = io.df[f"Pac@mpp-1-1"] / io.stc[f'Npv_per_str@{pt}'] * io.stc[f'Npv_per_str@mpp-3-1']
    io.df[f"Pac@mpp-3-1"] += io.df[f"Pac@mpp-3-1"] * noise
    io.df[f"Iac1@inv-3"] = io.df[f"Iac1@inv-1"] / io.stc[f'Npv_per_str@{pt}'] * io.stc[f'Npv_per_str@mpp-3-1']
    io.df[f"Iac1@inv-3"] += io.df[f"Iac1@inv-3"] * noise
    io.df[f"Vac1@inv-3"] = io.df[f"Pac@mpp-3-1"] / io.df[f"Iac1@inv-3"]
    print(io.df.filter(like="Vac").columns)
    print(io.df.filter(like="Iac").columns)
    
    if plot:
        plt.subplots(1,4,sharex=True)
        plt.subplot(1,4,1)
        plt.title("Pdc")
        plt.plot(io.df[f"Pdc@mpp-1-1"])
        plt.plot(io.df[f"Pdc@mpp-3-1"])
        plt.subplot(1,4,2)
        plt.title("Pac")
        plt.plot(io.df[f"Pac@mpp-1-1"])
        plt.plot(io.df[f"Pac@mpp-3-1"])
        plt.subplot(1,4,3)
        plt.title("Pac/Pdc")
        plt.plot(io.df[f"Pac@mpp-1-1"]/io.df[f"Pdc@mpp-1-1"],"o")
        plt.plot(io.df[f"Pac@mpp-3-1"]/io.df[f"Pdc@mpp-3-1"],"o")
        plt.subplot(1,4,4)
        plt.title("Pac/Pdc")
        plt.plot((io.df[f"Pac@mpp-1-1"]/io.df[f"Pdc@mpp-1-1"])/(io.df[f"Pac@mpp-3-1"]/io.df[f"Pdc@mpp-3-1"]),"o")
        plt.show()
    
    return io


def insert_shading(io,shade_azi_min = 0,shade_azi_max = 180,shade_alt = 50, plot=False,
                   new_pt_pdm = "mpp-3-1",
                   new_pt_inv = "inv-3",
                   ref_pt_pdm="mpp-1-1"):
    #shading
    # shade_azi_min = 0 # start of shading structure
    # shade_azi_max = 180 # end of shading structure
    # shade_alt = 50

    # shading emulator, flag and opacity
    io.df[f"shade"] = 0
    io.df.loc[(io.df[f"sol_azim"] > shade_azi_min) & (io.df[f"sol_azim"] < shade_azi_max) & (io.df[f"sol_alt"]<shade_alt),"shade"] = shade_alt
    ratio_Gi_Gd = io.df[f"Gi%1-0-0"] / io.df[f"Gid%1-0-0"]
    ratio_log_Gi_Gd = np.log(io.df[f"Gi%1-0-0"]) / np.log(io.df[f"Gid%1-0-0"])
    #todo: for a more accurate shading sim, use King model for Imp and Vmp
    #todo: simulate diode activation

    if plot:
        plt.plot(io.df[f"sol_azim"],io.df[f"sol_alt"],".")
        plt.plot(io.df[f"sol_azim"],io.df[f"shade"],".")
        plt.show()
        plt.plot(ratio_Gi_Gd)
        plt.plot(ratio_log_Gi_Gd)
        plt.show()
        plt.subplots(2,2,sharex=True)
        plt.suptitle("shading")
        plt.subplot(2,2,1)
        plt.title("Idc")
        plt.plot(io.df[f"Idc@{new_pt_pdm}"],label="no shading")
        plt.subplot(2,2,2)
        plt.title("Vdc")
        plt.plot(io.df[f"Vdc@{new_pt_pdm}"])
        plt.subplot(2,2,3)
        plt.title("Pdc")
        plt.plot(io.df[f"Pdc@{new_pt_pdm}"])
        plt.subplot(2,2,4)
        plt.title("Pac")
        plt.plot(io.df[f"Pac@{new_pt_pdm}"])
    
    # apply shading signal
    io.df.loc[(io.df[f"sol_azim"] > shade_azi_min) & (io.df[f"sol_azim"] < shade_azi_max) & (io.df[f"sol_alt"]<shade_alt),
              f"Idc@{new_pt_pdm}"] = io.df[f"Idc@{new_pt_pdm}"]/ratio_Gi_Gd
    io.df.loc[(io.df[f"sol_azim"] > shade_azi_min) & (io.df[f"sol_azim"] < shade_azi_max) & (io.df[f"sol_alt"]<shade_alt),
              f"Vdc@{new_pt_pdm}"] = io.df[f"Vdc@{new_pt_pdm}"]/np.power(ratio_log_Gi_Gd,0.5) # abtrary softening on effect by sqrt(2)
    io.df[f"Pdc@{new_pt_pdm}"] = io.df[f"Vdc@{new_pt_pdm}"] * io.df[f"Idc@{new_pt_pdm}"]
    io.df.loc[(io.df[f"sol_azim"] > shade_azi_min) & (io.df[f"sol_azim"] < shade_azi_max) & (io.df[f"sol_alt"]<shade_alt),
              f"Pac@{new_pt_pdm}"] = io.df[f"Pdc@{new_pt_pdm}"]*io.df[f"Pac@{ref_pt_pdm}"]/io.df[f"Pdc@{ref_pt_pdm}"]
              # "Pac@{new_pt_pdm}"] = io.df[f"Pdc@{new_pt_pdm}"]*io.stc[f'inv_efficiency@{pt}']
    io.df[f"Iac1@{new_pt_inv}"] = io.df[f"Pac@{new_pt_pdm}"] / io.df[f"Vac1@{new_pt_inv}"]
    
    if plot:
        # plt.subplots(2,2,sharex=True)
        plt.subplot(2,2,1)
        plt.title("Idc")
        plt.plot(io.df[f"Idc@{new_pt_pdm}"],label="fixed shading")
        plt.legend()
        plt.subplot(2,2,2)
        plt.title("Vdc")
        plt.plot(io.df[f"Vdc@{new_pt_pdm}"])
        plt.subplot(2,2,3)
        plt.title("Pdc")
        plt.plot(io.df[f"Pdc@{new_pt_pdm}"])
        plt.subplot(2,2,4)
        plt.title("Pac")
        plt.plot(io.df[f"Pac@{new_pt_pdm}"])
          
        plt.subplots(1,4,sharex=True)
        plt.suptitle("shading 1 vs 3")
        plt.subplot(1,4,1)
        plt.title("Pdc")
        plt.plot(io.df[f"Pdc@{ref_pt_pdm}"])
        plt.plot(io.df[f"Pdc@{new_pt_pdm}"])
        plt.subplot(1,4,2)
        plt.title("Pac")
        plt.plot(io.df[f"Pac@{ref_pt_pdm}"])
        plt.plot(io.df[f"Pac@{new_pt_pdm}"])
        plt.subplot(1,4,3)
        plt.title("Pac/Pdc")
        plt.plot(io.df[f"Pac@{ref_pt_pdm}"]/io.df[f"Pdc@{ref_pt_pdm}"],"o")
        plt.plot(io.df[f"Pac@{new_pt_pdm}"]/io.df[f"Pdc@{new_pt_pdm}"],"o")
        plt.subplot(1,4,4)
        plt.title("Pac/Pdc")
        plt.plot((io.df[f"Pac@{ref_pt_pdm}"]/io.df[f"Pdc@{ref_pt_pdm}"])/(io.df[f"Pac@{new_pt_pdm}"]/io.df[f"Pdc@{new_pt_pdm}"]),"o")
        plt.show()
    return io

def insert_peak_shaving(io,plot=False, Pac_max = 68000, new_pt_pdm = "mpp-3-1",new_pt_inv = "inv-3",ref_pt_pdm="mpp-1-1"):
    # peak shaving
    # Pac_max = 68000  # todo insert Pac limit into config for flag detection
    io.df[f"Pac@{new_pt_pdm}_noshaving"] = io.df[f"Pac@{new_pt_pdm}"]
    io.df.loc[io.df[f"Pac@{new_pt_pdm}"]>=Pac_max,"Pac@{new_pt_pdm}"] = Pac_max  # warning: subsequent loc affected by Pac limit
    peak_shaving_loss_factor = io.df[f"Pac@{new_pt_pdm}_noshaving"]/io.df[f"Pac@{new_pt_pdm}"]
    io.df.loc[io.df[f"Pac@{new_pt_pdm}"]>=Pac_max,"Pdc@{new_pt_pdm}"] = io.df[f"Pac@{new_pt_pdm}"]/io.stc[f'inv_efficiency@{ref_pt_pdm}']
    io.df[f"Iac1@inv-3"] = io.df[f"Pac@{new_pt_pdm}"] / io.df[f"Vac1@inv-3"]
    
    # approximate Idc,Vdc variation due to peak shaving:
    # reduce Idc by loss_factor
    io.df.loc[io.df[f"Pac@{new_pt_pdm}"]>=Pac_max,"Idc@{new_pt_pdm}"] = io.df[f"Idc@{new_pt_pdm}"]/peak_shaving_loss_factor
    io.df.loc[io.df[f"Pac@{new_pt_pdm}"]>=Pac_max,"Vdc@{new_pt_pdm}"] = io.df[f"Pdc@{new_pt_pdm}"]/io.df[f"Idc@{new_pt_pdm}"]
    
    # todo: use IV curve to estimate new operating point

    if plot:
        plt.subplots(2,2,sharex=True)
        plt.subplot(2,2,1)
        plt.title("Idc")
        plt.plot(io.df[f"Idc@{new_pt_pdm}"],label="Pac peak shaving")
        plt.legend()
        plt.subplot(2,2,2)
        plt.title("Vdc")
        plt.plot(io.df[f"Vdc@{new_pt_pdm}"])
        plt.subplot(2,2,3)
        plt.title("Pdc")
        plt.plot(io.df[f"Pdc@{new_pt_pdm}"])
        plt.subplot(2,2,4)
        plt.title("Pac")
        plt.plot(io.df[f"Pac@{new_pt_pdm}"])
        
        plt.subplots(2,2,sharex=True)
        plt.subplot(2,2,1)
        plt.title("Idc")
        plt.plot(io.df[f"Idc@{ref_pt_pdm}"])
        plt.plot(io.df[f"Idc@{new_pt_pdm}"])
        plt.subplot(2,2,2)
        plt.title("Vdc")
        plt.plot(io.df[f"Vdc@{ref_pt_pdm}"])
        plt.plot(io.df[f"Vdc@{new_pt_pdm}"])
        plt.subplot(2,2,3)
        plt.title("Pdc")
        plt.plot(io.df[f"Pdc@{ref_pt_pdm}"])
        plt.plot(io.df[f"Pdc@{new_pt_pdm}"])
        plt.subplot(2,2,4)
        plt.title("Pac")
        plt.plot(io.df[f"Pac@{ref_pt_pdm}"])
        plt.plot(io.df[f"Pac@{new_pt_pdm}"])
        plt.show()
    return io

def insert_noise(df,snratio = 0.001):
    #apply common noise signal to all data (must be very light)
    # snratio = 0.001
    # warning : should be numeric columns only
    noise = (snratio) * np.random.normal(-1, 1, io.df.shape[0])
    for col in df.columns:
        df[col] += df[col]*noise
    return df

def insert_shutdown(df,shutdown_start = "2018-04-26",shutdown_end = "2018-05-05", plot=False):
    #arret
    # shutdown_start = "2018-04-26"
    # shutdown_end = "2018-05-05"
    df3.loc[shutdown_start:shutdown_end] = 0
    if plot:
        plt.plot(df3)
        plt.legend(df3.columns)
        plt.show()
    return df

if __name__ == '__main__':
    """
    heliosite
    address 
    heliopolis
    date de mise en service 14/09/2011

    new pt list
    pt1 = mpp-1-1 danival 1
    pt2 = mpp-2-1 danival 2
    pt3 = mpp-1-1 avec chaine plus long + données figées + ombrage + ecretage + arret

    """
    repo_id = "danival_new3"
    io = load_io(
        scenario_name="scenario_flash",
        repo_id=repo_id,
        restore_step=16 # None to reload last chain step of the scenario
    )

    # undo timezone correction
    # io.df = DatetimeFormatter(io=io).set_tz(io.df, srce_tz="Europe/Paris", tz='Etc/GMT-2').tz_localize(None)
    # io.df = io.df['2020-07-08':'2020-07-10']

    io = insert_frozen_data(io, plot=False)
    io = insert_duplicate_data_point(io, src_pt="mpp-1-1", delta_Npv=7, snratio=0, plot=False)
    io = insert_shading(io,shade_azi_min = 0,shade_azi_max = 180,shade_alt = 50, plot=False, new_pt_pdm = "mpp-3-1",new_pt_inv = "inv-3",ref_pt_pdm="mpp-1-1")
    io = insert_peak_shaving(io,plot=False, Pac_max = 68000, new_pt_pdm = "mpp-3-1",new_pt_inv = "inv-3",ref_pt_pdm="mpp-1-1")

    df3 = io.df.filter(like="mpp-3-1")
    print(df3.shape)
    dfac = io.df.filter(like="ac1@inv-3")
    df3 = pd.concat([df3, dfac], axis=1)

    insert_noise(df3, snratio=0.001)
    insert_shutdown(df3, shutdown_start="2018-04-26", shutdown_end="2018-05-05", plot=False)

    plt.subplots(1,4,sharex=True)
    plt.subplot(1,4,1)
    plt.title("Pdc")
    plt.plot(io.df[f"Pdc@mpp-1-1"])
    plt.plot(io.df[f"Pdc@mpp-3-1"])
    plt.subplot(1,4,2)
    plt.title("Pac")
    plt.plot(io.df[f"Pac@mpp-1-1"])
    plt.plot(io.df[f"Pac@mpp-3-1"])
    plt.subplot(1,4,3)
    plt.title("Pac/Pdc")
    plt.plot(io.df[f"Pac@mpp-1-1"]/io.df[f"Pdc@mpp-1-1"],"o")
    plt.plot(io.df[f"Pac@mpp-3-1"]/io.df[f"Pdc@mpp-3-1"],"o")
    plt.subplot(1,4,4)
    plt.title("Pac/Pdc")
    plt.plot((io.df[f"Pac@mpp-1-1"]/io.df[f"Pdc@mpp-1-1"])/(io.df[f"Pac@mpp-3-1"]/io.df[f"Pdc@mpp-3-1"]),"o")
    plt.show()


    # export Excel pvdata
    print(df3.shape)
    datapath = "C:\Datasets\helioflash_test\heliosite\input"
    # dfoutput = df3.to_excel(datapath+"\heliosite_mpp-3-1.xlsx")

    # dfPdc = io.df.filter(regex="^Pdc.*")
    # dfPac = io.df.filter(regex="^Pac.*")
    # dfIac = io.df.filter(regex="^Iac.*")
    # dfIdc = io.df.filter(regex="^Idc.*")
    # dfVac = io.df.filter(regex="^Vac.*")
    # dfVdc = io.df.filter(regex="^Vdc.*")
