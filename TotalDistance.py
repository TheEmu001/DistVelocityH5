# Importing the toolbox (takes several seconds)
import warnings

import pandas as pd
from pathlib import Path
import numpy as np
import os
import matplotlib.pyplot as plt
import time_in_each_roi
from scipy import stats
from textwrap import wrap
from scipy import integrate

warnings.filterwarnings('ignore')

all_data = pd.DataFrame(columns=['init'])
list_no = np.arange(0.0, 108060.0, 1.0) #number of frames in 30 minutes
# list_no = np.arange(0.0, 180000, 1.0) #number of frames in 50 minutes
ms_time = np.arange(0.0, 2670.0, 0.4) #1ms increments of time
# frame rate of camera in those experiments
start_frame = 59 #frame to start at
pick_frame = 30 # pick ever __th frame

fps = 60
no_seconds = 15
moving_average_duration_frames = fps * no_seconds
updated_window = no_seconds/(pick_frame/fps)
updated_window = int(updated_window)

DLCscorer = 'DLC_resnet50_BigBinTopSep17shuffle1_250000'

def velocity(video, color, label):

    dataname = str(Path(video).stem) + DLCscorer + '.h5'
    print(dataname)

    #loading output of DLC
    Dataframe = pd.read_hdf(os.path.join(dataname), errors='ignore')
    # Dataframe.reset_index(drop=True)

    #you can read out the header to get body part names!
    bodyparts=Dataframe.columns.get_level_values(1)

    bodyparts2plot=bodyparts

    # let's calculate velocity of the back
    # this can be changed to whatever body part
    bpt='head'
    vel = time_in_each_roi.calc_distance_between_points_in_a_vector_2d(np.vstack([Dataframe[DLCscorer][bpt]['x'].values.flatten(), Dataframe[DLCscorer][bpt]['y'].values.flatten()]).T)


    time=np.arange(len(vel))*1./fps #time that is 1/60 sec

    #notice the units of vel are relative pixel distance [per time step]
    vel=vel*.03924
    velocity_pd = pd.Series(vel)
    # vel_avg = velocity_pd.rolling(moving_average_duration_frames).mean()

    vel_avg = velocity_pd
    inst_dist = vel*1./fps
    dist_sum = inst_dist.cumsum()

    # taking every 6th row
    every_6th = vel_avg.iloc[start_frame::pick_frame]
    nth_time = np.arange(len(every_6th))*.1

    # plt.plot(time[61:], vel_avg[61:], color=color, label=label) #instantaneous rolling average velocity plot raw
    # plt.plot(time, dist_sum, color=color) #cummulative distance plot

    every_6th_avg = every_6th.rolling(updated_window).mean()
    # plt.plot(nth_time[61:], every_6th_avg[61:], color=color, label=label)
    new_df = every_6th_avg.copy()
    all_data[video+"_dist"] = dist_sum[:108060]
    # all_data[video+"_vel"] = vel_avg[:108060]
    all_data[video + "_vel"] = new_df[:108060]
    # print(every_6th)


if __name__ == '__main__':
    # all_data['time'] = (list_no * (1 / 60)) / 60

    # """
    # Saline
    # """
    # velocity(video='Saline_Ai14_OPRK1_C2_F0_Top Down', color='pink', label='F0 Saline')
    # velocity(video='Saline_Ai14_OPRK1_C2_F1_Top Down', color='pink', label='F1 Saline')
    # velocity(video='Saline_Ai14_OPRK1_C1_F2_Top Down', color='pink', label='F2 Saline')
    #
    # velocity(video='Saline_Ai14_OPRK1_C1_M1_Top Down', color='pink', label='M1 Saline')
    # velocity(video='Saline_Ai14_OPRK1_C1_M2_Top Down', color='pink', label='M2 Saline')
    # velocity(video='Saline_Ai14_OPRK1_C1_M3_Top Down', color='pink', label='M3 Saline')
    # velocity(video='Saline_Ai14_OPRK1_C1_M4_Top Down', color='pink', label='M4 Saline')
    #
    # # saline_avg_dist = all_data.loc[:,
    # #            ['Saline_Ai14_OPRK1_C2_F0_Top Down_dist',
    # #             'Saline_Ai14_OPRK1_C2_F1_Top Down_dist',
    # #             'Saline_Ai14_OPRK1_C1_F2_Top Down_dist',
    # #             'Saline_Ai14_OPRK1_C1_M1_Top Down_dist',
    # #             'Saline_Ai14_OPRK1_C1_M2_Top Down_dist',
    # #             'Saline_Ai14_OPRK1_C1_M3_Top Down_dist',
    # #             'Saline_Ai14_OPRK1_C1_M4_Top Down_dist']]
    # # all_data['Avg Dist Saline'] = saline_avg_dist.mean(axis=1)
    # # all_data['Avg Dist Saline SEM'] = stats.sem(saline_avg_dist, axis=1)
    # #
    # # plt.plot(all_data['time'], all_data['Avg Dist Saline'], color='pink', linewidth=1,
    # #          label='Average Distance Saline')
    #
    #
    # saline_avg_vel = all_data.loc[:,
    #            ['Saline_Ai14_OPRK1_C2_F0_Top Down_vel',
    #             'Saline_Ai14_OPRK1_C2_F1_Top Down_vel',
    #             'Saline_Ai14_OPRK1_C1_F2_Top Down_vel',
    #             'Saline_Ai14_OPRK1_C1_M1_Top Down_vel',
    #             'Saline_Ai14_OPRK1_C1_M2_Top Down_vel',
    #             'Saline_Ai14_OPRK1_C1_M3_Top Down_vel',
    #             'Saline_Ai14_OPRK1_C1_M4_Top Down_vel']]
    #
    # # all_data['Avg Vel Saline'] = saline_avg_vel.mean(axis=1)
    # # all_data['Avg Vel Saline SEM'] = stats.sem(saline_avg_vel, axis=1)
    # # all_data['time'] = (list_no * (1 / 60)) / 60
    # # # plt.plot(all_data['time'], all_data['Avg Vel Saline'], color='black', linewidth=1,
    # # #          label='Average Velocity Saline')
    #
    # vel_only_df_sal = saline_avg_vel.mean(axis=1)
    # vel_only_df_sal = vel_only_df_sal.dropna()
    # vel_only_df_sal_sem = stats.sem(saline_avg_vel, axis=1)
    # index_vals_sal = vel_only_df_sal.index
    # better_time_sal = index_vals_sal * (1. / fps) / 60
    # plt.plot(better_time_sal, vel_only_df_sal, color='black', linewidth=1,
    #          label='Average Velocity Saline')
    #
    #
    #
    #
    # #
    # """
    # U50
    # """
    # velocity(video='U50_Ai14_OPRK1_C2_F0_Top Down', color='red', label='F0 Saline+5mgkg U50')
    # velocity(video='U50_Ai14_OPRK1_C1_F1_Top Down', color='orange', label='F1 Saline+5mgkg U50')
    # velocity(video='U50_Ai14_OPRK1_C2_F2_Top Down', color='purple', label='F2 Saline+5mgkg U50')
    #
    # velocity(video='U50_Ai14_OPRK1_C1_M1_Top Down', color='pink', label='M1 Saline+5mgkg U50')
    # velocity(video='U50_Ai14_OPRK1_C1_M2_Top Down', color='black', label='M2 Saline+5mgkg U50')
    # velocity(video='U50_Ai14_OPRK1_C1_M3_Top Down', color='green', label='M3 Saline+5mgkg U50')
    # velocity(video='U50_Ai14_OPRK1_C1_M4_Top Down', color='silver', label='M4 Saline+5mgkg U50')
    #
    # # u50_avg_dist = all_data.loc[:,
    # #            ['U50_Ai14_OPRK1_C2_F0_Top Down_dist',
    # #             'U50_Ai14_OPRK1_C1_F1_Top Down_dist',
    # #             'U50_Ai14_OPRK1_C2_F2_Top Down_dist',
    # #             'U50_Ai14_OPRK1_C1_M1_Top Down_dist',
    # #             'U50_Ai14_OPRK1_C1_M2_Top Down_dist',
    # #             'U50_Ai14_OPRK1_C1_M3_Top Down_dist',
    # #             'U50_Ai14_OPRK1_C1_M4_Top Down_dist']]
    # # all_data['Avg Dist U50'] = u50_avg_dist.mean(axis=1)
    # # all_data['Avg Dist U50 SEM'] = stats.sem(u50_avg_dist, axis=1)
    # #
    # # plt.plot(all_data['time'], all_data['Avg Dist U50'], color='orange', linewidth=1,
    # #          label='Average Distance U50')
    #
    # u50_avg_vel = all_data.loc[:,
    #            ['U50_Ai14_OPRK1_C2_F0_Top Down_vel',
    #             'U50_Ai14_OPRK1_C1_F1_Top Down_vel',
    #             'U50_Ai14_OPRK1_C2_F2_Top Down_vel',
    #             'U50_Ai14_OPRK1_C1_M1_Top Down_vel',
    #             'U50_Ai14_OPRK1_C1_M2_Top Down_vel',
    #             'U50_Ai14_OPRK1_C1_M3_Top Down_vel',
    #             'U50_Ai14_OPRK1_C1_M4_Top Down_vel']]
    # all_data['Avg Vel U50'] = u50_avg_vel.mean(axis=1)
    # all_data['Avg Vel U50 SEM'] = stats.sem(u50_avg_vel, axis=1)
    #
    # # plt.plot(all_data['time'], all_data['Avg Vel U50'], color='orange', linewidth=1,
    # #          label='Average Vel U50')
    #
    # vel_only_df_u50 = u50_avg_vel.mean(axis=1)
    # vel_only_df_u50 = vel_only_df_u50.dropna()
    # vel_only_df_u50_sem = stats.sem(u50_avg_vel, axis=1)
    # index_vals_u50 =  vel_only_df_u50.index
    # better_time_u50 = index_vals_u50 * (1. / fps) / 60
    # plt.plot(better_time_u50, vel_only_df_u50, color='orange', linewidth=1,
    #          label='Average Velocity U50')
    #
    # #
    # """
    # Naltrexone U50
    # """
    # velocity(video='Naltr_U50_Ai14_OPRK1_C2_F0_Top Down', color='red', label='F0 3mgkg Nalt+5mgkg U50')
    # velocity(video='Nalt_U50_Ai14_OPRK1_C1_F1_Top Down', color='orange', label='F1 3mgkg Nalt+5mgkg U50')
    # velocity(video='Nalt_U50_Ai14_OPRK1_C1_F2_Top Down', color='purple', label='F2 3mgkg Nalt+5mgkg U50')
    #
    # velocity(video='Nalt_U50_Ai14_OPRK1_C1_M1_Top Down', color='pink', label='M1 3mgkg Nalt+5mgkg U50')
    # velocity(video='Nalt_U50_Ai14_OPRK1_C1_M2_Top Down', color='black', label='M2 3mgkg Nalt+5mgkg U50')
    # velocity(video='Nalt_U50_Ai14_OPRK1_C1_M3_Top Down', color='green', label='M3 3mgkg Nalt+5mgkg U50')
    # velocity(video='Nalt_U50_Ai14_OPRK1_C1_M4_Top Down', color='silver', label='M4 3mgkg Nalt+5mgkg U50')
    #
    # # nalt_avg_dist = all_data.loc[:,
    # #            ['Naltr_U50_Ai14_OPRK1_C2_F0_Top Down_dist',
    # #             'Nalt_U50_Ai14_OPRK1_C1_F1_Top Down_dist',
    # #             'Nalt_U50_Ai14_OPRK1_C1_F2_Top Down_dist',
    # #             'Nalt_U50_Ai14_OPRK1_C1_M1_Top Down_dist',
    # #             'Nalt_U50_Ai14_OPRK1_C1_M2_Top Down_dist',
    # #             'Nalt_U50_Ai14_OPRK1_C1_M3_Top Down_dist',
    # #             'Nalt_U50_Ai14_OPRK1_C1_M4_Top Down_dist']]
    # # all_data['Avg Dist Nalt'] = nalt_avg_dist.mean(axis=1)
    # # all_data['Avg Dist Nalt SEM'] = stats.sem(nalt_avg_dist, axis=1)
    # #
    # # plt.plot(all_data['time'], all_data['Avg Dist Nalt'], color='red', linewidth=1,
    # #          label='Average Distance Nalt')
    #
    # nalt_avg_vel = all_data.loc[:,
    #            [
    #             'Naltr_U50_Ai14_OPRK1_C2_F0_Top Down_vel',
    #             'Nalt_U50_Ai14_OPRK1_C1_F1_Top Down_vel',
    #             'Nalt_U50_Ai14_OPRK1_C1_F2_Top Down_vel',
    #             'Nalt_U50_Ai14_OPRK1_C1_M1_Top Down_vel',
    #             'Nalt_U50_Ai14_OPRK1_C1_M2_Top Down_vel',
    #             'Nalt_U50_Ai14_OPRK1_C1_M3_Top Down_vel',
    #             'Nalt_U50_Ai14_OPRK1_C1_M4_Top Down_vel'
    #             ]]
    #
    #
    # all_data['Avg Vel Nalt'] = nalt_avg_vel.mean(axis=1)
    # all_data['Avg Vel Nalt SEM'] = stats.sem(nalt_avg_vel, axis=1)
    #
    # # plt.plot(all_data['time'], all_data['Avg Vel Nalt'], color='red', linewidth=1,
    # #          label='Average Velocity Nalt')
    #
    # vel_only_df_nal = nalt_avg_vel.mean(axis=1)
    # vel_only_df_nal = vel_only_df_nal.dropna()
    # vel_only_df_nal_sem = stats.sem(nalt_avg_vel, axis=1)
    # index_vals_nal =  vel_only_df_nal.index
    # better_time_nal = index_vals_nal * (1. / fps) / 60
    # plt.plot(better_time_nal, vel_only_df_nal, color='red', linewidth=1,
    #          label='Average Velocity Nalt')

    """
    NORBNI U50
    """
    velocity(video='NORBNI_U50_Ai14_OPRK1_C2_F0_Top Down', color='red', label='F0 10mgkg NORBNI+5mgkg U50')
    velocity(video='NORBNI_U50_Ai14_OPRK1_C2_F1_Top Down', color='orange', label='F1 10mgkg NORBNI+5mgkg U50')
    velocity(video='NORBNI_U50_Ai14_OPRK1_C2_F2_Top Down', color='purple', label='F2 10mgkg NORBNI+5mgkg U50')

    velocity(video='NORBNI_U50_Ai14_OPRK1_C1_M1_Top Down', color='pink', label='M1 10mgkg NORBNI+5mgkg U50')
    velocity(video='NORBNI_U50_Ai14_OPRK1_C1_M2_Top Down', color='black', label='M2 10mgkg NORBNI+5mgkg U50')
    velocity(video='NORBNI_U50_Ai14_OPRK1_C1_M3_Top Down', color='green', label='M3 10mgkg NORBNI+5mgkg U50')
    velocity(video='NORBNI_U50_Ai14_OPRK1_C1_M4_Top Down', color='silver', label='M4 10mgkg NORBNI+5mgkg U50')
    # norbni_u50_avg_dist = all_data.loc[:,
    #            ['NORBNI_U50_Ai14_OPRK1_C2_F0_Top Down_dist',
    #             'NORBNI_U50_Ai14_OPRK1_C2_F1_Top Down_dist',
    #             'NORBNI_U50_Ai14_OPRK1_C2_F2_Top Down_dist',
    #             'NORBNI_U50_Ai14_OPRK1_C1_M1_Top Down_dist',
    #             'NORBNI_U50_Ai14_OPRK1_C1_M2_Top Down_dist',
    #             'NORBNI_U50_Ai14_OPRK1_C1_M3_Top Down_dist',
    #             'NORBNI_U50_Ai14_OPRK1_C1_M4_Top Down_dist']]
    # all_data['Avg Dist NORBNI+U50'] = norbni_u50_avg_dist.mean(axis=1)
    # all_data['Avg Dist NORBNI+U50 SEM'] = stats.sem(norbni_u50_avg_dist, axis=1)
    #
    # plt.plot(all_data['time'], all_data['Avg Dist NORBNI+U50'], color='blue', linewidth=1,
    #          label='Average Distance NORBNI+U50')

    norbni_u50_avg_vel = all_data.loc[:,
               [
                'NORBNI_U50_Ai14_OPRK1_C2_F0_Top Down_vel',
                'NORBNI_U50_Ai14_OPRK1_C2_F1_Top Down_vel',
                'NORBNI_U50_Ai14_OPRK1_C2_F2_Top Down_vel',
                'NORBNI_U50_Ai14_OPRK1_C1_M1_Top Down_vel',
                'NORBNI_U50_Ai14_OPRK1_C1_M2_Top Down_vel',
                'NORBNI_U50_Ai14_OPRK1_C1_M3_Top Down_vel',
                'NORBNI_U50_Ai14_OPRK1_C1_M4_Top Down_vel'
                ]]
    all_data['Avg Vel NORBNI+U50'] = norbni_u50_avg_vel.mean(axis=1)
    all_data['Avg Vel NORBNI+U50 SEM'] = stats.sem(norbni_u50_avg_vel, axis=1)


    vel_only_df_nu = norbni_u50_avg_vel.mean(axis=1)
    vel_only_df_nu = vel_only_df_nu.dropna()
    norbni_u50_avg_vel.dropna()
    vel_only_df_nu_sem = stats.sem(vel_only_df_nu)
    index_vals = vel_only_df_nu.index
    better_time_nu = index_vals*(1./fps)/60
    plt.plot(better_time_nu, vel_only_df_nu, color='blue', linewidth=1,
             label='Average Velocity NORBNI+U50')

    # all_data['Avg Vel NORBNI+Saline'] = norbni_saline_avg_vel.mean(axis=1)
    # plt.plot(all_data['time'], all_data['Avg Vel NORBNI+U50'], color='blue', linewidth=1,
    #          label='Average Vel NORBNI+U50')

    """
    # NORBNI + Saline
    # """
    velocity(video='NORBNI_Saline_Ai14_OPRK1_C2_F0_Top Down', color='red', label='F0 10mgkg NORBNI+Saline')
    velocity(video='NORBNI_Saline_Ai14_OPRK1_C2_F1_Top Down', color='blue', label='F1 10mgkg NORBNI+Saline')
    velocity(video='NORBNI_Saline_Ai14_OPRK1_C2_F2_Top Down', color='green', label='F2 10mgkg NORBNI+Saline')

    velocity(video='NORBNI_Saline_Ai14_OPRK1_C1_M1_Top Down', color='orange', label='M1 10mgkg NORBNI+Saline')
    velocity(video='NORBNI_Saline_Ai14_OPRK1_C1_M2_Top Down', color='brown', label='M2 10mgkg NORBNI+Saline')
    velocity(video='NORBNI_Saline_Ai14_OPRK1_C1_M3_Top Down', color='black', label='M3 10mgkg NORBNI+Saline')
    velocity(video='NORBNI_Saline_Ai14_OPRK1_C1_M4_Top Down', color='purple', label='M4 10mgkg NORBNI+Saline')
    #
    # # norbni_saline_avg_dist = all_data.loc[:,
    # #            ['NORBNI_Saline_Ai14_OPRK1_C2_F0_Top Down_dist',
    # #             'NORBNI_Saline_Ai14_OPRK1_C2_F1_Top Down_dist',
    # #             'NORBNI_Saline_Ai14_OPRK1_C2_F2_Top Down_dist',
    # #             'NORBNI_Saline_Ai14_OPRK1_C1_M1_Top Down_dist',
    # #             'NORBNI_Saline_Ai14_OPRK1_C1_M2_Top Down_dist',
    # #             'NORBNI_Saline_Ai14_OPRK1_C1_M3_Top Down_dist',
    # #             'NORBNI_Saline_Ai14_OPRK1_C1_M3_Top Down_dist']]
    # # all_data['Avg Dist NORBNI+Saline'] = norbni_saline_avg_dist.mean(axis=1)
    # # all_data['Avg Dist NORBNI+Saline'] = stats.sem(norbni_saline_avg_dist, axis=1)
    # #
    # # plt.plot(all_data['time'], all_data['Avg Dist NORBNI+Saline'], color='purple', linewidth=1,
    # #          label='Average Distance NORBNI+Saline')
    #
    norbni_saline_avg_vel = all_data.loc[:,
               [
                'NORBNI_Saline_Ai14_OPRK1_C2_F0_Top Down_vel',
                'NORBNI_Saline_Ai14_OPRK1_C2_F1_Top Down_vel',
                'NORBNI_Saline_Ai14_OPRK1_C2_F2_Top Down_vel',
                'NORBNI_Saline_Ai14_OPRK1_C1_M1_Top Down_vel',
                'NORBNI_Saline_Ai14_OPRK1_C1_M2_Top Down_vel',
                'NORBNI_Saline_Ai14_OPRK1_C1_M3_Top Down_vel',
                'NORBNI_Saline_Ai14_OPRK1_C1_M4_Top Down_vel'
               ]]
    vel_only_df_og = norbni_saline_avg_vel.mean(axis=1)
    vel_only_df = vel_only_df_og.dropna()
    norbni_saline_avg_vel.dropna()
    norbni_saline_df = pd.DataFrame(data=vel_only_df)
    print(norbni_saline_df)

    # all_data['Avg Vel NORBNI+Saline'] = norbni_saline_avg_vel.mean(axis=1)
    # all_data['Avg Vel NORBNI+Saline SEM'] = stats.sem(norbni_saline_avg_vel, axis=1)

    # plt.plot(all_data['time'], all_data['Avg Vel NORBNI+Saline'], color='purple', linewidth=1,
    #          label='Average Velocity NORBNI+Saline')
    vel_norbni_sal_sem = stats.sem(norbni_saline_df, axis=1)
    index_vals_ns = vel_only_df.index
    better_time_ns = index_vals_ns*(1./fps)/60
    plt.plot(better_time_ns, vel_only_df, color='purple', linewidth=1,
             label='Average Velocity NORBNI+Saline')

    #
    # plt.fill_between(better_time_sal, vel_only_df_sal + vel_norbni_sal_sem,
    #                  vel_only_df_sal - vel_norbni_sal_sem, alpha=0.25, facecolor='black',
    #                  edgecolor='black')
    # plt.fill_between(better_time_u50, vel_only_df_u50 - vel_only_df_u50_sem,
    #                  vel_only_df_u50 + vel_only_df_u50_sem, alpha=0.25, facecolor='orange',
    #                  edgecolor='orange')
    # plt.fill_between(better_time_nal, vel_only_df_nal - vel_only_df_nal_sem,
    #                  vel_only_df_nal + vel_only_df_nal_sem, alpha=0.25, facecolor='red',
    #                  edgecolor='red')
    plt.fill_between(better_time_nu, vel_only_df_nu - vel_only_df_nu_sem, vel_only_df_nu + vel_only_df_nu_sem, alpha=0.25, facecolor='blue', edgecolor='blue')
    plt.fill_between(better_time_ns, vel_only_df - vel_norbni_sal_sem, vel_only_df + vel_norbni_sal_sem, alpha=0.25, facecolor='purple', edgecolor='purple')

    '-----------------------------------------------------------------------------------------------------------'
    plt.xlabel('Time [minutes]')
    plt.ylabel('Instantaneous Velocity [cm/s]')
    plt.title('Instantaneous Velocity [cm/s] '+str(updated_window*2)+' second average ')
    # plt.legend(loc='upper left')
    leg = plt.legend(loc='upper left')
    for i in leg.legendHandles:
        i.set_linewidth(3)
    plt.show()

    # plt.xlabel('Time')
    # plt.ylabel('Cummulative Sum')
    # plt.show()

