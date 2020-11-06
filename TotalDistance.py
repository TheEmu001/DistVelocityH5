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

DLCscorer = 'DLC_resnet50_BigBinTopSep17shuffle1_250000'
no_seconds = 10
def velocity(video, color, label):

    dataname = str(Path(video).stem) + DLCscorer + 'filtered.h5'
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

    # frame rate of camera in those experiments
    fps=60
    moving_average_duration_frames = fps * no_seconds
    time=np.arange(len(vel))*1./fps
    #notice the units of vel are relative pixel distance [per time step]
    vel=vel*.03924
    velocity_pd = pd.Series(vel)
    vel_avg = velocity_pd.rolling(moving_average_duration_frames).mean()

    inst_dist = vel*1./fps
    dist_sum = inst_dist.cumsum()


    # plt.plot(time, vel_avg, color=color, label=label) #instantaneous velocity plot
    plt.plot(time, dist_sum, color=color) #cummulative distance plot



if __name__ == '__main__':
    """
    Saline
    """
    velocity(video='Saline_Ai14_OPRK1_C2_F0_Top Down', color='pink', label='F0 Saline')
    velocity(video='Saline_Ai14_OPRK1_C2_F1_Top Down', color='pink', label='F1 Saline')
    velocity(video='Saline_Ai14_OPRK1_C1_F2_Top Down', color='pink', label='F2 Saline')

    velocity(video='Saline_Ai14_OPRK1_C1_M1_Top Down', color='pink', label='M1 Saline')
    velocity(video='Saline_Ai14_OPRK1_C1_M2_Top Down', color='pink', label='M2 Saline')
    velocity(video='Saline_Ai14_OPRK1_C1_M3_Top Down', color='pink', label='M3 Saline')
    velocity(video='Saline_Ai14_OPRK1_C1_M4_Top Down', color='pink', label='M4 Saline')

    """
    U50
    """
    velocity(video='U50_Ai14_OPRK1_C2_F0_Top Down', color='orange', label='F0 Saline+5mgkg U50')
    velocity(video='U50_Ai14_OPRK1_C1_F1_Top Down', color='orange', label='F1 Saline+5mgkg U50')
    velocity(video='U50_Ai14_OPRK1_C2_F2_Top Down', color='orange', label='F2 Saline+5mgkg U50')

    velocity(video='U50_Ai14_OPRK1_C1_M1_Top Down', color='orange', label='M1 Saline+5mgkg U50')
    velocity(video='U50_Ai14_OPRK1_C1_M2_Top Down', color='orange', label='M2 Saline+5mgkg U50')
    velocity(video='U50_Ai14_OPRK1_C1_M3_Top Down', color='orange', label='M3 Saline+5mgkg U50')
    velocity(video='U50_Ai14_OPRK1_C1_M4_Top Down', color='orange', label='M4 Saline+5mgkg U50')
    #
    #
    # """
    # Naltrexone U50
    # """
    # velocity(video='Naltr_U50_Ai14_OPRK1_C2_F0_Top Down', color='red', label='F0 3mgkg Nalt+5mgkg U50')
    # velocity(video='Nalt_U50_Ai14_OPRK1_C1_F1_Top Down', color='red', label='F1 3mgkg Nalt+5mgkg U50')
    # velocity(video='Nalt_U50_Ai14_OPRK1_C1_F2_Top Down', color='red', label='F2 3mgkg Nalt+5mgkg U50')
    #
    # velocity(video='Nalt_U50_Ai14_OPRK1_C1_M1_Top Down', color='red', label='M1 3mgkg Nalt+5mgkg U50')
    # velocity(video='Nalt_U50_Ai14_OPRK1_C1_M2_Top Down', color='red', label='M2 3mgkg Nalt+5mgkg U50')
    # velocity(video='Nalt_U50_Ai14_OPRK1_C1_M3_Top Down', color='red', label='M3 3mgkg Nalt+5mgkg U50')
    # velocity(video='Nalt_U50_Ai14_OPRK1_C1_M4_Top Down', color='red', label='M4 3mgkg Nalt+5mgkg U50')
    #
    #
    # """
    # NORBNI U50
    # """
    # velocity(video='NORBNI_U50_Ai14_OPRK1_C2_F0_Top Down', color='blue', label='F0 10mgkg NORBNI+5mgkg U50')
    # velocity(video='NORBNI_U50_Ai14_OPRK1_C2_F1_Top Down', color='blue', label='F1 10mgkg NORBNI+5mgkg U50')
    # velocity(video='NORBNI_U50_Ai14_OPRK1_C2_F2_Top Down', color='blue', label='F2 10mgkg NORBNI+5mgkg U50')
    #
    # velocity(video='NORBNI_U50_Ai14_OPRK1_C1_M1_Top Down', color='blue', label='M1 10mgkg NORBNI+5mgkg U50')
    # velocity(video='NORBNI_U50_Ai14_OPRK1_C1_M2_Top Down', color='blue', label='M2 10mgkg NORBNI+5mgkg U50')
    # velocity(video='NORBNI_U50_Ai14_OPRK1_C1_M3_Top Down', color='blue', label='M3 10mgkg NORBNI+5mgkg U50')
    # velocity(video='NORBNI_U50_Ai14_OPRK1_C1_M4_Top Down', color='blue', label='M4 10mgkg NORBNI+5mgkg U50')
    #
    # """
    # NORBNI U50 + Saline
    # """
    # velocity(video='NORBNI_Saline_Ai14_OPRK1_C2_F0_Top Down', color='purple', label='F0 10mgkg NORBNI+Saline')
    # velocity(video='NORBNI_Saline_Ai14_OPRK1_C2_F1_Top Down', color='purple', label='F1 10mgkg NORBNI+Saline')
    # velocity(video='NORBNI_Saline_Ai14_OPRK1_C2_F2_Top Down', color='purple', label='F2 10mgkg NORBNI+Saline')
    #
    # velocity(video='NORBNI_Saline_Ai14_OPRK1_C1_M1_Top Down', color='purple', label='M1 10mgkg NORBNI+Saline')
    # velocity(video='NORBNI_Saline_Ai14_OPRK1_C1_M2_Top Down', color='purple', label='M2 10mgkg NORBNI+Saline')
    # velocity(video='NORBNI_Saline_Ai14_OPRK1_C1_M3_Top Down', color='purple', label='M3 10mgkg NORBNI+Saline')
    # velocity(video='NORBNI_Saline_Ai14_OPRK1_C1_M4_Top Down', color='purple', label='M4 10mgkg NORBNI+Saline')


    '-----------------------------------------------------------------------------------------------------------'
    # plt.xlabel('Time')
    # plt.ylabel('Instantaneous Velocity')
    # plt.show()

    plt.xlabel('Time')
    plt.ylabel('Cummulative Sum')
    plt.show()

    # calculate mean for relevant u50 columns, this calculates an average velocity for each point in time
    # all_data["Average U50 Female Dist"] = only_U50_F.mean(axis=1)
    # all_data["Average U50 Female Other"] = only_U50_F_other.mean(axis=1)
    #
    # # then calculate the rolling mean over a specified period of time (based on df index)
    # # since filmed in 30fps, this is calculated over 30 frames to represent one second
    # all_data["U50 Female SEM"] = stats.sem(only_U50_F, axis=1)
    # # all_data["Cumulative Sum U50 Female"] = all_data["Average U50 Female Dist"].cumsum()
    # all_data.dropna(axis=1)
    # # plt.plot(all_data['Paper_Redo_10_17_5mgkg_U50_Ai14_OPRK1_C2_F0_Top Down Time'], all_data["Average U50 Female Dist"],
    # #          label='U50 Female Dist', color='#c4b7ff')
    # plt.plot(all_data['Paper_Redo_10_17_5mgkg_U50_Ai14_OPRK1_C2_F0_Top Down raw time'], all_data["Average U50 Female Other"],
    #          label='U50 Female Dist Other', color='#c4b7ff')
    # # plt.fill_between(all_data['Paper_Redo_10_17_5mgkg_U50_Ai14_OPRK1_C2_F0_Top Down Time'], all_data["Average U50 Female Dist"]-all_data["U50 Female SEM"],
    # #                  all_data["Average U50 Female Dist"]+all_data["U50 Female SEM"], alpha=0.5, facecolor='#c4b7ff')
    #