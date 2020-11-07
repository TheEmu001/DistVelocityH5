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

# frame rate of camera in those experiments
fps = 60
no_seconds = 10
moving_average_duration_frames = fps * no_seconds

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


    time=np.arange(len(vel))*1./fps
    #notice the units of vel are relative pixel distance [per time step]
    vel=vel*.03924
    velocity_pd = pd.Series(vel)
    vel_avg = velocity_pd.rolling(moving_average_duration_frames).mean()

    inst_dist = vel*1./fps
    dist_sum = inst_dist.cumsum()


    # plt.plot(time, vel_avg, color=color, label=label) #instantaneous rolling average velocity plot
    # plt.plot(time, dist_sum, color=color) #cummulative distance plot
    all_data[video+"_dist"] = dist_sum[:108060]
    all_data[video+"_vel"] = vel_avg[:108060]




if __name__ == '__main__':
    all_data['time'] = (list_no * (1 / 60)) / 60

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

    saline_avg_dist = all_data.loc[:,
               ['Saline_Ai14_OPRK1_C2_F0_Top Down_dist',
                'Saline_Ai14_OPRK1_C2_F1_Top Down_dist',
                'Saline_Ai14_OPRK1_C1_F2_Top Down_dist',
                'Saline_Ai14_OPRK1_C1_M1_Top Down_dist',
                'Saline_Ai14_OPRK1_C1_M2_Top Down_dist',
                'Saline_Ai14_OPRK1_C1_M3_Top Down_dist',
                'Saline_Ai14_OPRK1_C1_M4_Top Down_dist']]
    all_data['Avg Dist Saline'] = saline_avg_dist.mean(axis=1)
    all_data['Avg Dist Saline SEM'] = stats.sem(saline_avg_dist, axis=1)

    plt.plot(all_data['time'], all_data['Avg Dist Saline'], color='pink', linewidth=1,
             label='Average Distance Saline')


    # saline_avg_vel = all_data.loc[:,
    #            ['Saline_Ai14_OPRK1_C2_F0_Top Down_vel',
    #             'Saline_Ai14_OPRK1_C2_F1_Top Down_vel',
    #             'Saline_Ai14_OPRK1_C1_F2_Top Down_vel',
    #             'Saline_Ai14_OPRK1_C1_M1_Top Down_vel',
    #             'Saline_Ai14_OPRK1_C1_M2_Top Down_vel',
    #             'Saline_Ai14_OPRK1_C1_M3_Top Down_vel',
    #             'Saline_Ai14_OPRK1_C1_M4_Top Down_vel']]
    #
    # all_data['Avg Vel Saline'] = saline_avg_vel.mean(axis=1)
    # all_data['Avg Vel Saline SEM'] = stats.sem(saline_avg_vel, axis=1)
    # all_data['time'] = (list_no * (1 / 60)) / 60
    # plt.plot(all_data['time'], all_data['Avg Vel Saline'], color='pink', linewidth=1,
    #          label='Average Velocity Saline')

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

    u50_avg_dist = all_data.loc[:,
               ['U50_Ai14_OPRK1_C2_F0_Top Down_dist',
                'U50_Ai14_OPRK1_C1_F1_Top Down_dist',
                'U50_Ai14_OPRK1_C2_F2_Top Down_dist',
                'U50_Ai14_OPRK1_C1_M1_Top Down_dist',
                'U50_Ai14_OPRK1_C1_M2_Top Down_dist',
                'U50_Ai14_OPRK1_C1_M3_Top Down_dist',
                'U50_Ai14_OPRK1_C1_M4_Top Down_dist']]
    all_data['Avg Dist U50'] = u50_avg_dist.mean(axis=1)
    all_data['Avg Dist U50 SEM'] = stats.sem(u50_avg_dist, axis=1)

    plt.plot(all_data['time'], all_data['Avg Dist U50'], color='orange', linewidth=1,
             label='Average Distance U50')

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
    # plt.plot(all_data['time'], all_data['Avg Vel U50'], color='orange', linewidth=1,
    #          label='Average Vel U50')

    """
    Naltrexone U50
    """
    velocity(video='Naltr_U50_Ai14_OPRK1_C2_F0_Top Down', color='red', label='F0 3mgkg Nalt+5mgkg U50')
    velocity(video='Nalt_U50_Ai14_OPRK1_C1_F1_Top Down', color='red', label='F1 3mgkg Nalt+5mgkg U50')
    velocity(video='Nalt_U50_Ai14_OPRK1_C1_F2_Top Down', color='red', label='F2 3mgkg Nalt+5mgkg U50')

    velocity(video='Nalt_U50_Ai14_OPRK1_C1_M1_Top Down', color='red', label='M1 3mgkg Nalt+5mgkg U50')
    velocity(video='Nalt_U50_Ai14_OPRK1_C1_M2_Top Down', color='red', label='M2 3mgkg Nalt+5mgkg U50')
    velocity(video='Nalt_U50_Ai14_OPRK1_C1_M3_Top Down', color='red', label='M3 3mgkg Nalt+5mgkg U50')
    velocity(video='Nalt_U50_Ai14_OPRK1_C1_M4_Top Down', color='red', label='M4 3mgkg Nalt+5mgkg U50')

    nalt_avg_dist = all_data.loc[:,
               ['Naltr_U50_Ai14_OPRK1_C2_F0_Top Down_dist',
                'Nalt_U50_Ai14_OPRK1_C1_F1_Top Down_dist',
                'Nalt_U50_Ai14_OPRK1_C1_F2_Top Down_dist',
                'Nalt_U50_Ai14_OPRK1_C1_M1_Top Down_dist',
                'Nalt_U50_Ai14_OPRK1_C1_M2_Top Down_dist',
                'Nalt_U50_Ai14_OPRK1_C1_M3_Top Down_dist',
                'Nalt_U50_Ai14_OPRK1_C1_M4_Top Down_dist']]
    all_data['Avg Dist Nalt'] = nalt_avg_dist.mean(axis=1)
    all_data['Avg Dist Nalt SEM'] = stats.sem(nalt_avg_dist, axis=1)

    plt.plot(all_data['time'], all_data['Avg Dist Nalt'], color='red', linewidth=1,
             label='Average Distance Nalt')

    # nalt_avg_vel = all_data.loc[:,
    #            ['Naltr_U50_Ai14_OPRK1_C2_F0_Top Down_vel',
    #             'Nalt_U50_Ai14_OPRK1_C1_F1_Top Down_vel',
    #             'Nalt_U50_Ai14_OPRK1_C1_F2_Top Down_vel',
    #             'Nalt_U50_Ai14_OPRK1_C1_M1_Top Down_vel',
    #             'Nalt_U50_Ai14_OPRK1_C1_M2_Top Down_vel',
    #             'Nalt_U50_Ai14_OPRK1_C1_M3_Top Down_vel',
    #             'Nalt_U50_Ai14_OPRK1_C1_M4_Top Down_vel']]
    # all_data['Avg Vel Nalt'] = nalt_avg_vel.mean(axis=1)
    # all_data['Avg Vel Nalt SEM'] = stats.sem(nalt_avg_vel, axis=1)
    #
    # plt.plot(all_data['time'], all_data['Avg Vel Nalt'], color='red', linewidth=1,
    #          label='Average Velocity Nalt')

    """
    NORBNI U50
    """
    velocity(video='NORBNI_U50_Ai14_OPRK1_C2_F0_Top Down', color='blue', label='F0 10mgkg NORBNI+5mgkg U50')
    velocity(video='NORBNI_U50_Ai14_OPRK1_C2_F1_Top Down', color='blue', label='F1 10mgkg NORBNI+5mgkg U50')
    velocity(video='NORBNI_U50_Ai14_OPRK1_C2_F2_Top Down', color='blue', label='F2 10mgkg NORBNI+5mgkg U50')

    velocity(video='NORBNI_U50_Ai14_OPRK1_C1_M1_Top Down', color='blue', label='M1 10mgkg NORBNI+5mgkg U50')
    velocity(video='NORBNI_U50_Ai14_OPRK1_C1_M2_Top Down', color='blue', label='M2 10mgkg NORBNI+5mgkg U50')
    velocity(video='NORBNI_U50_Ai14_OPRK1_C1_M3_Top Down', color='blue', label='M3 10mgkg NORBNI+5mgkg U50')
    velocity(video='NORBNI_U50_Ai14_OPRK1_C1_M4_Top Down', color='blue', label='M4 10mgkg NORBNI+5mgkg U50')
    norbni_u50_avg_dist = all_data.loc[:,
               ['NORBNI_U50_Ai14_OPRK1_C2_F0_Top Down_dist',
                'NORBNI_U50_Ai14_OPRK1_C2_F1_Top Down_dist',
                'NORBNI_U50_Ai14_OPRK1_C2_F2_Top Down_dist',
                'NORBNI_U50_Ai14_OPRK1_C1_M1_Top Down_dist',
                'NORBNI_U50_Ai14_OPRK1_C1_M2_Top Down_dist',
                'NORBNI_U50_Ai14_OPRK1_C1_M3_Top Down_dist',
                'NORBNI_U50_Ai14_OPRK1_C1_M4_Top Down_dist']]
    all_data['Avg Dist NORBNI+U50'] = norbni_u50_avg_dist.mean(axis=1)
    all_data['Avg Dist NORBNI+U50 SEM'] = stats.sem(norbni_u50_avg_dist, axis=1)

    plt.plot(all_data['time'], all_data['Avg Dist NORBNI+U50'], color='blue', linewidth=1,
             label='Average Distance NORBNI+U50')

    # norbni_u50_avg_vel = all_data.loc[:,
    #            ['NORBNI_U50_Ai14_OPRK1_C2_F0_Top Down_vel',
    #             'NORBNI_U50_Ai14_OPRK1_C2_F1_Top Down_vel',
    #             'NORBNI_U50_Ai14_OPRK1_C2_F2_Top Down_vel',
    #             'NORBNI_U50_Ai14_OPRK1_C1_M1_Top Down_vel',
    #             'NORBNI_U50_Ai14_OPRK1_C1_M2_Top Down_vel',
    #             'NORBNI_U50_Ai14_OPRK1_C1_M3_Top Down_vel',
    #             'NORBNI_U50_Ai14_OPRK1_C1_M4_Top Down_vel']]
    # all_data['Avg Vel NORBNI+U50'] = norbni_u50_avg_dist.mean(axis=1)
    # all_data['Avg Vel NORBNI+U50 SEM'] = stats.sem(norbni_u50_avg_vel, axis=1)
    #
    # plt.plot(all_data['time'], all_data['Avg Vel NORBNI+U50'], color='blue', linewidth=1,
    #          label='Average Vel NORBNI+U50')

    """
    NORBNI + Saline
    """
    velocity(video='NORBNI_Saline_Ai14_OPRK1_C2_F0_Top Down', color='purple', label='F0 10mgkg NORBNI+Saline')
    velocity(video='NORBNI_Saline_Ai14_OPRK1_C2_F1_Top Down', color='purple', label='F1 10mgkg NORBNI+Saline')
    velocity(video='NORBNI_Saline_Ai14_OPRK1_C2_F2_Top Down', color='purple', label='F2 10mgkg NORBNI+Saline')

    velocity(video='NORBNI_Saline_Ai14_OPRK1_C1_M1_Top Down', color='purple', label='M1 10mgkg NORBNI+Saline')
    velocity(video='NORBNI_Saline_Ai14_OPRK1_C1_M2_Top Down', color='purple', label='M2 10mgkg NORBNI+Saline')
    velocity(video='NORBNI_Saline_Ai14_OPRK1_C1_M3_Top Down', color='purple', label='M3 10mgkg NORBNI+Saline')
    velocity(video='NORBNI_Saline_Ai14_OPRK1_C1_M4_Top Down', color='purple', label='M4 10mgkg NORBNI+Saline')

    norbni_saline_avg_dist = all_data.loc[:,
               ['NORBNI_Saline_Ai14_OPRK1_C2_F0_Top Down_dist',
                'NORBNI_Saline_Ai14_OPRK1_C2_F1_Top Down_dist',
                'NORBNI_Saline_Ai14_OPRK1_C2_F2_Top Down_dist',
                'NORBNI_Saline_Ai14_OPRK1_C1_M1_Top Down_dist',
                'NORBNI_Saline_Ai14_OPRK1_C1_M2_Top Down_dist',
                'NORBNI_Saline_Ai14_OPRK1_C1_M3_Top Down_dist',
                'NORBNI_Saline_Ai14_OPRK1_C1_M3_Top Down_dist']]
    all_data['Avg Dist NORBNI+Saline'] = norbni_saline_avg_dist.mean(axis=1)
    all_data['Avg Dist NORBNI+Saline'] = stats.sem(norbni_saline_avg_dist, axis=1)

    plt.plot(all_data['time'], all_data['Avg Dist NORBNI+Saline'], color='purple', linewidth=1,
             label='Average Distance NORBNI+Saline')

    # norbni_saline_avg_vel = all_data.loc[:,
    #            ['NORBNI_Saline_Ai14_OPRK1_C2_F0_Top Down_vel',
    #             'NORBNI_Saline_Ai14_OPRK1_C2_F1_Top Down_vel',
    #             'NORBNI_Saline_Ai14_OPRK1_C2_F2_Top Down_vel',
    #             'NORBNI_Saline_Ai14_OPRK1_C1_M1_Top Down_vel',
    #             'NORBNI_Saline_Ai14_OPRK1_C1_M2_Top Down_vel',
    #             'NORBNI_Saline_Ai14_OPRK1_C1_M3_Top Down_vel',
    #             'NORBNI_Saline_Ai14_OPRK1_C1_M3_Top Down_vel']]
    # all_data['Avg Vel NORBNI+Saline'] = norbni_saline_avg_vel.mean(axis=1)
    # all_data['Avg Vel NORBNI+Saline'] = stats.sem(norbni_saline_avg_vel, axis=1)
    #
    # plt.plot(all_data['time'], all_data['Avg Vel NORBNI+Saline'], color='purple', linewidth=1,
    #          label='Average Velocity NORBNI+Saline')

    '-----------------------------------------------------------------------------------------------------------'
    # plt.xlabel('Time')
    # plt.ylabel('Instantaneous Velocity')
    # plt.show()

    plt.xlabel('Time')
    plt.ylabel('Cummulative Sum')
    plt.show()

