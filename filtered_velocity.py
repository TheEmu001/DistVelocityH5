import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


# prevent numpy exponential
# notation on print, default False
np.set_printoptions(suppress=True)

avg_df = pd.DataFrame(data=None, columns=['Time'])
list_no = np.arange(0.0, 108000.0, 1.0)
avg_df['Time'] = (list_no*(1/60))/60
rolling_avg_duration= 10 #in seconds

def vel_det(file, legend_label, line_color):
    fps=60

    data_df = pd.read_hdf(path_or_buf=file)
    bodyparts = data_df.columns.get_level_values(1)
    coords = data_df.columns.get_level_values(2)
    bodyparts2plot = bodyparts
    scorer = data_df.columns.get_level_values(0)[0]
    Time = np.arange(np.size(data_df[scorer][bodyparts2plot[0]]['x'].values))
    column_title = bodyparts + "_" + coords
    data_df.columns = column_title

    # calculate the time elapsed per frame and append column
    data_df['Time Elapsed'] = Time / fps

    # calculate the difference from row under to row before
    # then calculate absolute value
    data_df['|diff X|'] = data_df['head_x'].diff(-1)
    data_df['|diff X|'] = data_df['|diff X|'].abs()

    data_df['|diff Y|'] = data_df['head_y'].diff(-1)
    data_df['|diff Y|'] = data_df['|diff Y|'].abs()

    # calculating the cummulative sum down the column
    data_df['sumX'] = data_df['|diff X|'].cumsum()
    data_df['sumY'] = data_df['|diff Y|'].cumsum()

    # squaring delta x and y values
    data_df['deltax^2'] = data_df['|diff X|']**2
    data_df['deltay^2'] = data_df['|diff Y|']**2

    # adding deltaX^2 + deltaY^2
    data_df['deltaSummed'] = (data_df['deltax^2'] + data_df['deltay^2'])*.03924

    # taking square root of deltaX^2 + deltaY^2
    data_df['eucDist'] = data_df['deltaSummed']**(1/2)
    data_df['velocity'] = data_df['eucDist']*1/fps
    data_df['velocity_roll'] = data_df['eucDist'].rolling(rolling_avg_duration*fps).mean()

    # print(data_df)

    # what's being plotted
    # plt.plot(data_df['Time Elapsed'], data_df['velocity_roll'], color=line_color, marker='o', markersize=0.4, linewidth=0.3, label=legend_label) # scatter plot with faint lines
    # plt.plot(data_df['Time Elapsed']/60, data_df['velocity_roll'], color=line_color, linewidth=1, label=legend_label)
    # plot formatting
    # plt.xlabel('time (seconds)')
    # plt.ylabel('velocity (pixels/second)')
    # plt.legend(loc=2)
    # plt.title('total distance traveled vs. time: ' + path)
    animal = []
    animal[:] = ' '.join(file.split()[2:5])
    # plt.title('Total Distance vs. Time for: ' + ' '.join(file.split()[:2]) + " "+ ''.join(animal[:2]))
    # plt.title(str(rolling_avg_duration)+' second Rolling Velocity Pretreat 3mkgNaltrexone+5mgkg U50')
    print(data_df)

    avg_df[file] = data_df['velocity_roll']
if __name__ == '__main__':

    """Saline Data"""
    vel_det(file='Saline_Ai14_OPRK1_C1_M4_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered - Copy.h5',
                 legend_label='Saline M4', line_color='lightgreen')
    vel_det(file='Saline_Ai14_OPRK1_C1_M3_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered - Copy.h5',
                 legend_label='Saline M3', line_color='springgreen')
    vel_det(file='Saline_Ai14_OPRK1_C2_F1_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered - Copy.h5',
                 legend_label='Saline F1', line_color='seagreen')
    vel_det(file='Saline_Ai14_OPRK1_C1_M1_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered - Copy.h5',
                 legend_label='Saline M1', line_color='forestgreen')
    vel_det(file='Saline_Ai14_OPRK1_C1_M2_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered - Copy.h5',
                 legend_label='Saline M2', line_color='lime')
    vel_det(file='Saline_Ai14_OPRK1_C1_F0_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered - Copy.h5',
                 legend_label='Saline F0', line_color='yellowgreen')
    vel_det(file='Saline_Ai14_OPRK1_C1_F2_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered - Copy.h5',
                 legend_label='Saline F2', line_color='olivedrab')
    only_saline = avg_df.loc[:, ['Saline_Ai14_OPRK1_C1_M4_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered - Copy.h5',
                                 'Saline_Ai14_OPRK1_C1_M3_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered - Copy.h5',
                                 'Saline_Ai14_OPRK1_C2_F1_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered - Copy.h5',
                                 'Saline_Ai14_OPRK1_C1_M1_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered - Copy.h5',
                                 'Saline_Ai14_OPRK1_C1_M2_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered - Copy.h5',
                                 'Saline_Ai14_OPRK1_C1_F0_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered - Copy.h5',
                                 'Saline_Ai14_OPRK1_C1_F2_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered - Copy.h5']]
    avg_df['Avg Vel Saline'] = only_saline.mean(axis=1)
    avg_df['Avg Vel Saline SEM'] = stats.sem(only_saline, axis=1)
    plt.plot(avg_df['Time'], avg_df['Avg Vel Saline'], color='black', linewidth=1, label='Average Velocity Saline+Saline')

    """Naltrexone Data"""
    vel_det(file='Nalt_U50_Ai14_OPRK1_C1_F2_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered - Copy.h5',
                 legend_label='F2 Pretreat 3mgkg Naltrexone+5mgkg U50', line_color='darkred')
    vel_det(file='Nalt_U50_Ai14_OPRK1_C1_M2_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered - Copy.h5',
                 legend_label='M2 Pretreat 3mgkg Naltrexone+5mgkg U50', line_color='lightcoral')
    vel_det(file='Nalt_U50_Ai14_OPRK1_C1_M3_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered - Copy.h5',
                 legend_label='M3 Pretreat 3mgkg Naltrexone+5mgkg U50', line_color='red')
    vel_det(file='Nalt_U50_Ai14_OPRK1_C1_M1_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered - Copy.h5',
                 legend_label='M1 Pretreat 3mgkg Naltrexone+5mgkg U50', line_color='firebrick')
    vel_det(file='Nalt_U50_Ai14_OPRK1_C1_M4_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered - Copy.h5',
                 legend_label='M4 Pretreat 3mgkg Naltrexone+5mkg U50', line_color='darksalmon')
    vel_det(file='Naltr_U50_Ai14_OPRK1_C2_F0_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered - Copy.h5',
                 legend_label='F0 Pretreat 3mkg Naltrexone+5mgkg U50', line_color='#ee4466')
    vel_det(file='Nalt_U50_Ai14_OPRK1_C1_F1_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered - Copy.h5',
                 legend_label='F1 Pretreat 3mgkg Naltrexone+5mgkg U50', line_color='orangered')
    only_naltr = avg_df.loc[:,
                 ['Nalt_U50_Ai14_OPRK1_C1_F2_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered - Copy.h5',
                  'Nalt_U50_Ai14_OPRK1_C1_M2_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered - Copy.h5',
                  'Nalt_U50_Ai14_OPRK1_C1_M3_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered - Copy.h5',
                  'Nalt_U50_Ai14_OPRK1_C1_M1_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered - Copy.h5',
                  'Nalt_U50_Ai14_OPRK1_C1_M4_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered - Copy.h5',
                  'Naltr_U50_Ai14_OPRK1_C2_F0_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered - Copy.h5',
                  'Nalt_U50_Ai14_OPRK1_C1_F1_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered - Copy.h5']]
    avg_df['Avg Vel Naltr'] = only_naltr.mean(axis=1)
    avg_df['Avg Vel Naltr SEM'] = stats.sem(only_naltr, axis=1)
    plt.plot(avg_df['Time'], avg_df['Avg Vel Naltr'], color='red', linewidth=1, label='Average Velocity 3mgkg Naltr+5mgkg U50')


    """U50 Data"""

    vel_det(file='U50_Ai14_OPRK1_C1_F1_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered - Copy.h5',
                 legend_label='F1 5mgkg U50', line_color='deepskyblue')
    vel_det(file='U50_Ai14_OPRK1_C1_F0_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered - Copy.h5',
            legend_label='F0 5mgkg U50', line_color='steelblue')
    vel_det(file='U50_Ai14_OPRK1_C1_M1_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered - Copy.h5',
                 legend_label='M1 5mgkg U50', line_color='lightblue')
    vel_det(file='U50_Ai14_OPRK1_C1_M2_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered - Copy.h5',
                 legend_label='M2 5mgkg U50', line_color='cornflowerblue')
    vel_det(file='U50_Ai14_OPRK1_C2_F2_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered - Copy.h5',
                 legend_label='F2 5mgkg U50', line_color='powderblue')
    vel_det(file='U50_Ai14_OPRK1_C1_M3_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered - Copy.h5',
                 legend_label='M3 5mgkg U50', line_color='aquamarine')
    vel_det(file='U50_Ai14_OPRK1_C1_M4_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered - Copy.h5',
                 legend_label='M4 5mgkg U50', line_color='turquoise')
    only_U50 = avg_df.loc[:,
               ['U50_Ai14_OPRK1_C1_F1_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered - Copy.h5',
                'U50_Ai14_OPRK1_C1_F0_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered - Copy.h5',
                'U50_Ai14_OPRK1_C1_M1_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered - Copy.h5',
                'U50_Ai14_OPRK1_C1_M2_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered - Copy.h5',
                'U50_Ai14_OPRK1_C2_F2_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered - Copy.h5',
                'U50_Ai14_OPRK1_C1_M3_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered - Copy.h5',
                'U50_Ai14_OPRK1_C1_M4_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered - Copy.h5']]
    avg_df['Avg Vel U50'] = only_U50.mean(axis=1)
    avg_df['Avg Vel U50 SEM'] = stats.sem(only_U50, axis=1)
    plt.plot(avg_df['Time'], avg_df['Avg Vel U50'], color='orange', linewidth=1, label='Average Velocity Saline+5mgkg U50')


    """NORBNI Data"""

    vel_det(file='NORBNI_U50_Ai14_OPRK1_C2_F1_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered.h5',
            legend_label='F1 10mgkg NORBNI+5mgkg U50', line_color='deepskyblue')
    vel_det(file='NORBNI_U50_Ai14_OPRK1_C2_F2_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered.h5',
            legend_label='F2 10mgkg NORBNI+5mgkg U50', line_color='steelblue')
    vel_det(file='NORBNI_U50_Ai14_OPRK1_C1_M3_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered.h5',
                 legend_label='M3 10mgkg NORBNI+5mgkg U50', line_color='steelblue')
    vel_det(file='NORBNI_U50_Ai14_OPRK1_C1_M4_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered.h5',
                 legend_label='M4 10mgkg NORBNI+5mgkg U50', line_color='steelblue')
    only_NORBNI = avg_df.loc[:,
               [
                  'NORBNI_U50_Ai14_OPRK1_C2_F1_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered.h5',
                'NORBNI_U50_Ai14_OPRK1_C2_F2_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered.h5',
                'NORBNI_U50_Ai14_OPRK1_C1_M3_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered.h5',
                'NORBNI_U50_Ai14_OPRK1_C1_M4_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered.h5'
                ]]
    avg_df['Avg Vel NORBNI'] = only_NORBNI.mean(axis=1)
    avg_df['Avg Vel NORBNI SEM'] = stats.sem(only_NORBNI, axis=1)
    plt.plot(avg_df['Time'], avg_df['Avg Vel NORBNI'], color='blue', linewidth=1,
             label='Average Velocity 10mgkg NORBNI +5mgkg U50')

    plt.fill_between(avg_df['Time'], avg_df["Avg Vel Saline"]-avg_df["Avg Vel Saline SEM"],
                     avg_df["Avg Vel Saline"]+avg_df["Avg Vel Saline SEM"], alpha=0.25, facecolor='black', edgecolor='black')
    plt.fill_between(avg_df['Time'], avg_df["Avg Vel Naltr"]-avg_df["Avg Vel Naltr SEM"],
                     avg_df["Avg Vel Naltr"]+avg_df["Avg Vel Naltr SEM"], alpha=0.25, facecolor='red', edgecolor='red')
    plt.fill_between(avg_df['Time'], avg_df["Avg Vel U50"]-avg_df["Avg Vel U50 SEM"],
                     avg_df["Avg Vel U50"]+avg_df["Avg Vel U50 SEM"], alpha=0.25, facecolor='orange', edgecolor='orange')
    plt.fill_between(avg_df['Time'], avg_df["Avg Vel NORBNI"]-avg_df["Avg Vel NORBNI SEM"],
                     avg_df["Avg Vel NORBNI"]+avg_df["Avg Vel NORBNI SEM"], alpha=0.25, facecolor='blue', edgecolor='blue')

    leg = plt.legend()
    font = {'family': 'Arial',
            'size': 12}
    plt.rc('font', **font)
    plt.rc('lines', linewidth = 1)
    for i in leg.legendHandles:
        i.set_linewidth(3)
    plt.xlabel('time (minutes)', fontsize=12)
    plt.ylabel('velocity (cm/second)', fontsize=12)
    plt.title('Velocity vs Time, Rolling Average '+str(rolling_avg_duration)+" second interval")
    plt.show()