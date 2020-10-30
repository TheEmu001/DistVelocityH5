import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


# prevent numpy exponential
# notation on print, default False
np.set_printoptions(suppress=True)

avg_df = pd.DataFrame(data=None, columns=['Time'])
list_no = np.arange(0.0, 158400.0, 1.0)
avg_df['Time'] = (list_no*(1/60))/60

def distance_det(file, legend_label, line_color):
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
    data_df['Time Elapsed'] = Time / fps/60

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
    data_df['deltaSummed'] = data_df['deltax^2'] + data_df['deltay^2']

    # taking square root of deltaX^2 + deltaY^2
    data_df['eucDist'] = data_df['deltaSummed']**(1/2)
    data_df['eucDistSum'] = data_df['eucDist'].cumsum()

    print(data_df)

    # what's being plotted
    # plt.plot(data_df['Time Elapsed'], data_df['sumX'],color='blue', marker='o', markersize=0.1, linewidth=0.1, label='xSum')
    # plt.plot(data_df['Time Elapsed'], data_df['sumY'],color='red', marker='o', markersize=0.1, linewidth=0.1, label='ySum')
    # plt.plot(data_df['Time Elapsed'], data_df['eucDistSum'], color=line_color, marker='o', markersize=0.1, linewidth=0.1, label=legend_label)
    #
    # # plot formatting
    # plt.xlabel('time (minutes)')
    # plt.ylabel('distance travelled (pixels)')
    # plt.legend(loc=2)
    # # plt.title('total distance traveled vs. time: ' + path)
    # animal = []
    # animal[:] = ' '.join(file.split()[2:5])
    # # plt.title('Total Distance vs. Time for: ' + ' '.join(file.split()[:2]) + " "+ ''.join(animal[:2]))
    # plt.title('5mgkg U50 Prelim Distance')
    # leg = plt.legend()
    # for i in leg.legendHandles:
    #     i.set_linewidth(3)

    avg_df[file] = data_df['eucDistSum']
if __name__ == '__main__':

    """Saline Data"""
    distance_det(file='Saline_Ai14_OPRK1_C1_M4_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered - Copy.h5',
                 legend_label='Saline M4', line_color='lightgreen')
    distance_det(file='Saline_Ai14_OPRK1_C1_M3_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered - Copy.h5',
                 legend_label='Saline M3', line_color='springgreen')
    distance_det(file='Saline_Ai14_OPRK1_C2_F1_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered - Copy.h5',
                 legend_label='Saline F1', line_color='seagreen')
    distance_det(file='Saline_Ai14_OPRK1_C1_M1_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered - Copy.h5',
                 legend_label='Saline M1', line_color='forestgreen')
    distance_det(file='Saline_Ai14_OPRK1_C1_M2_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered - Copy.h5',
                 legend_label='Saline M2', line_color='lime')
    distance_det(file='Saline_Ai14_OPRK1_C1_F0_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered - Copy.h5',
                 legend_label='Saline F0', line_color='yellowgreen')
    distance_det(file='Saline_Ai14_OPRK1_C1_F2_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered - Copy.h5',
                 legend_label='Saline F2', line_color='olivedrab')
    only_saline = avg_df.loc[:, ['Saline_Ai14_OPRK1_C1_M4_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered - Copy.h5',
                                 'Saline_Ai14_OPRK1_C1_M3_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered - Copy.h5',
                                 'Saline_Ai14_OPRK1_C2_F1_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered - Copy.h5',
                                 'Saline_Ai14_OPRK1_C1_M1_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered - Copy.h5',
                                 'Saline_Ai14_OPRK1_C1_M2_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered - Copy.h5',
                                 'Saline_Ai14_OPRK1_C1_F0_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered - Copy.h5',
                                 'Saline_Ai14_OPRK1_C1_F2_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered - Copy.h5']]
    avg_df['Avg Dist Saline'] = only_saline.mean(axis=1)
    avg_df['Avg Dist Saline SEM'] = stats.sem(only_saline, axis=1)
    plt.plot(avg_df['Time'], avg_df['Avg Dist Saline'], color='green', linewidth=1, label='Average Dist Saline+Saline')

    # """Naltrexone Data"""
    distance_det(file='Nalt_U50_Ai14_OPRK1_C1_F2_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered - Copy.h5',
                 legend_label='F2 Pretreat 3mgkg Naltrexone+5mgkg U50', line_color='darkred')
    distance_det(file='Nalt_U50_Ai14_OPRK1_C1_M2_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered - Copy.h5',
                 legend_label='M2 Pretreat 3mgkg Naltrexone+5mgkg U50', line_color='lightcoral')
    distance_det(file='Nalt_U50_Ai14_OPRK1_C1_M3_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered - Copy.h5',
                 legend_label='M3 Pretreat 3mgkg Naltrexone+5mgkg U50', line_color='red')
    distance_det(file='Nalt_U50_Ai14_OPRK1_C1_M1_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered - Copy.h5',
                 legend_label='M1 Pretreat 3mgkg Naltrexone+5mgkg U50', line_color='firebrick')
    distance_det(file='Nalt_U50_Ai14_OPRK1_C1_M4_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered - Copy.h5',
                 legend_label='M4 Pretreat 3mgkg Naltrexone+5mkg U50', line_color='darksalmon')
    distance_det(file='Naltr_U50_Ai14_OPRK1_C2_F0_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered - Copy.h5',
                 legend_label='F0 Pretreat 3mkg Naltrexone+5mgkg U50', line_color='#ee4466')
    distance_det(file='Nalt_U50_Ai14_OPRK1_C1_F1_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered - Copy.h5',
                 legend_label='F1 Pretreat 3mgkg Naltrexone+5mgkg U50', line_color='orangered')
    only_naltr = avg_df.loc[:, ['Nalt_U50_Ai14_OPRK1_C1_F2_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered - Copy.h5',
                                 'Nalt_U50_Ai14_OPRK1_C1_M2_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered - Copy.h5',
                                 'Nalt_U50_Ai14_OPRK1_C1_M3_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered - Copy.h5',
                                'Nalt_U50_Ai14_OPRK1_C1_M1_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered - Copy.h5',
                                'Nalt_U50_Ai14_OPRK1_C1_M4_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered - Copy.h5',
                                'Naltr_U50_Ai14_OPRK1_C2_F0_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered - Copy.h5',
                                'Nalt_U50_Ai14_OPRK1_C1_F1_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered - Copy.h5']]
    avg_df['Avg Dist Naltr'] = only_naltr.mean(axis=1)
    avg_df['Avg Dist Naltr SEM'] = stats.sem(only_naltr, axis=1)
    plt.plot(avg_df['Time'], avg_df['Avg Dist Naltr'], color='red', linewidth=1, label='Average Dist 3mgkg Naltr+5mgkg U50')


    """U50 Data"""

    distance_det(file='U50_Ai14_OPRK1_C1_F1_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered - Copy.h5',
                 legend_label='F1 5mgkg U50', line_color='deepskyblue')
    distance_det(file='U50_Ai14_OPRK1_C1_F0_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered - Copy.h5',
                 legend_label='F0 5mgkg U50', line_color='steelblue')
    distance_det(file='U50_Ai14_OPRK1_C1_M1_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered - Copy.h5',
                 legend_label='M1 5mgkg U50', line_color='lightblue')
    distance_det(file='U50_Ai14_OPRK1_C1_M2_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered - Copy.h5',
                 legend_label='M2 5mgkg U50', line_color='cornflowerblue')
    distance_det(file='U50_Ai14_OPRK1_C2_F2_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered - Copy.h5',
                 legend_label='F2 5mgkg U50', line_color='powderblue')
    distance_det(file='U50_Ai14_OPRK1_C1_M3_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered - Copy.h5',
                 legend_label='M3 5mgkg U50', line_color='aquamarine')
    distance_det(file='U50_Ai14_OPRK1_C1_M4_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered - Copy.h5',
                 legend_label='M4 5mgkg U50', line_color='turquoise')
    only_U50 = avg_df.loc[:, ['U50_Ai14_OPRK1_C1_F1_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered - Copy.h5',
                                 'U50_Ai14_OPRK1_C1_F0_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered - Copy.h5',
                                 'U50_Ai14_OPRK1_C1_M1_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered - Copy.h5',
                                'U50_Ai14_OPRK1_C1_M2_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered - Copy.h5',
                                'U50_Ai14_OPRK1_C2_F2_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered - Copy.h5',
                              'U50_Ai14_OPRK1_C1_M3_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered - Copy.h5',
                              'U50_Ai14_OPRK1_C1_M4_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered - Copy.h5']]
    avg_df['Avg Dist U50'] = only_U50.mean(axis=1)
    avg_df['Avg Dist U50 SEM'] = stats.sem(only_U50, axis=1)
    plt.rc('font', family='Arial')
    plt.plot(avg_df['Time'], avg_df['Avg Dist U50'], color='blue', linewidth=1, label='Average Dist Saline+5mgkg U50')

    plt.fill_between(avg_df['Time'], avg_df["Avg Dist Saline"]-avg_df["Avg Dist Saline SEM"],
                     avg_df["Avg Dist Saline"]+avg_df["Avg Dist Saline SEM"], alpha=0.25, facecolor='green')
    plt.fill_between(avg_df['Time'], avg_df["Avg Dist Naltr"]-avg_df["Avg Dist Naltr SEM"],
                     avg_df["Avg Dist Naltr"]+avg_df["Avg Dist Naltr SEM"], alpha=0.25, facecolor='red')
    plt.fill_between(avg_df['Time'], avg_df["Avg Dist U50"]-avg_df["Avg Dist U50 SEM"],
                     avg_df["Avg Dist U50"]+avg_df["Avg Dist U50 SEM"], alpha=0.25, facecolor='blue')

    # plot formatting
    plt.xlabel('time (minutes)', fontname='Arial', fontsize=12)
    plt.ylabel('distance travelled (pixels)', fontname='Arial', fontsize=12)
    plt.legend(loc=2)
    plt.title('Average Cummulative Distance vs Time', fontname='Arial', fontsize=12)
    leg = plt.legend()
    for i in leg.legendHandles:
        i.set_linewidth(3)

    
    plt.show()

    # plt.savefig('generated_figs\Cummulative Distance.eps', format='eps')
