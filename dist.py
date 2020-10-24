import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# prevent numpy exponential
# notation on print, default False
np.set_printoptions(suppress=True)



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
    data_df['|diff X|'] = data_df['snout_x'].diff(-1)
    data_df['|diff X|'] = data_df['|diff X|'].abs()

    data_df['|diff Y|'] = data_df['snout_y'].diff(-1)
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
    plt.plot(data_df['Time Elapsed'], data_df['eucDistSum'], color=line_color, marker='o', markersize=0.1, linewidth=0.1, label=legend_label)

    # plot formatting
    plt.xlabel('time (minutes)')
    plt.ylabel('distance travelled (pixels)')
    plt.legend(loc=2)
    # plt.title('total distance traveled vs. time: ' + path)
    animal = []
    animal[:] = ' '.join(file.split()[2:5])
    # plt.title('Total Distance vs. Time for: ' + ' '.join(file.split()[:2]) + " "+ ''.join(animal[:2]))
    plt.title('5mgkg U50 Prelim Distance')
    leg = plt.legend()
    for i in leg.legendHandles:
        i.set_linewidth(3)


if __name__ == '__main__':

    """Saline Data"""
    distance_det(file='Saline_Ai14_OPRK1_C1_M4_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered - Copy.h5',
                 legend_label='Saline M4', line_color='lightgreen')


    # """Naltrexone Data"""
    # distance_det(file='Nalt_U50_Ai14_OPRK1_C1_F2_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered - Copy.h5',
    #              legend_label='F2 Pretreat 3mgkg Naltrexone+5mgkg U50', line_color='darkred')
    # distance_det(file='Nalt_U50_Ai14_OPRK1_C1_M2_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered - Copy.h5',
    #              legend_label='M2 Pretreat 3mgkg Naltrexone+5mgkg U50', line_color='lightcoral')
    # distance_det(file='Nalt_U50_Ai14_OPRK1_C1_M3_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered - Copy.h5',
    #              legend_label='M3 Pretreat 3mgkg Naltrexone+5mgkg U50', line_color='red')



    """U50 Data"""

    distance_det(file='U50_Ai14_OPRK1_C1_F1_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered - Copy.h5',
                 legend_label='F1 5mgkg U50', line_color='deepskyblue')
    distance_det(file='U50_Ai14_OPRK1_C1_F0_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered - Copy.h5',
                 legend_label='F0 5mgkg U50', line_color='steelblue')
    distance_det(file='U50_Ai14_OPRK1_C1_M1_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered - Copy.h5',
                 legend_label='M1 5mgkg U50', line_color='lightblue')

    plt.show()
