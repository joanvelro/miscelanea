f"""
    This script produce the following function plots:

    lineplots
    barplots
    histograms
    congestion matrix (heat map)
    correlation matrix (heat map)
    boxplots
    scatterplots
    timeseries
    reference patterns

    @ Jose Angel Velasco (joseangel.velasco@yahoo.es)
    (C) Universidad Carlos III de Madrid
"""





""" ===================== line plots ===================
"""


def plot_line(df, var_x, var_y, var_color, road):
    """ Plot sinmple time series volume with color per algorithm
    :param df:
    :param var_x:
    :param var_y:
    :param var_color:
    :return:
    """
    import plotly.express as px
    from dl_ia_cla_settings import config_plotly
    config_plotly()

    show = True
    print_ = True

    fig = px.line(df,
                  x=var_x,
                  y=var_y,
                  color=var_color)
    if print_:
        fig.write_html("figures\\plot_line_{}_{}_{}.html".format(road, var_x, var_y))
    if show:
        fig.show()





""" ===================== Bar plots ===================
"""



def plot_bar_df(df, variable):
    """ produce a bar plot with pandas function
    :param df:
    :param variable:
    :return:
    """
    df.sort_values(by=[variable], ascending=False, inplace=True)
    df.plot(kind='bar',
            ylim=(0, 100))


def plot_bar(df, variable, road, var_group):
    """ produce a barplot with plotly

    :param df:
    :param road:
    :return:
    """
    import plotly.express as px
    from dl_ia_cla_settings import config_plotly
    config_plotly()

    print_ = True
    show=  True
    error = 0

    try:
        fig = px.bar(df, x=var_group, y=variable)
        fig.update_yaxes(title_text='')
        fig.update_xaxes(title_text='Variables')

        fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
        fig.update_layout(title='Bar plot - {}'.format(road))
        fig.update_layout(showlegend=True)

        if print_:
            fig.write_html("figures\\{}\\plot_bar_{}.html".format(road, variable))
        if show:
            fig.show()

        return error

    except Exception as exception_msg:

        ECLA_comm('(!) Error in plot_bar: ' + str(exception_msg))
        error = 1
        return error



def plot_bar_frequency(df, road):
    """ Description
    :param df:
    :param variable:
    :return:
    """
    import plotly.graph_objects as go
    from dl_ia_cla_settings import create_time_array

    t = create_time_array('00:00:00', '23:59:59', '15min')

    df_count = df.groupby(by=['hour', 'minute'])['volume'].sum()



    volume_relative = 100 * df_count.values / df_count.values.sum()


    fig = go.Figure([go.Bar(x=t, y=volume_relative)])
    fig.update_yaxes(title_text=' Freq (%)')
    fig.update_xaxes(title_text='Time steps (15 min)')

    fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
    fig.update_layout(title='{}  - Freq. distribution - Volume'.format(road))
    fig.update_layout(showlegend=True)
    fig.write_html("figures\\{}_freq_barplot_volume.html".format(road))
    fig.show()




""" ===================== Histograms ===================
"""



def plot_histogram(df, variable, n_bins_, road):
    """ plot a histogram using plotly from a vairable in a dataframe

    :param df: Data frame with the variable
    :param variable: name of the variable (column name)
    :param n_bins_: number of bins of the histogram
    :param road: name of the road asset
    :return error: error code 0: everything ok, 1: something happened
    """

    import plotly.express as px
    from dl_ia_cla_settings import units_, config_plotly, CLA_comm, dl_ia_cla_settings_check_folder
    import numpy as np
    config_plotly()

    print_ = True
    show = True
    fontsize_ = 18
    error = 0

    try:
        max_value = int(df[variable].max())
        x_axis = np.arange(0, max_value, int(max_value / 20))

        fig = px.histogram(df, x = variable, nbins = n_bins_, marginal="box", hover_data=df.columns)
        fig.update_xaxes(title_text=variable + ' ' + units_[variable])
        fig.update_layout(font=dict(family="Courier New, monospace", size=fontsize_, color="#7f7f7f"))
        fig.update_layout(title='Histogram - {} - {}'.format(road, variable))
        fig.update_layout(showlegend=True)
        fig.update_traces(opacity=0.9)
        fig.update_layout(bargap=0.2)  # gap between bars of adjacent location coordinates
        fig.update_xaxes(ticktext=x_axis, tickvals=x_axis)
        # now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
        if print_:
            dl_ia_cla_settings_check_folder('figures\\{}'.format(road))
            fig.write_html("figures\\{}\\plot_histogram_{}.html".format(road, variable))

        if show:
            fig.show()
        return error
    except Exception as exception_msg:
        CLA_comm('(!) Error in plot_histogram: ' + str(exception_msg))
        error = 1
        return error


def plot_distribution(df, variable, road):
    """ plot the distribution using plotly from a vairable in a dataframe

    :param df: Data frame with the variable
    :param variable: name of the variable (column name)
    :param road: name of the road asset
    :return error: error code 0: everything ok, 1: something happened
    """

    import plotly.figure_factory as ff
    from dl_ia_cla_settings import units_, config_plotly, CLA_comm
    import numpy as np
    config_plotly()

    print_ = True
    show = True
    fontsize_ = 18
    error = 0

    try:
        max_value = int(df[variable].max())
        x_axis = np.arange(0, max_value, int(max_value / 20))

        fig = ff.create_distplot([list(df[variable].values)], variable)
        fig.update_xaxes(title_text=variable + ' ' + units_[variable])
        fig.update_layout(font=dict(family="Courier New, monospace", size=fontsize_, color="#7f7f7f"))
        fig.update_layout(title='Distribution - {} - {}'.format(road, variable))
        fig.update_layout(showlegend=True)
        fig.update_traces(opacity=0.9)
        fig.update_layout(bargap=0.2)  # gap between bars of adjacent location coordinates
        fig.update_xaxes(ticktext=x_axis, tickvals=x_axis)

        if print_:
            fig.write_html("figures\\{}\\plot_distribution_{}.html".format(road, variable))
        if show:
            fig.show()
        return error
    except Exception as exception_msg:
        CLA_comm('(!) Error in plot_distribution: ' + str(exception_msg))
        error = 1
        return error

def plot_histogram_mpl(df, variable, n_bins, road):
    """

    :param df: Data frame with the vairable to plot
    :param variable: name of the variable (column name)
    :param n_bins_: number of bins of the histogram
    :return:
    """

    import matplotlib.pyplot as plt
    from dl_ia_cla_settings import units_
    from datetime import datetime
    import numpy as np

    print_ = True
    show = True
    fontsize_ = 18

    fig = plt.figure(figsize=(10, 8), dpi=150, facecolor='w', edgecolor='w')
    ax = fig.add_subplot(1, 1, 1)

    # fig, ax = plt.subplots(1, 1, sharey=True, tight_layout=True)
    n = ax.hist(df[variable], bins=n_bins, edgecolor='b')
    ax.set_xlabel(variable + ' ' + units_[variable], fontsize=fontsize_)
    ax.set_ylabel('Count', fontsize=fontsize_)
    xmax = df[variable].max()
    x_range = list(np.arange(0, xmax, int(xmax / 20)))
    # ax.set_xticks = x_range
    # ax.set_xticklabels = x_range
    ax.tick_params(axis='both', labelsize=fontsize_)
    now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
    plt.title('{} - Histogram - {} '.format(road, variable), fontsize=fontsize_)
    if show:
        plt.show()

    if print_:
        plt.savefig('figures\\{}_histogram_{}.png'.format(road, variable),
                    bbox_inches='tight', pad_inches=0)




"""  =================== Congestion matrix ====================
"""



def plot_congestion_matrix(df_congestions):
    """

    :param df: Data frame with the vairable to plot
    :param variable: name of the variable (column name)
    :param n_bins_: number of bins of the histogram
    :return:
    """
    congestion = df_congestions.values
    r = round(congestion, 2)

    import plotly.figure_factory as ff

    print_ = True
    show = True

    fig = ff.create_annotated_heatmap(congestion,
                                      colorscale='Viridis')
    fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
    fig.update_layout(title='Congestion Matrix - Brisbane')
    fig.update_layout(showlegend=True)

    if print_:
        fig.write_html("figures\\plot_congestion_matrix_{}.html".format(road))
    if show:
        fig.show()




"""  =================== Correlation matrix ====================
"""



def plot_correlation_matrix(df, road):
    """ This function calcualte the correlation coefficient between the columns of a dataframe and plot
        in a heat map using plotly

    :param df: dataframe with the columns being the variables
    :param road: road of the variables
    :return R: dataframe with the correlations
    """

    import plotly.figure_factory as ff
    from dl_ia_cla_settings import CLA_comm, dl_ia_cla_settings_check_folder
    from dl_ia_cla_settings import config_plotly
    config_plotly()

    error = 0
    print_ = True
    show = True

    x_ = df.columns
    y_ = df.columns

    R = df.corr()
    r = round(R, 2)

    try:
        fig = ff.create_annotated_heatmap(r.values,
                                          x = list(r.columns),
                                          y = list(r.columns),
                                          colorscale='Viridis')
        fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
        fig.update_layout(title='{} - Correlation Matrix (Pearson correlation coefficient)'.format(road))
        fig.update_layout(showlegend=True)
        # now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
        if show:
            fig.show()
        if print_:
            dl_ia_cla_settings_check_folder('figures\\{}'.format(road))
            fig.write_html("figures\\{}\\correlation_matrix.html".format(road))

        return error, R

    except Exception as exception_msg:
        CLA_comm('(!) Error in plot_correlation_matrix: ' + str(exception_msg))
        error = 1
        R = 0
        return error, R








"""  =========================== Boxplots ===========================
"""



def boxplot_plotly(df, var, var_group, var_color, road):
    """ This function do this-

    :param df:
    :param road:
    :param columns_:
    :return:
    """
    import plotly.express as px
    from dl_ia_cla_settings import units_

    print_ = True
    show = True

    fig = px.box(df, y=var, x=var_group, color=var_color)
    fig.update_xaxes(title_text=var_group)
    fig.update_yaxes(title_text='{}'.format(var + ' ' + units_[variable]))
    fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
    fig.update_layout(title='Boxplot - {} - {}'.format(road, variable))
    fig.update_layout(showlegend=True)

    # now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")

    if print_:
        fig.write_html(
        "figures\\{}\\boxplot_{}_{}_{}.html".format(road, var, var_group, var_color))
    if show:
        fig.show()




""" ================= Scatter plots ============================== 
"""





def plot_scatterplot(df, var_x, var_y, var_color, var_size, road):
    """ Produce a simple scatter plot with plotly

    :param df: dataframe that contains the variables
    :param variable_x: variable to plot in x axis
    :param variable_y: variable to plot in y axis
    :param variable_to_color: variable to use as color
    :param variable_to_color: variable to use as size
    :return:
    """
    import plotly.express as px
    from dl_ia_cla_settings import units_, CLA_comm
    from dl_ia_cla_settings import config_plotly, dl_ia_cla_settings_check_folder
    config_plotly()

    print_ = True
    show = True
    error = 0

    try:
        fig = px.scatter(df,
                         x = var_x,
                         y = var_y,  # marker = dict(color='blue')
                         size = var_size,
                         color = var_color)

        fig.update_xaxes(title_text=var_x + ' ' + units_[var_x])
        fig.update_yaxes(title_text=var_y + ' ' + units_[var_y])
        fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
        fig.update_layout(title='Scatterplot - {}'.format(road))
        fig.update_layout(showlegend=True)
        # now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")

        if print_:
            dl_ia_cla_settings_check_folder('figures\\{}'.format(road))
            fig.write_html("figures\\{}\\scatterplot_{}_{}.html".format(road, var_x, var_y))
        if show:
            fig.show()

        return error


    except Exception as exception_msg:

        CLA_comm('(!) Error in plot_scatterplot: ' + str(exception_msg))
        error = 1
        return error





def plot_scatterplot_mpl(df, variable_x, variable_y, group_variable, road):
    """
    Scatterplot matplotlib individual with color by variable

    :param df:
    :param variable_x:
    :param variable_y:
    :param group_variable:
    :param road:
    :return:
    """

    import matplotlib.pyplot as plt
    from dl_ia_cla_settings import units_
    from datetime import datetime

    fig = plt.figure(figsize=(12, 8), dpi=150, facecolor='w', edgecolor='w')
    ax = fig.add_subplot(1, 1, 1)

    for var in df[group_variable].unique():
        df_aux = df[df[group_variable] == var]
        x = df_aux[variable_x].values
        y = df_aux[variable_y].values
        ax.scatter(x, y, label=var, alpha=0.8, edgecolor='k')

    fontsize_ = 12
    ax.set_xlabel(variable_x + ' ' + units_[variable_x], fontsize=fontsize_)
    ax.set_ylabel(variable_y + ' ' + units_[variable_y], fontsize=fontsize_)
    ax.tick_params(axis='both', labelsize=fontsize_)
    plt.legend()

    now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
    plt.title('{}'.format(road), fontsize=fontsize_)
    plt.show()

    plt.savefig('figures\\{}_scatterplot_{}_{}.png'.format(road, variable_y, variable_x),
                bbox_inches='tight',
                pad_inches=0)
    plt.close()




def plot_scatterplot_mpl2(df):
    """ This function do this-

    :param df:
    :param road:
    :param columns_:
    :return:
    """
    fontsize_ = 12

    import matplotlib.pyplot as plt
    from dl_ia_cla_settings import units_

    print_ = True
    show = True

    fig = plt.figure(figsize=(12, 6), dpi=90, facecolor='w', edgecolor='k')

    ax1 = fig.add_subplot(1, 1, 1)
    var_x = 'flow'
    var_y = 'speed'
    ax1.scatter(df[var_x], df[var_y], marker='o', s=3, c='b')
    ax1.set_ylabel(units_[var_y], fontsize=fontsize_)
    ax1.set_xlabel(units_[var_x], fontsize=fontsize_)

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.scatter(df['occupancy'], df['speed'])
    ax2.set_ylabel(units_['speed'], fontsize=fontsize_)
    ax2.set_xlabel(units_['occupancy'], fontsize=fontsize_)

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.scatter(df['flow'], df['occupancy'])
    ax3.set_ylabel(units_['speed'], fontsize=fontsize_)
    ax3.set_xlabel(units_['occupancy'], fontsize=fontsize_)

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.scatter(df['occupancy'], df['volume'])
    ax4.set_ylabel(units_['speed'], fontsize=fontsize_)
    ax4.set_xlabel(units_['occupancy'], fontsize=fontsize_)

    if show:
        plt.show()
    if print_:
        plt.savefig('figures\\plot_scatterplot_matplotlib.pdf', bbox_inches='tight', pad_inches=0.1)
    # plt.close()


def plot_scatterplot_mpl_multi(df, variable_x, variable_y, group_var, road):
    """
        Scatterplot matplotlib (4 plots)

    :param df:
    :param variable_x:
    :param variable_y:
    :param group_var:
    :param road:
    :return:
    """
    fontsize_ = 12
    marker_size_ = 10

    import matplotlib.pyplot as plt
    from dl_ia_cla_settings import units_
    from datetime import datetime
    import numpy as np

    fig = plt.figure(figsize=(12, 8), dpi=150, facecolor='w', edgecolor='w')

    for i in np.arange(0, len(variable_x)):
        ax1 = fig.add_subplot(2, 2, i + 1)
        sc1 = ax1.scatter(df[variable_x[i]],
                          df[variable_y[i]], marker='o', s=marker_size_, alpha=0.8, c=df[group_var])
        cbar1 = plt.colorbar(sc1)
        cbar1.set_label(group_var, rotation=270, fontsize=fontsize_)
        ax1.set_xlabel(variable_x[i] + ' ' + units_[variable_x[i]], fontsize=fontsize_)
        ax1.set_ylabel(variable_y[i] + ' ' + units_[variable_y[i]], fontsize=fontsize_)

        y_min = df[variable_y[i]].min()
        y_max = df[variable_y[i]].max()
        x_min = df[variable_x[i]].min()
        x_max = df[variable_x[i]].max()
        y_range = np.arange(y_min, y_max, round(y_max / 5))
        x_range = np.arange(x_min, x_max, round(x_max / 5))
        ax1.set_yticks(y_range)
        ax1.set_yticklabels(y_range)
        ax1.set_xticks(x_range)
        ax1.set_xticklabels(x_range)
        ax1.tick_params(axis='both', labelsize=fontsize_)

    now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
    plt.title('{}'.format(road), fontsize=fontsize_)
    plt.subplots_adjust(wspace=0.2, hspace=0.4)
    plt.show()
    plt.savefig('figures\\{}_scatterplot_multi_{}.png'.format(road, now),
                bbox_inches='tight', pad_inches=0)

    plt.close()




def plot_scatterplot_by_color(df, variable_x, variable_y, color_variable, road):
    """
    Statter plot plotly individual all variables segmented by schedule (!) ALL COMBINATIONS (6x7=42 plots)

    :param df:
    :param variable_x:
    :param variable_y:
    :param color_variable:
    :param road:
    :return:
    """
    import plotly.express as px
    from dl_ia_cla_settings import units_

    fig = px.scatter(df,
                     x=variable_x,
                     y=variable_y,
                     color=color_variable,
                     opacity=0.8,
                     hover_data=[color_variable])

    fig.update_xaxes(title_text=variable_x + ' ' + units_[variable_x])
    fig.update_yaxes(title_text=variable_y + ' ' + units_[variable_y])
    fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
    fig.update_layout(title='{} - Scatterplots'.format(road))
    fig.update_layout(showlegend=True)
    # now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
    # fig.show()
    fig.write_html(
        "figures\\{}_scatterplot_by_color_{}_{}.html".format(road, variable_x, variable_y))




def plot_scatterplot_by_color_multi(df, variable_x, variable_y, road):
    """
    Plotly scatterplot multiple with all values and segmented by schedule time



    :param df:
    :param variable_x:
    :param variable_y:
    :param road:
    :return:
    """

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import numpy as np

    fig = make_subplots(rows=2, cols=3)

    index = {'0-6': (1, 1),
             '7-12': (1, 2),
             '13-17': (1, 3),
             '18-21': (2, 1),
             '22-23': (2, 2),
             }

    colors_ = {'0-6': 'darkblue',
               '7-12': 'darkgreen',
               '13-17': 'darkred',
               '18-21': 'darkviolet',
               '22-23': 'darkgrey',
               }

    for schedule_ in df['schedule_def'].unique():
        df_aux = df.loc[df['schedule_def'] == schedule_]

        fig.add_trace(go.Scatter(x=df_aux[variable_x].values,
                                 y=df_aux[variable_y].values,
                                 name=schedule_,
                                 mode='markers',
                                 marker_color=colors_[schedule_]), row=index[schedule_][0], col=index[schedule_][1])

        x_axis = np.arange(0, int(df[variable_x].max()), int(df[variable_x].max() / 10))
        y_axis = np.arange(0, int(df[variable_y].max()), int(df[variable_y].max() / 10))
        fig.update_yaxes(title_text=variable_y,
                         tickfont=dict(size=14),
                         ticktext=y_axis,
                         tickvals=y_axis, row=index[schedule_][0], col=index[schedule_][1])

        fig.update_xaxes(title_text=variable_x,
                         tickfont=dict(size=14),
                         ticktext=x_axis,
                         tickvals=x_axis, row=index[schedule_][0], col=index[schedule_][1])

    fig.update_layout(showlegend=True)
    fig.update_layout(font=dict(family="Courier New, monospace", size=14, color="#7f7f7f"))
    fig.update_layout(title_text='{} - Scatterplots '.format(road))
    # now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
    # fig.show()
    fig.write_html("figures\\{}_scatterplot_multi_{}_{}_by_schedule_{}.html".format(road, variable_x, variable_y))




def plot_scatterplot_by_schedule_multi(df, variable_x, variable_y, road):
    """
    Plotly scatterplot individual with values segmented by schedule time

    :param df:
    :param variable_x:
    :param variable_y:
    :param road:
    :return:
    """

    from dl_ia_cla_settings import colors_, units_
    import plotly.graph_objects as go
    from datetime import datetime
    import numpy as np

    for schedule_ in df['schedule_def'].unique():
        df_aux = df.loc[df['schedule_def'] == schedule_]

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=df[variable_x].values,
                                 y=df[variable_y].values,
                                 name='All',
                                 mode='markers',
                                 opacity=0.9,
                                 marker_color='lightgrey'))

        fig.add_trace(go.Scatter(x=df_aux[variable_x].values,
                                 y=df_aux[variable_y].values,
                                 name=schedule_,
                                 mode='markers',
                                 marker_color=colors_[schedule_]))

        x_axis = np.arange(0, int(df[variable_x].max()), int(df[variable_x].max() / 10))
        y_axis = np.arange(0, int(df[variable_y].max()), int(df[variable_y].max() / 10))
        fig.update_yaxes(title_text=variable_y + units_[variable_y],
                         tickfont=dict(size=14),
                         ticktext=y_axis,
                         tickvals=y_axis)

        fig.update_xaxes(title_text=variable_x + units_[variable_x],
                         tickfont=dict(size=14),
                         ticktext=x_axis,
                         tickvals=x_axis)

        fig.update_layout(showlegend=True)
        fig.update_layout(font=dict(family="Courier New, monospace", size=14, color="#7f7f7f"))
        fig.update_layout(title_text='{} - Scatterplot '.format(road))
        now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
        fig.write_html(
            "figures\\{}_scatterplot_multi_{}_{}_schedule_{}_{}.html".format(road, variable_x, variable_y,
                                                                             schedule_, now))
        # fig.show()



def plot_scatterplot_with_histogram(df, variable_y, variable_x, grouping_variable, road):
    """

    :param variable_y:
    :param variable_x:
    :param grouping_variable:
    :param road:
    :return:
    """
    ""

    import matplotlib.pyplot as plt
    from dl_ia_cla_settings import units_
    import numpy as np

    # definitions for the axes
    left, width = 0.1, 0.4
    bottom, height = 0.1, 0.4
    spacing = 0.005

    fontsize_ = 16

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]

    # start with a square Figure
    fig = plt.figure(figsize=(12, 8), dpi=150, facecolor='w', edgecolor='w')

    ax = fig.add_axes(rect_scatter)
    ax_histx = fig.add_axes(rect_histx, sharex=ax)
    ax_histy = fig.add_axes(rect_histy, sharey=ax)

    x = df[variable_x].values
    y = df[variable_y].values

    color_ = df[grouping_variable].values

    # now determine nice limits by hand:
    binwidth = 0.5
    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    lim = (int(xymax / binwidth) + 1) * binwidth
    bins = np.arange(0, lim + binwidth, binwidth)

    ax_histx.hist(x, bins=bins)
    ax_histy.hist(y, bins=bins, orientation='horizontal')
    ax_histx.axis('off')
    ax_histy.axis('off')

    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    sc = ax.scatter(x, y, c=color_)
    cbar = plt.colorbar(sc)
    cbar.set_label(grouping_variable, rotation=270, fontsize=fontsize_)

    xmax = int(df[variable_x].max())
    ymax = int(df[variable_y].max())

    ax.set_xlabel(variable_x + ' ' + units_[variable_x], fontsize=fontsize_)
    ax.set_ylabel(variable_y + ' ' + units_[variable_y], fontsize=fontsize_)

    ax.tick_params(axis='both', labelsize=fontsize_)
    ax.set_xticks = list(np.arange(0, xmax, int(xmax / 10)))
    ax.set_yticks = list(np.arange(0, ymax, int(ymax / 10)))
    ax.set_xticklabels = list(np.arange(0, xmax, int(xmax / 10)))
    ax.set_yticklabels = list(np.arange(0, ymax, int(ymax / 10)))

    # now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")

    plt.title('{}'.format(road), fontsize=fontsize_)
    plt.show()
    plt.savefig('figures\\{}_scatterplot_with_histogram_{}_{}.png'.format(road, variable_y, variable_x),
                bbox_inches='tight', pad_inches=0)
    plt.close()





"""  ==================== Time series =======================
"""



def plot_time_series_mpl(df, start_date, end_date, road, variable):
    """
    Plot time series with matplotlib, one variable per plot,
    segmented by section, for a specific data range
    The plot is saved in png format

    :param df:
    :param start_date:
    :param end_date:
    :param road:
    :return:
    """
    fontsize_ = 18

    import matplotlib.pyplot as plt
    from dl_ia_cla_settings import units_

    fig = plt.figure(figsize=(25, 10), dpi=150, facecolor='w', edgecolor='w')
    ax = fig.add_subplot(1, 1, 1)

    for section_ in df['sub_section'].unique():
        df_aux = df[df['sub_section'] == section_]
        y = df_aux.loc[start_date:end_date, [variable]]
        ax.plot(y, '-o', label='section:{}'.format(section_))

    ax.set_xlabel('time (15 min)', fontsize=fontsize_)
    ax.set_ylabel(variable + units_[variable], fontsize=fontsize_)
    ax.tick_params(axis='both', labelsize=fontsize_)
    ax.legend(fontsize=fontsize_)
    ax.set_title('{} - Time Series - {} to {}'.format(road, start_date, end_date), fontsize=fontsize_)
    # now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
    plt.subplots_adjust(wspace=0.2, hspace=0.4)
    plt.show()
    plt.savefig('figures\\{}_timeseries_{}.png'.format(road, variable), bbox_inches='tight', pad_inches=0)
    plt.close()



def plot_timeseries_with_slider(df, var_x, var_y, road):
    """ Plot simple time series with time slider

    :param df: dataframe with timestamp index and variable
    :param variable: name of the variable
    :param road: name of the road/asset
    :return:
    """

    import plotly.graph_objects as go
    from dl_ia_cla_settings import units_, CLA_comm, dl_ia_cla_settings_check_folder

    print_ = True
    show = True
    error = 0

    try:
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=df[var_x],
                                 y=df[var_y],
                                 mode='markers+lines',
                                 marker=dict(color='red'),
                                 name='{}'.format(var_y)))

        # Add range slider
        fig.update_layout(
            xaxis=go.layout.XAxis(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1,
                             label="1 day",
                             step="day",
                             stepmode="backward"),
                        dict(count=3,
                             label="3 days",
                             step="day",
                             stepmode="backward"),
                        dict(count=7,
                             label="1 week",
                             step="day",
                             stepmode="backward"),
                    ])
                ),
                rangeslider=dict(
                    visible=True
                ),
                type="date"
            )
        )

        fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
        fig.update_layout(showlegend=True)
        fig.update_yaxes(title_text=var_y + ' ' + units_[var_y])
        fig.update_xaxes(title_text='Time')
        fig.update_layout(title='Timeseries')

        if print_:
            dl_ia_cla_settings_check_folder('figures\\{}'.format(road))
            fig.write_html("figures\\{}\\timeseries_with_slider_{}.html".format(road, var_y))
        if show:
            fig.show()

    except Exception as exception_msg:
        CLA_comm('(!) Error in plot_timeseries_with_slider: ' + str(exception_msg))
        error = 1
        return error




def plot_timeseries(df, var_x, var_y, road):
    """ Plot simple time series with time slider

    :param df: dataframe with timestamp index and variable
    :param variable: name of the variable
    :param road: name of the road/asset
    :return:
    """

    import plotly.graph_objects as go
    from dl_ia_cla_settings import units_, CLA_comm, dl_ia_cla_settings_check_folder

    print_ = True
    show = True
    error = 0

    try:
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=df[var_x],
                                 y=df[var_y],
                                 mode='markers+lines',
                                 marker=dict(color='red'),
                                 name='{}'.format(var_y)))


        fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
        fig.update_layout(showlegend=True)
        fig.update_yaxes(title_text=variable + ' ' + units_[var_y])
        fig.update_xaxes(title_text='Time')
        fig.update_layout(title='Timeseries - {} - {}'.format(road, var_y))

        if print_:
            dl_ia_cla_settings_check_folder('figures')
            fig.write_html("figures\\{}\\timeseries_with_slider_{}.html".format(road, var_y))
        if show:
            fig.show()

    except Exception as exception_msg:

        CLA_comm('(!) Error in plot_timeseries: ' + str(exception_msg))
        error = 1
        return error


def plot_time_Series(df, variable, road):
    """ Produce the time series plot

    :param df: data frame with the variable to plot, the time index, and the columns of section and lanes
    :param variable: name of the road traffic variable to plot
    """

    import plotly.graph_objects as go
    from dl_ia_cla_settings import units_
    import numpy as np

    fig = go.Figure()

    for section_ in np.sort(df['section'].unique()):
        df_aux = df[df['section'] == section_]
        for lane_ in np.sort(df_aux['lanes'].unique()):
            df_aux2 = df_aux[df_aux['lanes'] == lane_]
            df_aux2.sort_values(by=['time'], inplace=True)

            fig.add_trace(go.Scatter(x=df_aux2['time'],
                                     y=df_aux2[variable],
                                     line=dict(width=2, dash='dot'),
                                     mode='markers+lines',
                                     name=variable + ' ' + units_[variable] + '{}'.format(section_) + '{}'.format(lane_)))

    fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
    fig.update_layout(
        title='{} - timeseries '.format(road),
        xaxis_title='Time',
        yaxis_title='',
        showlegend=True
    )

    # Add range slider
    fig.update_layout(
        xaxis=go.layout.XAxis(
            rangeselector=dict(
                buttons=list([
                    dict(count=1,
                         label="1 month",
                         step="month",
                         stepmode="backward"),
                    dict(count=6,
                         label="6 month",
                         step="month",
                         stepmode="backward"),
                    dict(count=7,
                         label="1 week",
                         step="day",
                         stepmode="backward"),
                    dict(count=1,
                         label="1 year",
                         step="year",
                         stepmode="backward"),
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            type="date"
        )
    )
    fig.show()
    fig.write_html("figures\\{}\\plot_time_Series_{}.html".format(road, road))


def time_series_comparison(time_index, y1, y2, label1, label2):
    """ Plot two time series with the same time index

    :param time_index:
    :param y1:
    :param y2:
    :param label1:
    :param label2:
    :return:
    """

    import plotly.graph_objects as go
    from dl_ia_cla_settings import units_, config_plotly
    config_plotly()

    print_ = True
    show = True
    title = 'time_series_comparison'

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_index,
                             y=y1,
                             mode='markers',
                             marker=dict(color='red'),
                             name='{}'.format(label1)))

    fig.add_trace(go.Scatter(x=time_index,
                             y=y2,
                             mode='markers+lines',
                             marker=dict(color='blue'),
                             name='{}'.format(label2)))

    fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
    fig.update_layout(showlegend=True)
    fig.update_yaxes(title_text=label1 + ' '+ units_[label1])
    fig.update_xaxes(title_text='Time')
    fig.update_layout(title='Time series comparison')

    # Add range slider
    fig.update_layout(
        xaxis=go.layout.XAxis(
            rangeselector=dict(
                buttons=list([
                    dict(count=1,
                         label="1 day",
                         step="day",
                         stepmode="backward"),
                    dict(count=3,
                         label="3 day",
                         step="day",
                         stepmode="backward"),
                    dict(count=7,
                         label="1 week",
                         step="day",
                         stepmode="backward"),
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            type="date"
        )
    )
    if print_:
        fig.write_html("figures\\{}_{}_{}.html".format(title, label1, label2))
    if show:
        fig.show()





def plot_time_Series_slider(df, variable, road):
    """ plot a time series with a time slider

    :param df: dataframe with the data
    :return:
    """
    from dl_ia_cla_settings import units_
    import plotly.graph_objects as go


    subsection_ = df['sub_section'].unique()[0]
    df_aux = df[df['sub_section'] == subsection_]

    lane_ = df_aux[df_aux['sub_section'] == subsection_]['lanes'].unique()[0]
    section_ = df_aux[df_aux['sub_section'] == subsection_]['section'].unique()[0]

    df_aux['date'] = df_aux.index


    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df_aux['date'],
                             y=df_aux[variable],
                             name=variable + ' ' + units_[variable]))

    fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
    fig.update_layout(
        title='{} - timeseries - Section:{}-Lane:{}'.format(road, int(section_), int(lane_)),
        xaxis_title='Time',
        yaxis_title='',
        showlegend=True
    )

    # Add range slider
    fig.update_layout(
        xaxis=go.layout.XAxis(
            rangeselector=dict(
                buttons=list([
                    dict(count=1,
                         label="1 month",
                         step="month",
                         stepmode="backward"),
                    dict(count=6,
                         label="6 month",
                         step="month",
                         stepmode="backward"),
                    dict(count=7,
                         label="1 week",
                         step="day",
                         stepmode="backward"),
                    dict(count=1,
                         label="1 year",
                         step="year",
                         stepmode="backward"),
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            type="date"
        )
    )
    fig.show()


def plot_time_series_by_date(df, variable, start_date, end_date):
    """

    :param df:
    :param variable:
    :param start_date:
    :param end_date:
    :return:
    """

    from dl_ia_cla_settings import units_
    import plotly.graph_objects as go

    # filter by dates
    df_aux = df[(df.index > start_date) & (df.index < end_date)]

    # filter by sectopm
    section_ = 1
    df_aux = df_aux[(df_aux['section'] == section_)]

    # filter by lanes
    df_aux = df_aux[(df['lanes'] == 1)]

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df_aux[variable].index,
                             y=df_aux[variable],
                             mode='markers',
                             name=variable + ' ' + units_[variable]))

    fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
    fig.update_layout(showlegend=True)
    fig.update_yaxes(title_text=variable + ' ' + units_[variable])
    fig.update_xaxes(title_text='Time (15 min)')
    fig.update_layout(title='{}  - Time series dates: {} to {} section:{}'.format(road, start_date, end_date, section_))

    # Add range slider
    fig.update_layout(
        xaxis=go.layout.XAxis(
            rangeselector=dict(
                buttons=list([
                    dict(count=1,
                         label="1 day",
                         step="day",
                         stepmode="backward"),
                    dict(count=3,
                         label="3 days",
                         step="day",
                         stepmode="backward"),
                    dict(count=7,
                         label="1 week",
                         step="day",
                         stepmode="backward")
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            type="date"
        )
    )
    # now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
    fig.write_html("figures\\{}_timeseries_{}.html".format(road, variable))
    # fig.show()



def plot_time_series_segmented(df, start_date, end_date, road, variable, group_variable):
    """
        definition

    :param df:
    :param start_date:
    :param end_date:
    :param road:
    :return:
    """

    import plotly.graph_objects as go
    from dl_ia_cla_settings import units_

    fig = go.Figure()

    for group in df[group_variable].unique():
        df_aux = df[df[group_variable] == group]
        df_aux = df_aux.loc[start_date:end_date]

        t = df_aux.index
        y = df_aux[variable]

        fig.add_trace(go.Scatter(x=t,
                                 y=y,
                                 mode='markers',
                                 name=group,
                                 opacity=0.6))

    # fig.update_xaxes(title_text='Time (15 min)')
    fig.update_yaxes(title_text=variable + ' ' + units_[variable])

    fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
    fig.update_layout(title='{}  - Time series - {} to {}'.format(road, start_date, end_date))
    fig.update_layout(showlegend=True)

    # Add range slider
    fig.update_layout(
        xaxis=go.layout.XAxis(
            rangeselector=dict(
                buttons=list([
                    dict(count=1,
                         label="1 day",
                         step="day",
                         stepmode="backward"),
                    dict(count=3,
                         label="3 days",
                         step="day",
                         stepmode="backward"),
                    dict(count=7,
                         label="1 week",
                         step="day",
                         stepmode="backward")
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            type="date"
        )
    )

    # now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
    fig.write_html("figures\\{}_timeseries_segmented_{}.html".format(road, variable))
    # fig.show()




def plot_time_series_by_section(df, road, variable, start_date, end_date):
    """
        definition

    :param df:
    :param start_date:
    :param end_date:
    :param road:
    :return:
    """

    df_aux = df[(df.index > start_date) & (df.index < end_date)]
    section_ = df_aux['section'].unique()[0]

    df_aux = df_aux[df_aux['section'] == section_]


    import plotly.graph_objects as go
    from dl_ia_cla_settings import units_

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df_aux[variable].index,
                             y=df_aux[variable],
                             mode='markers',
                             name='{}-Section:{}'.format(variable, str(section_))))

    fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
    fig.update_layout(showlegend=True)
    fig.update_yaxes(title_text=variable + ' ' + units_[variable])
    fig.update_xaxes(title_text='Time (15 min)')
    fig.update_layout(title='{}  - Time series'.format(road))

    # Add range slider
    fig.update_layout(
        xaxis=go.layout.XAxis(
            rangeselector=dict(
                buttons=list([
                    dict(count=1,
                         label="1 day",
                         step="day",
                         stepmode="backward"),
                    dict(count=3,
                         label="3 days",
                         step="day",
                         stepmode="backward"),
                    dict(count=7,
                         label="1 week",
                         step="day",
                         stepmode="backward")
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            type="date"
        )
    )
    # now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
    fig.write_html("figures\\{}_timeseries_by_section_{}.html".format(road, variable))
    # fig.show()



def plot_time_series_by_speed_bandwith(df, road, variable, start_date, end_date):
    """
        definition

    :param df:
    :param start_date:
    :param end_date:
    :param road:
    :return:
    """

    df_aux = df[(df.index > start_date) & (df.index < end_date)]

    # Plot all sections

    df_aux1 = df_aux[(df_aux['speed'] < 80) & (df_aux['speed'] >= 70)]
    df_aux2 = df_aux[(df_aux['speed'] < 70) & (df_aux['speed'] >= 50)]

    print(df_aux1[variable].describe())
    print(df_aux2[variable].describe())

    t1 = df_aux1[variable].index
    t2 = df_aux2[variable].index

    y1 = df_aux1[variable]
    y2 = df_aux2[variable]

    import plotly.graph_objects as go
    from dl_ia_cla_settings import units_

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=t1,
                             y=y1,
                             mode='markers',
                             marker=dict(color='blue'),
                             name='{} with speed in (80-70)'.format(variable)))

    fig.add_trace(go.Scatter(x=t2,
                             y=y2,
                             mode='markers',
                             marker=dict(color='gray'),
                             name='{} with speed in (70-50)'.format(variable)))

    fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
    fig.update_layout(showlegend=True)
    fig.update_yaxes(title_text=variable + ' ' + units_[variable])
    fig.update_xaxes(title_text='Time (15 min)')
    fig.update_layout(title='{}  - Time series '.format(road))

    # Add range slider
    fig.update_layout(
        xaxis=go.layout.XAxis(
            rangeselector=dict(
                buttons=list([
                    dict(count=1,
                         label="1 day",
                         step="day",
                         stepmode="backward"),
                    dict(count=3,
                         label="3 days",
                         step="day",
                         stepmode="backward"),
                    dict(count=7,
                         label="1 week",
                         step="day",
                         stepmode="backward")
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            type="date"
        )
    )
    # now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
    fig.write_html("figures\\{}_timeseries_by_speed_bandwith_{}.html".format(road, variable))
    fig.show()





def plot_time_series_congestion_by_day_by_lane(df, df_congestions):
    """
        definition

    :param df:
    :param start_date:
    :param end_date:
    :param road:
    :return:
    """



    import plotly.graph_objects as go
    from dl_ia_cla_settings import create_time_array

    t = create_time_array('00:00:00', '23:59:59', '15min')

    fig = go.Figure()

    for day in ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']:
        y_congestion = df_congestions['Congestion_{}'.format(day)]

        fig.add_trace(go.Scatter(x=t,
                                 y=y_congestion,
                                 mode='markers+lines',
                                 name='({}) Avg. Congestion (%)'.format(day)))

        # to plot volume and gap

        df_aux = df[df['weekday'] == dict_days[day]]

        subsection_ = df['sub_section'].unique()[0]

        df_aux = df_aux[df_aux['sub_section'] == subsection_]

        lane_ = df_aux['lanes'].unique()
        section_ = df_aux['section'].unique()[0]

        y_volume = df_aux.groupby(['hour', 'minute'])['volume'].mean()
        fig.add_trace(go.Scatter(x=t,
                                 y=y_volume.values,
                                 mode='markers+lines',
                                 name='({}) Avg. Volume (Veh./15min)'.format(day)))

        if 1 == 1:
            y_speed = df_aux.groupby(['hour', 'minute'])['speed'].mean()
            fig.add_trace(go.Scatter(x=t,
                                     y=y_speed.values,
                                     mode='markers+lines',
                                     name='({}) Avg. speed (km/h)'.format(day)))

        if 1 == 0:
            y_gap = df_aux.groupby(['hour', 'minute'])['gap'].mean()
            fig.add_trace(go.Scatter(x=t,
                                     y=y_gap.values * 0.1,
                                     mode='markers+lines',
                                     name='({}) Avg. Gap (1/10 sec)'.format(day)))

    fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
    fig.update_layout(showlegend=True)
    fig.update_yaxes(title_text=' ')
    fig.update_xaxes(title_text='Time (15 min)')
    fig.update_layout(title='{} - Section:{}  -Lane:{}  '.format(road, section_, int(lane_)))

    # now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
    fig.write_html("figures\\{}_congestion.html".format(road))
    fig.show()



def plot_time_series_congestion_by_day_by_section(df, df_congestions):
    """
        definition

    :param df:
    :param start_date:
    :param end_date:
    :param road:
    :return:
    """



    import plotly.graph_objects as go
    from dl_ia_cla_settings import create_time_array

    t = create_time_array('00:00:00', '23:59:59', '15min')

    fig = go.Figure()

    for day in ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']:
        y_congestion = df_congestions['Congestion_{}'.format(day)]

        fig.add_trace(go.Scatter(x=t,
                                 y=y_congestion,
                                 mode='markers+lines',
                                 name='({}) Avg. Congestion (%)'.format(day)))

        # to plot volume and gap

        df_aux = df[df['weekday'] == dict_days[day]]

        section_ = df['section'].unique()[0]

        df_aux = df_aux[df_aux['section'] == section_]

        y_volume = df_aux.groupby(['hour', 'minute'])['volume'].mean()
        fig.add_trace(go.Scatter(x=t,
                                 y=y_volume.values,
                                 mode='markers+lines',
                                 name='({}) Avg. Volume (Veh./15min)'.format(day)))

        if 1 == 1:
            y_speed = df_aux.groupby(['hour', 'minute'])['speed'].mean()
            fig.add_trace(go.Scatter(x=t,
                                     y=y_speed.values,
                                     mode='markers+lines',
                                     name='({}) Avg. speed (km/h)'.format(day)))

        if 1 == 0:
            y_gap = df_aux.groupby(['hour', 'minute'])['gap'].mean()
            fig.add_trace(go.Scatter(x=t,
                                     y=y_gap.values * 0.1,
                                     mode='markers+lines',
                                     name='({}) Avg. Gap (1/10 sec)'.format(day)))

    fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
    fig.update_layout(showlegend=True)
    fig.update_yaxes(title_text=' ')
    fig.update_xaxes(title_text='Time (15 min)')
    fig.update_layout(title='{} - Section:{} '.format(road, section_))

    # now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
    fig.write_html("figures\\{}_congestion_by_day_by_section_{}.html".format(road, section_))
    fig.show()



def plot_time_series_congestion_total_by_section(df, df_congestions):
    """
        definition

    :param df:
    :param start_date:
    :param end_date:
    :param road:
    :return:
    """



    import plotly.graph_objects as go
    import numpy as np
    from dl_ia_cla_settings import create_time_array

    t = create_time_array('00:00:00', '23:59:59', '15min')

    fig = go.Figure()

    Y_congestion = []
    for i in range(0, len(df_congestions)):
        Y_congestion.append(df_congestions.iloc[i, :].mean())

    fig.add_trace(go.Scatter(x=t,
                             y=Y_congestion,
                             mode='markers+lines',
                             name='Avg. Congestion Brisbane 2019 (%)'))

    # to plot volume and gap

    color_1 = {1: 'brown', 2: 'crimson', 3: 'darkred', 4: 'tomato'}
    color_2 = {1: 'sandybrown', 2: 'salmon', 3: 'orange', 4: 'goldenrod'}

    for section_ in np.sort(df['section'].unique()):
        df_aux = df[df['section'] == section_]

        y_volume = df_aux.groupby(['hour', 'minute'])['volume'].mean()

        fig.add_trace(go.Scatter(x=t,
                                 y=y_volume.values,
                                 mode='markers+lines',
                                 marker=dict(color=color_1[section_]),
                                 name='Section:{} Avg. Volume (Veh./15min)'.format(section_)))

        y_speed = df_aux.groupby(['hour', 'minute'])['speed'].mean()
        fig.add_trace(go.Scatter(x=t,
                                 y=y_speed.values,
                                 mode='markers+lines',
                                 marker=dict(color=color_2[section_]),
                                 name='Section:{} Avg. speed (km/h)'.format(section_)))

    fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
    fig.update_layout(showlegend=True)
    fig.update_yaxes(title_text=' ')
    fig.update_xaxes(title_text='Time (15 min)')
    fig.update_layout(title='{} -'.format(road))

    # now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
    fig.write_html("figures\\{}_congestion_total_by_section.html".format(road))
    fig.show()



def plot_time_series_congestion_total_by_lane(df, df_congestions):
    """
        definition

    :param df:
    :param start_date:
    :param end_date:
    :param road:
    :return:
    """



    import plotly.graph_objects as go
    import numpy as np
    from dl_ia_cla_settings import create_time_array

    t = create_time_array('00:00:00', '23:59:59', '15min')

    fig = go.Figure()

    Y_congestion = []
    for i in range(0, len(df_congestions)):
        Y_congestion.append(df_congestions.iloc[i, :].mean())

    fig.add_trace(go.Scatter(x=t,
                             y=Y_congestion,
                             mode='markers+lines',
                             name='Avg. Congestion Brisbane 2019 (%)'))

    # to plot volume and gap

    color_1 = {1: 'brown', 2: 'crimson', 3: 'darkred', 4: 'tomato'}
    color_2 = {1: 'sandybrown', 2: 'salmon', 3: 'orange', 4: 'goldenrod'}

    for subsection_ in np.sort(df['sub_section'].unique()):
        df_aux = df[df['sub_section'] == subsection_]

        lane_ = df_aux['lanes'].unique()
        section_ = df_aux['section'].unique()[0]

        y_volume = df_aux.groupby(['hour', 'minute'])['volume'].mean()

        fig.add_trace(go.Scatter(x=t,
                                 y=y_volume.values,
                                 mode='markers+lines',
                                 marker=dict(color=color_1[section_]),
                                 name='Section:{} Lane:{} Avg. Volume (Veh./15min)'.format(section_, int(lane_))))

        y_speed = df_aux.groupby(['hour', 'minute'])['speed'].mean()
        fig.add_trace(go.Scatter(x=t,
                                 y=y_speed.values,
                                 mode='markers+lines',
                                 marker=dict(color=color_2[section_]),
                                 name='Section:{} Lane:{} Avg. speed (km/h)'.format(section_, int(lane_))))

    fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
    fig.update_layout(showlegend=True)
    fig.update_yaxes(title_text=' ')
    fig.update_xaxes(title_text='Time (15 min)')
    fig.update_layout(title='{} -'.format(road))

    # now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
    fig.write_html("figures\\{}_congestion_total_by_lane.html".format(road))
    fig.show()




def plot_time_series_congestion_total(df, df_congestions):
    """
        definition

    :param df:
    :param start_date:
    :param end_date:
    :param road:
    :return:
    """



    import plotly.graph_objects as go
    from dl_ia_cla_settings import create_time_array

    t = create_time_array('00:00:00', '23:59:59', '15min')

    fig = go.Figure()

    Y_congestion = []
    for i in range(0, len(df_congestions)):
        Y_congestion.append(df_congestions.iloc[i, :].mean())

    fig.add_trace(go.Scatter(x=t,
                             y=Y_congestion,
                             mode='markers+lines',
                             name='Avg. Congestion Brisbane 2019 (%)'))

    y_volume = df.groupby(['hour', 'minute'])['volume'].mean()

    fig.add_trace(go.Scatter(x=t,
                             y=y_volume.values,
                             mode='markers+lines',
                             name='Avg. Volume (Veh./15min)'))

    y_speed = df.groupby(['hour', 'minute'])['speed'].mean()
    fig.add_trace(go.Scatter(x=t,
                             y=y_speed.values,
                             mode='markers+lines',
                             name='Avg. speed (km/h)'))

    fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
    fig.update_layout(showlegend=True)
    fig.update_yaxes(title_text=' ')
    fig.update_xaxes(title_text='Time (15 min)')
    fig.update_layout(title='{}  '.format(road))

    fig.write_html("figures\\{}_congestion_total.html".format(road))
    fig.show()




def plot_time_series_smooth(df, road, variable, start_date, end_date, smooth):
    """
        definition

    :param df:
    :param start_date:
    :param end_date:
    :param road:
    :return:
    """
    from scipy.signal import savgol_filter
    import numpy as np
    import plotly.graph_objects as go
    from dl_ia_cla_settings import units_

    df_aux = df[(df.index > start_date) & (df.index < end_date)]

    subsection_ = df_aux['sub_section'].unique()[0]
    df_aux = df_aux[df_aux['sub_section'] == subsection_]

    print(df_aux[variable].describe())

    t = df_aux[variable].index
    y = df_aux[variable]

    if smooth == 0:
        print('no smooth')

    if smooth == 1:
        smooth_label = 'Savgol filter'

        yhat = savgol_filter(y, 15, 2)  # window size 51, polynomial order 3

    if smooth == 2:
        smooth_label = 'Convolute'

        def smooth(y, box_pts):
            box = np.ones(box_pts) / box_pts
            y_smooth = np.convolve(y, box, mode='same')
            return y_smooth

        yhat = smooth(y, 3)

    if smooth == 3:
        smooth_label = 'Rolling'
        yhat = y.rolling(window=5, min_periods=5, center=True).median()

    if smooth == 4:
        smooth_label = 'Interpolate'
        from scipy.interpolate import interp1d

        x = np.arange(0, len(t))
        xx = np.linspace(x, x, 1000)

        # interpolate + smooth
        itp = interp1d(t, y.values, kind='linear')

        window_size, poly_order = 101, 3
        yhat = savgol_filter(itp(xx), window_size, poly_order)

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=t,
                             y=y,
                             mode='markers',
                             marker=dict(color='blue'),
                             name='{} data'.format(variable)))
    if smooth != 0:
        fig.add_trace(go.Scatter(x=t,
                                 y=yhat,
                                 mode='lines',
                                 marker=dict(color='gray'),
                                 name='{} smooth {}'.format(variable, smooth_label)))

    fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
    fig.update_layout(showlegend=True)
    fig.update_yaxes(title_text=variable + ' ' + units_[variable])
    fig.update_xaxes(title_text='Time (15 min)')
    fig.update_layout(title='{}  - Time series'.format(road))

    # Add range slider
    fig.update_layout(
        xaxis=go.layout.XAxis(
            rangeselector=dict(
                buttons=list([
                    dict(count=1,
                         label="1 day",
                         step="day",
                         stepmode="backward"),
                    dict(count=3,
                         label="3 days",
                         step="day",
                         stepmode="backward"),
                    dict(count=7,
                         label="1 week",
                         step="day",
                         stepmode="backward")
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            type="date"
        )
    )
    # now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
    fig.write_html("figures\\{}_timeseries_smooth_{}.html".format(road, variable))
    fig.show()




""" ===================== Reference Pattern ========================
"""



def plot_reference_daily_pattern(df, road, variable):
    """ This function do this-

    :param df:
    :param road:
    :param columns_:
    :return:
    """
    import plotly.graph_objects as go
    from dl_ia_cla_settings import units_
    from dl_ia_cla_settings import create_time_array

    t = create_time_array('00:00:00', '23:59:59', '15min')

    df_mean = df.groupby(by=['hour', 'minute'])[variable].mean()

    df_std = df.groupby(by=['hour', 'minute'])[variable].std()


    y_mean = df_mean.values

    y_upper_std = df_mean.values + df_std.values
    y_lower_std = df_mean.values - df_std.values
    y_lower_std[y_lower_std < 0] = 0

    y_upper_std2 = df_mean.values + 2 * df_std.values
    y_lower_std2 = df_mean.values - 2 * df_std.values
    y_lower_std2[y_lower_std2 < 0] = 0

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t,
                             y=y_mean,
                             mode='markers+lines',
                             name='Mean Value',
                             marker=dict(color='blue'),
                             opacity=0.6))

    fig.add_trace(go.Scatter(x=t,
                             y=y_upper_std,
                             name='Mean Value + 1 Standard deviation',
                             marker=dict(color='red'),
                             mode='lines',
                             opacity=0.6))

    fig.add_trace(go.Scatter(x=t,
                             y=y_lower_std,
                             name='Mean Value - 1 Standard deviation',
                             marker=dict(color='red'),
                             mode='lines',
                             opacity=0.6))

    fig.add_trace(go.Scatter(x=t,
                             y=y_upper_std2,
                             name='Mean Value + 2 Standard deviation',
                             marker=dict(color='darkred'),
                             mode='lines',
                             opacity=0.6))

    fig.add_trace(go.Scatter(x=t,
                             y=y_lower_std2,
                             name='Mean Value - 2 Standard deviation',
                             marker=dict(color='darkred'),
                             mode='lines',
                             opacity=0.6))

    index_ = df.index.dayofyear.unique()

    if 1 == 0:
        for days_ in index_:
            df_aux = df[df.index.dayofyear == days_]

            df_aux = df_aux[df_aux['section'] == df_aux['section'].unique()[0]]

            fig.add_trace(go.Scatter(x=t,
                                     y=df_aux[variable].values,

                                     name='Data',
                                     marker=dict(color='gray'),
                                     mode='markers',
                                     opacity=0.6))

    fig.update_yaxes(title_text=variable + ' ' + units_[variable])
    fig.update_xaxes(title_text='time (15 min)')

    fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
    fig.update_layout(title='{}  - Time series - (Reference Pattern)'.format(road))
    fig.update_layout(showlegend=True)
    # now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
    fig.write_html("figures\\{}_reference_pattern_{}.html".format(road, variable))
    fig.show()




def plot_reference_daily_pattern_by_weekend(df, road, variable):
    """ This function do this-

    :param df:
    :param road:
    :param columns_:
    :return:
    """


    import plotly.graph_objects as go
    from datetime import datetime
    from dl_ia_cla_settings import create_time_array

    t = create_time_array('00:00:00', '23:59:59', '15min')

    fig = go.Figure()

    weekend_dict = {1: 'weekend',
                    0: 'weekday'}

    for week_day_ in df['weekend'].unique():
        df_aux = df[df['weekend'] == week_day_]

        df_mean = df_aux.groupby(by=['hour', 'minute'])[variable].mean()
        df_std = df_aux.groupby(by=['hour', 'minute'])[variable].std()
        y_mean = df_mean.values
        y_upper_std1 = df_mean.values + df_std.values
        y_lower_std1 = df_mean.values - df_std.values
        y_lower_std1[y_lower_std1 < 0] = 0

        y_upper_std2 = df_mean.values + 2 * df_std.values
        y_lower_std2 = df_mean.values - 2 * df_std.values
        y_lower_std2[y_lower_std2 < 0] = 0

        fig.add_trace(go.Scatter(x=t,
                                 y=y_mean,
                                 mode='markers+lines',
                                 marker=dict(color='darkred'),
                                 name='Mean Value ({})'.format(weekend_dict[week_day_]),
                                 opacity=0.6))

        fig.add_trace(go.Scatter(x=t,
                                 y=y_upper_std1,
                                 marker=dict(color='blue'),
                                 name='Mean Value + 1 Standard deviation ({})'.format(weekend_dict[week_day_]),
                                 mode='lines',
                                 opacity=0.6))

        fig.add_trace(go.Scatter(x=t,
                                 y=y_lower_std1,
                                 marker=dict(color='blue'),
                                 name='Mean Value - 1 Standard deviation ({})'.format(weekend_dict[week_day_]),
                                 mode='lines',
                                 opacity=0.6))

        fig.add_trace(go.Scatter(x=t,
                                 y=y_upper_std2,
                                 marker=dict(color='darkblue'),
                                 name='Mean Value + 2 Standard deviation ({})'.format(weekend_dict[week_day_]),
                                 mode='lines',
                                 opacity=0.6))

        fig.add_trace(go.Scatter(x=t,
                                 y=y_lower_std2,
                                 marker=dict(color='darkblue'),
                                 name='Mean Value - 2 Standard deviation ({})'.format(weekend_dict[week_day_]),
                                 mode='lines',
                                 opacity=0.6))

    fig.update_yaxes(title_text=variable + ' ' + units_[variable])
    fig.update_xaxes(title_text='time (15 min)')

    fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
    fig.update_layout(title='{}  - Time series - (Reference Pattern)'.format(road))
    fig.update_layout(showlegend=True)
    now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
    # fig.write_html("figures\\{}_timeseries_{}_{}.html".format(road, variable, now, ))
    fig.show()





def plot_reference_daily_pattern_by_speed_bandwith(df, road, variable):
    """ This function do this-

    :param df:
    :param road:
    :param columns_:
    :return:
    """

    import plotly.graph_objects as go
    from dl_ia_cla_settings import units_
    from dl_ia_cla_settings import create_time_array

    t = create_time_array('00:00:00', '23:59:59', '15min')

    fig = go.Figure()

    df_aux = df

    # Plot all sections
    colors_ = [('red', 'darkred'), ('blue', 'darkblue')]

    speed_limits = [(80, 70), (70, 60)]


    for i in range(0, len(speed_limits)):

        # print(i)

        df_aux1 = df_aux[(df_aux['speed'] < speed_limits[i][0]) & (df_aux['speed'] >= speed_limits[i][1])]

        print(df_aux1.describe())

        df_mean = df_aux1.groupby(by=['hour', 'minute'])[variable].mean()
        df_std = df_aux1.groupby(by=['hour', 'minute'])[variable].std()
        y_mean = df_mean.values
        y_upper_1std = df_mean.values + df_std.values
        y_lower_1std = df_mean.values - df_std.values
        y_upper_2std = df_mean.values + 2 * df_std.values
        y_lower_2std = df_mean.values - 2 * df_std.values
        y_lower_1std[y_lower_1std < 0] = 0
        y_lower_2std[y_lower_2std < 0] = 0

        fig.add_trace(go.Scatter(x=t,
                                 y=y_mean,
                                 mode='markers+lines',
                                 marker=dict(color=colors_[0][i]),
                                 name='Mean Value (speed:{}-{})'.format(speed_limits[i][0], speed_limits[i][1]),
                                 opacity=0.6))

        if 0 == 1:
            fig.add_trace(go.Scatter(x=t,
                                     y=y_upper_1std,
                                     mode='lines',
                                     marker=dict(color='blue'),
                                     name='Mean Value + 1 Standard deviation (speed:{}-{})'.format(
                                         speed_limits[i][0], speed_limits[i][1]),
                                     opacity=0.6))

            fig.add_trace(go.Scatter(x=t,
                                     y=y_lower_1std,
                                     mode='markers',
                                     marker=dict(color='blue'),
                                     name='Mean Value - 1 Standard deviation (speed:{}-{})'.format(
                                         speed_limits[i][0], speed_limits[i][1]),
                                     opacity=0.6))

        fig.add_trace(go.Scatter(x=t,
                                 y=y_upper_2std,
                                 mode='markers',
                                 marker=dict(color=colors_[1][i]),
                                 name='Mean Value + 2 Standard deviation (speed:{}-{})'.format(speed_limits[i][0],
                                                                                               speed_limits[i][1]),
                                 opacity=0.6))

        fig.add_trace(go.Scatter(x=t,
                                 y=y_lower_2std,
                                 mode='markers',
                                 marker=dict(color=colors_[1][i]),
                                 name='Mean Value - 2 Standard deviation (speed:{}-{})'.format(speed_limits[i][0],
                                                                                               speed_limits[i][1]),
                                 opacity=0.6))

    fig.update_yaxes(title_text=variable + ' ' + units_[variable])
    fig.update_xaxes(title_text='time (15 min)')

    fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
    fig.update_layout(title='{}  - Time series - (Reference Pattern)'.format(road))
    fig.update_layout(showlegend=True)
    # now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
    fig.write_html("figures\\{}_reference_pattern_by_speed_bandwith_{}.html".format(road, variable))
    fig.show()



def plot_reference_daily_pattern_by_speed_bandwith_per_section(df, road, variable):
    """ This function do this-

    :param df:
    :param road:
    :param columns_:
    :return:
    """

    import plotly.graph_objects as go
    from dl_ia_cla_settings import units_, segments_dict_LW
    from dl_ia_cla_settings import create_time_array

    t = create_time_array('00:00:00', '23:59:59', '15min')

    fig = go.Figure()

    colors_ = [('red', 'darkred'), ('blue', 'darkblue')]
    speed_limits = [(80, 70), (70, 0)]

    df_aux0 = df


    for subsection_ in df_aux0['sub_section'].unique():

        df_aux = df_aux0[df_aux0['sub_section'] == subsection_]

        for i in range(0, len(speed_limits)):
            # print(i)

            df_aux1 = df_aux[(df_aux['speed'] < speed_limits[i][0]) & (df_aux['speed'] >= speed_limits[i][1])]

            print(df_aux1.describe())

            df_mean = df_aux1.groupby(by=['hour', 'minute'])[variable].mean()
            df_std = df_aux1.groupby(by=['hour', 'minute'])[variable].std()
            y_mean = df_mean.values
            # y_upper_1std = df_mean.values + df_std.values
            # y_lower_1std = df_mean.values - df_std.values
            y_upper_2std = df_mean.values + 2 * df_std.values
            y_lower_2std = df_mean.values - 2 * df_std.values
            # y_lower_1std[y_lower_1std < 0] = 0
            y_lower_2std[y_lower_2std < 0] = 0

            fig.add_trace(go.Scatter(x=t,
                                     y=y_mean,
                                     mode='markers+lines',
                                     marker=dict(color=colors_[0][i]),
                                     name='Reference Pattern (speed:{}-{}) section:{}'.format(
                                         speed_limits[i][0],
                                         speed_limits[i][1],
                                         segments_dict_LW[subsection_]),
                                     opacity=0.6))
            if 1 == 0:
                fig.add_trace(go.Scatter(x=t,
                                         y=y_upper_2std,
                                         mode='markers',
                                         marker=dict(color=colors_[1][i]),
                                         name='Upper limit (speed:{}-{}) section:{}'.format(
                                             speed_limits[i][0],
                                             speed_limits[i][1],
                                             segments_dict_LW[subsection_]),
                                         opacity=0.6))

                fig.add_trace(go.Scatter(x=t,
                                         y=y_lower_2std,
                                         mode='markers',
                                         marker=dict(color=colors_[1][i]),
                                         name='Lower Limit (speed:{}-{}) section:{}'.format(
                                             speed_limits[i][0],
                                             speed_limits[i][1],
                                             segments_dict_LW[subsection_]),
                                         opacity=0.6))

    fig.update_yaxes(title_text=variable + ' ' + units_[variable])
    fig.update_xaxes(title_text='time (15 min)')

    fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
    fig.update_layout(title='{}  - Reference Pattern'.format(road))
    fig.update_layout(showlegend=True)
    # now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
    fig.write_html("figures\\{}_reference_pattern_by_speed_bandwith_{}_per_section.html".format(road, variable))
    fig.show()



def plot_gaps_betwen_referece_patterns_by_speed_bandwidth(df, road, variable):
    """ This function do this-

    :param df:
    :param road:
    :param columns_:
    :return:
    """

    import plotly.graph_objects as go
    from dl_ia_cla_settings import units_, segments_dict_LW
    from dl_ia_cla_settings import create_time_array

    t = create_time_array('00:00:00', '23:59:59', '15min')

    fig = go.Figure()

    colors_ = [('red', 'darkred'), ('blue', 'darkblue')]
    speed_limits = [(80, 70), (70, 60)]

    df_aux0 = df



    subsection_ = 2

    df_aux = df_aux0[df_aux0['sub_section'] == subsection_]

    df_aux1 = df_aux[(df_aux['speed'] < speed_limits[0][0]) & (df_aux['speed'] >= speed_limits[0][1])]
    df_mean1 = df_aux1.groupby(by=['hour', 'minute'])[variable].mean()
    y_mean1 = df_mean1.values

    df_aux2 = df_aux[(df_aux['speed'] < speed_limits[1][0]) & (df_aux['speed'] >= speed_limits[1][1])]
    df_mean2 = df_aux2.groupby(by=['hour', 'minute'])[variable].mean()
    y_mean2 = df_mean2.values

    diff = y_mean2 - y_mean1

    fig.add_trace(go.Scatter(x=t,
                             y=diff,
                             mode='markers+lines',
                             marker=dict(color=colors_[0][0]),
                             name=' $ \Delta V_{i,1,t}^{SP0} \hspace{3cm} $',
                             opacity=0.6))

    subsection_ = 3

    df_aux = df_aux0[df_aux0['sub_section'] == subsection_]

    df_aux1 = df_aux[(df_aux['speed'] < speed_limits[0][0]) & (df_aux['speed'] >= speed_limits[0][1])]
    df_mean1 = df_aux1.groupby(by=['hour', 'minute'])[variable].mean()
    y_mean1 = df_mean1.values

    df_aux2 = df_aux[(df_aux['speed'] < speed_limits[1][0]) & (df_aux['speed'] >= speed_limits[1][1])]
    df_mean2 = df_aux2.groupby(by=['hour', 'minute'])[variable].mean()
    y_mean2 = df_mean2.values

    diff = y_mean2 - y_mean1

    fig.add_trace(go.Scatter(x=t,
                             y=diff,
                             mode='markers+lines',
                             marker=dict(color=colors_[0][1]),
                             name=' $ \Delta V_{i,2,t}^{SP0} \hspace{3cm} $',
                             opacity=0.6))

    fig.update_yaxes(title_text=variable + ' ' + units_[variable])
    fig.update_xaxes(title_text='time (15 min)')

    fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
    fig.update_layout(title='{}  - Reference Pattern - {}'.format(road, segments_dict_LW[subsection_]))
    fig.update_layout(showlegend=True)
    # now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
    fig.write_html("figures\\{}_reference_pattern_by_speed_bandwith_{}_per_section.html".format(road, variable))
    fig.show()




def plot_two_timeserie(y1, y2, var_1, var_2, road):
    """ This plots two time series from a dataframe

    :param df: dataframe with the two time series to plot and a column timestamp
    :param var_1: first variable to plot (str)
    :param var_2: second variarble to plot (str)
    :param road: name of the road from which the variables belongs
    :return:
    """

    import plotly.graph_objects as go

    print_ = True
    show = True

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=y1.index,
                             y=y1,
                             mode='markers+lines',
                             marker=dict(color='red'),
                             name='{}'.format(name1)))

    fig.add_trace(go.Scatter(x=y2.index,
                             y=y2,
                             mode='markers+lines',
                             marker=dict(color='blue'),
                             name='{}'.format(name2)))

    fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
    fig.update_layout(showlegend=True)
    fig.update_yaxes(title_text=' ')
    fig.update_xaxes(title_text='Time')
    fig.update_layout(title='Timeseries plot - {}'.format(road))
    if show:
        fig.show()
    if print_:
        fig.write_html("figures\\{}_congestion.html".format(road))





def plot_daily_patterns(y1, y2, name1, name2):
    """ This function do this-

    :param df:
    :param road:
    :param columns_:
    :return:
    """

    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y1.index,
                             y=y1,
                             mode='markers+lines',
                             marker=dict(color='red'),
                             name='{} Section:{} Lane:{} Day:{} '.format(name1)))
    fig.add_trace(go.Scatter(x=y2.index,
                             y=y2,
                             mode='markers+lines',
                             marker=dict(color='blue'),
                             name='{} Section:{} Lane:{} Day:{} '.format(name2)))

    fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
    fig.update_layout(showlegend=True)
    fig.update_yaxes(title_text=' ')
    fig.update_xaxes(title_text='Time (15 min)')
    fig.update_layout(title='{} -  '.format(road))
    # fig.write_html("figures\\{}_congestion.html".format(road))
    fig.show()


def plot_scatter_plotly(df, variable_x, variable_y):
    """ This function do this-

    :param df:
    :param road:
    :param columns_:
    :return:
    """
    import plotly.graph_objects as go
    from dl_ia_cla_settings import units_

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[variable_x],
                             y=df[variable_y],
                             mode='markers',
                             marker=dict(color='blue')))

    fig.update_xaxes(title_text=variable_x + ' ' + units_[variable_y])
    fig.update_yaxes(title_text=variable_y + ' ' + units_[variable])

    fig.update_layout(font=dict(family="Courier New, monospace", size=12, color="#7f7f7f"))
    fig.update_layout(title=road)
    fig.update_layout(showlegend=False)
    # fig.show()


def plot_scatter_mpl(df, variable):
    """ This function do this-

    :param df:
    :param road:
    :param columns_:
    :return:
    """

    import matplotlib.pyplot as plt

    print_ = True
    show = True

    fig = plt.figure()

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.scatter(df['occupancy'],
                df[variable])

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.scatter(df['speed'],
                df[variable])

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.scatter(df['headway'],
                df[variable])

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.scatter(df['meanlength'],
                df[variable])

    if print_:
        plt.show()
    if show:
        plt.savefig('figures\\plot_scatter_mpl.png')

def plot_scatter_together(df):
    """ This function do this-

    :param df:
    :param road:
    :param columns_:
    :return:
    """

    import plotly.graph_objects as go
    from dl_ia_cla_settings import units_
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=2, cols=2)

    fig.add_trace(go.Scatter(x=df['occupancy'],
                             y=df['flow'],
                             mode='markers',
                             name='Occupacy-flow',
                             marker=dict(color='blue')), row=1, col=1)

    fig.add_trace(go.Scatter(x=df['speed'],
                             y=df['flow'],
                             mode='markers',
                             name='speed-flow',
                             marker=dict(color='blue')), row=2, col=1)

    fig.add_trace(go.Scatter(x=df['headway'],
                             y=df['flow'],
                             mode='markers',
                             name='headway-flow',
                             marker=dict(color='blue')), row=1, col=2)

    fig.add_trace(go.Scatter(x=df['meanlength'],
                             y=df['flow'],
                             mode='markers',
                             name='meanlength-flow',
                             marker=dict(color='blue')), row=2, col=2)

    fig.update_xaxes(title_text="occupancy" + ' ' + units_['occupancy'], row=1, col=1)
    fig.update_yaxes(title_text="flow" + ' ' + units_['flow'], row=1, col=1)

    fig.update_xaxes(title_text="speed" + ' ' + units_['speed'], row=2, col=1)
    fig.update_yaxes(title_text="flow" + ' ' + units_['flow'], row=2, col=1)

    fig.update_xaxes(title_text="headway" + ' ' + units_['headway'], row=1, col=2)
    fig.update_yaxes(title_text="flow" + ' ' + units_['flow'], row=1, col=2)

    fig.update_xaxes(title_text="meanlength" + ' ' + units_['meanlength'], row=2, col=2)
    fig.update_yaxes(title_text="flow" + ' ' + units_['flow'], row=2, col=2)

    fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
    fig.update_layout(title=road)
    fig.update_layout(showlegend=False)
    # fig.show()



def plot_simulation(df_sim):
    """ This function do this-

    :param df:
    :param road:
    :param columns_:
    :return:
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=4, cols=1,
                        subplot_titles=("Volume Closure Section: {} Direction:{}".format(section_to_close, direction_),
                                        "Speed Closure Section: {} Direction:{}".format(section_to_close, direction_),
                                        "Volume Previous Section: {} Direction:{}".format(prev_section, direction_),
                                        "Speed Previous Section: {} Direction:{}".format(prev_section, direction_)))

    fig.add_trace(go.Scatter(x=df_sim['volume_prev_lane_{}'.format(lane_to_close)].index,
                             y=df_sim['volume_prev_lane_{}'.format(lane_to_close)],
                             mode='markers+lines',
                             marker=dict(color='red'),
                             line=dict(width=3),
                             legendgroup="lane1",
                             name='{}'.format('lane 1')), row=3, col=1)

    fig.add_trace(go.Scatter(x=df_sim['volume_prev_lane_{}'.format(lane_free)].index,
                             y=df_sim['volume_prev_lane_{}'.format(lane_free)],
                             mode='markers+lines',
                             line=dict(width=3),
                             legendgroup="lane2",
                             marker=dict(color='darkred'),
                             name='{}'.format('lane 2')), row=3, col=1)

    fig.add_trace(go.Scatter(x=df_sim['volume_prev_lane_{}_modify'.format(lane_to_close)].index,
                             y=df_sim['volume_prev_lane_{}_modify'.format(lane_to_close)],
                             mode='markers+lines',
                             line=dict(width=2, dash='dash'),
                             marker=dict(color='red'),
                             legendgroup="lane1_after",
                             name='{}'.format('lane 1 sim.')), row=3, col=1)

    fig.add_trace(go.Scatter(x=df_sim['volume_prev_lane_{}_modify'.format(lane_free)].index,
                             y=df_sim['volume_prev_lane_{}_modify'.format(lane_free)],
                             mode='markers+lines',
                             line=dict(width=2, dash='dash'),
                             legendgroup="lane2_after",
                             marker=dict(color='darkred'),
                             name='{}'.format('lane 2 sim. ')), row=3, col=1)

    fig.add_trace(go.Scatter(x=df_sim['volume_actual_lane_{}_modify'.format(lane_to_close)].index,
                             y=df_sim['volume_actual_lane_{}_modify'.format(lane_to_close)],
                             mode='markers+lines',
                             legendgroup="lane1_after",
                             line=dict(width=2, dash='dash'),
                             marker=dict(color='red'),
                             showlegend=False,
                             name='{}'.format('lane 1 - sim.')), row=1, col=1)

    fig.add_trace(go.Scatter(x=df_sim['volume_actual_lane_{}_modify'.format(lane_free)].index,
                             y=df_sim['volume_actual_lane_{}_modify'.format(lane_free)],
                             mode='markers+lines',
                             legendgroup="lane2_after",
                             line=dict(width=2, dash='dash'),
                             marker=dict(color='darkred'),
                             showlegend=False,
                             name='{}'.format('lane 2 - sim.')), row=1, col=1)

    fig.add_trace(go.Scatter(x=df_sim['volume_actual_lane_{}'.format(lane_to_close)].index,
                             y=df_sim['volume_actual_lane_{}'.format(lane_to_close)],
                             mode='markers+lines',
                             legendgroup="lane1",
                             line=dict(width=3),
                             marker=dict(color='red'),
                             showlegend=False,
                             name='{}'.format('lane 1')), row=1, col=1)

    fig.add_trace(go.Scatter(x=df_sim['volume_actual_lane_{}'.format(lane_free)].index,
                             y=df_sim['volume_actual_lane_{}'.format(lane_free)],
                             mode='markers+lines',
                             legendgroup="lane2",
                             line=dict(width=3),
                             marker=dict(color='darkred'),
                             showlegend=False,
                             name='{}'.format('lane 1')), row=1, col=1)

    fig.add_trace(go.Scatter(x=df_sim['speed_prev_lane_{}'.format(lane_to_close)].index,
                             y=df_sim['speed_prev_lane_{}'.format(lane_to_close)],
                             mode='markers+lines',
                             legendgroup="lane1",
                             line=dict(width=3),
                             marker=dict(color='red'),
                             showlegend=False,
                             name='{}'.format('Lane 1')), row=4, col=1)

    fig.add_trace(go.Scatter(x=df_sim['speed_prev_lane_{}'.format(lane_free)].index,
                             y=df_sim['speed_prev_lane_{}'.format(lane_free)],
                             mode='markers+lines',
                             legendgroup="lane2",
                             line=dict(width=3),
                             marker=dict(color='darkred'),
                             showlegend=False,
                             name='{}'.format('Lane 2')), row=4, col=1)

    fig.add_trace(go.Scatter(x=df_sim['speed_prev_lane_{}_modify'.format(lane_to_close)].index,
                             y=df_sim['speed_prev_lane_{}_modify'.format(lane_to_close)],
                             mode='markers+lines',
                             line=dict(width=2, dash='dash'),
                             legendgroup="lane1_after",
                             marker=dict(color='red'),
                             showlegend=False,
                             name='{}'.format('Lane 1')), row=4, col=1)

    fig.add_trace(go.Scatter(x=df_sim['speed_prev_lane_{}_modify'.format(lane_free)].index,
                             y=df_sim['speed_prev_lane_{}_modify'.format(lane_free)],
                             mode='markers+lines',
                             line=dict(width=2, dash='dash'),
                             legendgroup="lane2_after",
                             marker=dict(color='darkred'),
                             showlegend=False,
                             name='{}'.format('Lane 2')), row=4, col=1)

    fig.add_trace(go.Scatter(x=df_sim['speed_actual_lane_{}_modify'.format(lane_to_close)].index,
                             y=df_sim['speed_actual_lane_{}_modify'.format(lane_to_close)],
                             mode='markers+lines',
                             legendgroup="lane1_after",
                             line=dict(width=2, dash='dash'),
                             marker=dict(color='red'),
                             showlegend=False,
                             name='{}'.format('Lane 1')), row=2, col=1)

    fig.add_trace(go.Scatter(x=df_sim['speed_actual_lane_{}_modify'.format(lane_free)].index,
                             y=df_sim['speed_actual_lane_{}_modify'.format(lane_free)],
                             mode='markers+lines',
                             legendgroup="lane2_after",
                             marker=dict(color='darkred'),
                             line=dict(width=2, dash='dash'),
                             showlegend=False,
                             name='{}'.format('Lane 2')), row=2, col=1)

    fig.add_trace(go.Scatter(x=df_sim['speed_actual_lane_{}'.format(lane_to_close)].index,
                             y=df_sim['speed_actual_lane_{}'.format(lane_to_close)],
                             mode='markers+lines',
                             legendgroup="lane1",
                             line=dict(width=3),
                             marker=dict(color='red'),
                             showlegend=False,
                             name='{}'.format('Lane 1')), row=2, col=1)

    fig.add_trace(go.Scatter(x=df_sim['speed_actual_lane_{}'.format(lane_free)].index,
                             y=df_sim['speed_actual_lane_{}'.format(lane_free)],
                             mode='markers+lines',
                             legendgroup="lane2",
                             line=dict(width=3),
                             marker=dict(color='darkred'),
                             showlegend=False,
                             name='{}'.format('Lane 2')), row=2, col=1)

    fig.update_layout(font=dict(family="Courier New, monospace", size=11, color="#7f7f7f"))
    fig.update_layout(showlegend=True)
    fig.update_yaxes(title_text=' ')
    # fig.update_xaxes(title_text='Time (15 min)')
    fig.update_layout(title='{} - Road Traffic simulation - date:{} from {}am to {}am [{}] '.format(road, date_.date(), start_closure, end_closure, message))
    fig.write_html("figures\\{}\\road_traffic_simulation.html".format(road, section_to_close, lane_to_close))
    fig.show()


"""
================================== DESCRIPTION =====================================

    This script produce the following plots:
    Scatterplots
    Histograms
    Boxplots
    Time series



================================== NOMENCLATURE =====================================



    vol (veh/30 seg ) --> x 30 --> vol (veh/15 min)

    vol (veh/30 seg ) = 1 --> flow(veh/h)= 120

    vol (veh/15 min ) = 1 --> flow(veh/h)= 4

    volume              Vehicles per 15 min
    flow :              Vehicles per hour (Hourly intensity)
    speed:              average vehicles speed (Km/h)
    meanlegth:          average vehicles length (dm)
    gap:                average time separation between vehicles (tenth of second)
    headway:            average time separation from the front of a vehicle to the front of the next vehicle (tenth of second)
    occupancy :         percentage of the integration time (30 sec) in which a vehicle has been detected

"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import numpy as np
from tqdm import tqdm
import os
from datetime import datetime

from LW_settings import config_pandas, config_plotly, data_path, road, units_, columns_, all_columns_

config_pandas()
config_plotly()

df = pd.read_csv(data_path + 'LW_road_traffic_resampled.csv')
df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S')
df.set_index(['time'], inplace=True)
df.sort_index(inplace=True)

if not os.path.isdir('figures\\'):
    os.mkdir('figures\\')


def filter_by_std(df, variable, option):
    if option == 2:
        df_aux = df[(df[variable] < (df[variable].mean() + 2 * df[variable].std()))
                    & (df[variable] > (df[variable].mean() - 2 * df[variable].std()))]
    elif option == 1:
        df_aux = df[(df[variable] < (df[variable].mean() + df[variable].std()))
                    & (df[variable] > (df[variable].mean() - df[variable].std()))]

    print('Rows dropped:{} %'.format(round(100 * (1 - (len(df_aux) / len(df))), 3)))
    return df_aux


filter_by_std(df, 'speed', 2)

"""
"""
"""  ===================== Data Visualization =====================
"""
"""      
"""
if 1 == 1:

    """
    """
    """ ===================== Bar plots ===================
    """
    """
    """


    def plot_freq_bar_plot_volume(df, road):

        df_count = df.groupby(by=['hour', 'minute'])['volume'].sum()

        t = pd.date_range('2020-05-25 00:00:00', '2020-05-25 23:45:00', freq='15min')

        volume_relative = 100 * df_count.values / df_count.values.sum()

        fig = go.Figure([go.Bar(x=t, y=volume_relative)])
        fig.update_yaxes(title_text=' Freq (%)')
        fig.update_xaxes(title_text='Time steps (15 min)')

        fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
        fig.update_layout(title='{}  - Freq. distribution - Volume'.format(road))
        fig.update_layout(showlegend=True)
        fig.write_html("figures\\{}_freq_barplot_volume.html".format(road))
        fig.show()


    if 1 == 0:
        plot_freq_bar_plot_volume(df, road)

    """
    """
    """ ===================== Histograms ===================
    """
    """
    """


    def plot_histogram(df, variable, n_bins_, road):
        fontsize_ = 18
        fig = px.histogram(df, x=variable, nbins=n_bins_)
        fig.update_xaxes(title_text=variable + ' ' + units_[variable])
        fig.update_layout(font=dict(family="Courier New, monospace", size=fontsize_, color="#7f7f7f"))
        fig.update_layout(title='{} - Histogram  - {}'.format(road, variable))
        fig.update_layout(showlegend=True)
        fig.update_traces(opacity=0.9)
        fig.update_layout(bargap=0.2)  # gap between bars of adjacent location coordinates
        max_value = int(df[variable].max())
        x_axis = np.arange(0, max_value, int(max_value / 20))
        fig.update_xaxes(ticktext=x_axis, tickvals=x_axis)
        # now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
        fig.write_html("figures\\{}_histogram_{}.html".format(road, variable))
        fig.show()


    if 1 == 0:
        plot_histogram(df, 'volume', 150, road)

        df_aux = filter_by_std(df, 'speed', 2, road)
        plot_histogram(df_aux, 'speed', 150)


    def plot_histogram_mpl(df, variable, n_bins, road):
        fig = plt.figure(figsize=(10, 8), dpi=150, facecolor='w', edgecolor='w')
        ax = fig.add_subplot(1, 1, 1)
        fontsize_ = 18
        # fig, ax = plt.subplots(1, 1, sharey=True, tight_layout=True)
        n = ax.hist(df[variable], bins=n_bins, edgecolor='b')
        ax.set_xlabel(variable + ' ' + units_[variable], fontsize=fontsize_)
        ax.set_ylabel('Count', fontsize=fontsize_)
        xmax = df[variable].max()
        x_range = list(np.arange(0, xmax, int(xmax / 20)))
        # ax.set_xticks = x_range
        # ax.set_xticklabels = x_range
        ax.tick_params(axis='both', labelsize=fontsize_)
        now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
        plt.title('{} - Histogram - {} '.format(road, variable), fontsize=fontsize_)
        plt.show()
        plt.savefig('figures\\{}_histogram_{}.png'.format(road, variable),
                    bbox_inches='tight', pad_inches=0)
        plt.close()


    if 0 == 1:
        plot_histogram_mpl(df, 'volume', 25, road)

        df_aux = filter_by_std(df, 'speed', 2)
        plot_histogram_mpl(df_aux, 'speed', 100, road)

        plot_histogram_mpl(df, 'occupancy', 50, road)
        plot_histogram_mpl(df, 'flow', 50, road)

        df_aux = df[df['meanlength'] > 0]
        plot_histogram_mpl(df_aux, 'meanlength', 100, road)

    """
    """
    """  =================== Correlation matrix ====================
    """
    """
    """


    def plot_correlation_matrix(df, road, columns_):
        """ Plot correlation matrix with values
        """
        R = df.loc[:, columns_].corr()
        r = round(R, 2)
        fig = ff.create_annotated_heatmap(r.values,
                                          colorscale='Viridis',
                                          x=columns_,
                                          y=columns_)
        fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
        fig.update_layout(title='{} - Correlation Matrix - Pearson correlation coefficient'.format(road))
        fig.update_layout(showlegend=True)
        # now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
        fig.write_html("figures\\{}_correlation_matrix.html".format(road))
        # fig.show()


    if 1 == 0:
        plot_correlation_matrix(df, road, columns_)

    """
    """
    """  =========================== Boxplots ===========================
    """
    """
    """


    def boxplot_plotly(df, variable, group_variable, color_variable, road):
        fig = px.box(df, y=variable, x=group_variable, color=color_variable)
        fig.update_xaxes(title_text=group_variable)
        fig.update_yaxes(title_text='{}'.format(variable + ' ' + units_[variable]))
        fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
        fig.update_layout(title='{} - Boxplot - {}'.format(road, variable))
        fig.update_layout(showlegend=True)
        # now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
        fig.write_html(
            "figures\\{}_boxplot_{}_{}_{}.html".format(road, variable, group_variable, color_variable))
        # fig.show()


    if 1 == 0:
        boxplot_plotly(df, 'volume', 'section', 'hour', road)

    """
    """
    """ ================= Scatter plots ============================== 
    """
    """
    """


    def plot_scatterplot(df, variable_x, variable_y, road):

        fig = px.scatter(df,
                         x=variable_x,
                         y=variable_y, )

        fig.update_xaxes(title_text=variable_x + ' ' + units_[variable_x])
        fig.update_yaxes(title_text=variable_y + ' ' + units_[variable_y])
        fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
        fig.update_layout(title='{} - Scatterplots'.format(road))
        fig.update_layout(showlegend=True)
        # now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
        fig.write_html(
            "figures\\{}_scatterplot_{}_{}.html".format(road, variable_x, variable_y))
        # fig.show()


    if 1 == 0:
        plot_scatterplot(df, 'speed', 'volume', road)


    def plot_scatterplot_mpl(df, variable_x, variable_y, group_variable, road):
        """
        Scatterplot matplotlib individual with color by variable

        :param df:
        :param variable_x:
        :param variable_y:
        :param group_variable:
        :param road:
        :return:
        """
        fig = plt.figure(figsize=(12, 8), dpi=150, facecolor='w', edgecolor='w')
        ax = fig.add_subplot(1, 1, 1)

        for var in df[group_variable].unique():
            df_aux = df[df[group_variable] == var]
            x = df_aux[variable_x].values
            y = df_aux[variable_y].values
            ax.scatter(x, y, label=var, alpha=0.8, edgecolor='k')

        fontsize_ = 12
        ax.set_xlabel(variable_x + ' ' + units_[variable_x], fontsize=fontsize_)
        ax.set_ylabel(variable_y + ' ' + units_[variable_y], fontsize=fontsize_)
        ax.tick_params(axis='both', labelsize=fontsize_)
        plt.legend()

        now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
        plt.title('{}'.format(road), fontsize=fontsize_)
        plt.show()

        plt.savefig('figures\\{}_scatterplot_{}_{}.png'.format(road, variable_y, variable_x),
                    bbox_inches='tight',
                    pad_inches=0)
        plt.close()


    # plot_scatterplot_mpl(df, 'flow', 'occupancy', 'schedule_def', road)

    def plot_scatterplot_mpl_multi(df, variable_x, variable_y, group_var, road):
        """
        Scatterplot matplotlib (4 plots)

        :param df:
        :param variable_x:
        :param variable_y:
        :param group_var:
        :param road:
        :return:
        """
        fontsize_ = 12
        marker_size_ = 10

        fig = plt.figure(figsize=(12, 8), dpi=150, facecolor='w', edgecolor='w')

        for i in np.arange(0, len(variable_x)):
            ax1 = fig.add_subplot(2, 2, i + 1)
            sc1 = ax1.scatter(df[variable_x[i]],
                              df[variable_y[i]], marker='o', s=marker_size_, alpha=0.8, c=df[group_var])
            cbar1 = plt.colorbar(sc1)
            cbar1.set_label(group_var, rotation=270, fontsize=fontsize_)
            ax1.set_xlabel(variable_x[i] + ' ' + units_[variable_x[i]], fontsize=fontsize_)
            ax1.set_ylabel(variable_y[i] + ' ' + units_[variable_y[i]], fontsize=fontsize_)

            y_min = df[variable_y[i]].min()
            y_max = df[variable_y[i]].max()
            x_min = df[variable_x[i]].min()
            x_max = df[variable_x[i]].max()
            y_range = np.arange(y_min, y_max, round(y_max / 5))
            x_range = np.arange(x_min, x_max, round(x_max / 5))
            ax1.set_yticks(y_range)
            ax1.set_yticklabels(y_range)
            ax1.set_xticks(x_range)
            ax1.set_xticklabels(x_range)
            ax1.tick_params(axis='both', labelsize=fontsize_)

        now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
        plt.title('{}'.format(road), fontsize=fontsize_)
        plt.subplots_adjust(wspace=0.2, hspace=0.4)
        plt.show()
        plt.savefig('figures\\{}_scatterplot_multi_{}.png'.format(road, now),
                    bbox_inches='tight', pad_inches=0)

        plt.close()


    if 1 == 0:
        variable_x = ['occupancy', 'gap', 'speed', 'meanlength']
        variable_y = ['volume', 'volume', 'volume', 'volume']
        plot_scatterplot_mpl_multi(df, variable_x, variable_y, 'hour', road)


    def plot_scatterplot_by_color(df, variable_x, variable_y, color_variable, road):
        """
        Statter plot plotly individual all variables segmented by schedule (!) ALL COMBINATIONS (6x7=42 plots)

        :param df:
        :param variable_x:
        :param variable_y:
        :param color_variable:
        :param road:
        :return:
        """
        fig = px.scatter(df,
                         x=variable_x,
                         y=variable_y,
                         color=color_variable,
                         opacity=0.8,
                         hover_data=[color_variable])

        fig.update_xaxes(title_text=variable_x + ' ' + units_[variable_x])
        fig.update_yaxes(title_text=variable_y + ' ' + units_[variable_y])
        fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
        fig.update_layout(title='{} - Scatterplots'.format(road))
        fig.update_layout(showlegend=True)
        # now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
        # fig.show()
        fig.write_html(
            "figures\\{}_scatterplot_by_color_{}_{}.html".format(road, variable_x, variable_y))


    if 1 == 0:
        df_aux = filter_by_std(df, 'speed', 2)
        plot_scatterplot_by_color(df_aux, 'speed', 'volume', 'hour', road)


    def plot_scatterplot_by_color_multi(df, variable_x, variable_y, road):
        """
        Plotly scatterplot multiple with all values and segmented by schedule time
        (!)  Too slow


        :param df:
        :param variable_x:
        :param variable_y:
        :param road:
        :return:
        """
        fig = make_subplots(rows=2, cols=3)

        index = {'0-6': (1, 1),
                 '7-12': (1, 2),
                 '13-17': (1, 3),
                 '18-21': (2, 1),
                 '22-23': (2, 2),
                 }

        colors_ = {'0-6': 'darkblue',
                   '7-12': 'darkgreen',
                   '13-17': 'darkred',
                   '18-21': 'darkviolet',
                   '22-23': 'darkgrey',
                   }

        for schedule_ in df['schedule_def'].unique():
            df_aux = df.loc[df['schedule_def'] == schedule_]

            fig.add_trace(go.Scatter(x=df_aux[variable_x].values,
                                     y=df_aux[variable_y].values,
                                     name=schedule_,
                                     mode='markers',
                                     marker_color=colors_[schedule_]), row=index[schedule_][0], col=index[schedule_][1])

            x_axis = np.arange(0, int(df[variable_x].max()), int(df[variable_x].max() / 10))
            y_axis = np.arange(0, int(df[variable_y].max()), int(df[variable_y].max() / 10))
            fig.update_yaxes(title_text=variable_y,
                             tickfont=dict(size=14),
                             ticktext=y_axis,
                             tickvals=y_axis, row=index[schedule_][0], col=index[schedule_][1])

            fig.update_xaxes(title_text=variable_x,
                             tickfont=dict(size=14),
                             ticktext=x_axis,
                             tickvals=x_axis, row=index[schedule_][0], col=index[schedule_][1])

        fig.update_layout(showlegend=True)
        fig.update_layout(font=dict(family="Courier New, monospace", size=14, color="#7f7f7f"))
        fig.update_layout(title_text='{} - Scatterplots '.format(road))
        # now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
        # fig.show()
        fig.write_html("figures\\{}_scatterplot_multi_{}_{}_by_schedule_{}.html".format(road, variable_x, variable_y))


    # plot_scatterplot_by_color_multi(df, 'occupancy', 'volume', road)

    def plot_scatterplot_by_schedule_multi(df, variable_x, variable_y, road):
        """

        (!) Too slow
        Plotly scatterplot individual with values segmented by schedule time

        :param df:
        :param variable_x:
        :param variable_y:
        :param road:
        :return:
        """
        colors_ = {'0-6': 'darkblue',
                   '7-12': 'darkgreen',
                   '13-17': 'darkred',
                   '18-21': 'darkviolet',
                   '22-23': 'darkgrey',
                   }

        for schedule_ in df['schedule_def'].unique():
            df_aux = df.loc[df['schedule_def'] == schedule_]

            fig = go.Figure()

            fig.add_trace(go.Scatter(x=df[variable_x].values,
                                     y=df[variable_y].values,
                                     name='All',
                                     mode='markers',
                                     opacity=0.9,
                                     marker_color='lightgrey'))

            fig.add_trace(go.Scatter(x=df_aux[variable_x].values,
                                     y=df_aux[variable_y].values,
                                     name=schedule_,
                                     mode='markers',
                                     marker_color=colors_[schedule_]))

            x_axis = np.arange(0, int(df[variable_x].max()), int(df[variable_x].max() / 10))
            y_axis = np.arange(0, int(df[variable_y].max()), int(df[variable_y].max() / 10))
            fig.update_yaxes(title_text=variable_y + units_[variable_y],
                             tickfont=dict(size=14),
                             ticktext=y_axis,
                             tickvals=y_axis)

            fig.update_xaxes(title_text=variable_x + units_[variable_x],
                             tickfont=dict(size=14),
                             ticktext=x_axis,
                             tickvals=x_axis)

            fig.update_layout(showlegend=True)
            fig.update_layout(font=dict(family="Courier New, monospace", size=14, color="#7f7f7f"))
            fig.update_layout(title_text='{} - Scatterplot '.format(road))
            now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
            fig.write_html(
                "figures\\{}_scatterplot_multi_{}_{}_schedule_{}_{}.html".format(road, variable_x, variable_y,
                                                                                 schedule_, now))
            # fig.show()


    # plot_scatterplot_by_schedule_multi(df, 'volume', 'speed', road)

    def plot_scatterplot_with_histogram(df, variable_y, variable_x, grouping_variable, road):
        """

        :param variable_y:
        :param variable_x:
        :param grouping_variable:
        :param road:
        :return:
        """
        ""
        # definitions for the axes
        left, width = 0.1, 0.4
        bottom, height = 0.1, 0.4
        spacing = 0.005

        fontsize_ = 16

        rect_scatter = [left, bottom, width, height]
        rect_histx = [left, bottom + height + spacing, width, 0.2]
        rect_histy = [left + width + spacing, bottom, 0.2, height]

        # start with a square Figure
        fig = plt.figure(figsize=(12, 8), dpi=150, facecolor='w', edgecolor='w')

        ax = fig.add_axes(rect_scatter)
        ax_histx = fig.add_axes(rect_histx, sharex=ax)
        ax_histy = fig.add_axes(rect_histy, sharey=ax)

        x = df[variable_x].values
        y = df[variable_y].values

        color_ = df[grouping_variable].values

        # now determine nice limits by hand:
        binwidth = 0.5
        xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
        lim = (int(xymax / binwidth) + 1) * binwidth
        bins = np.arange(0, lim + binwidth, binwidth)

        ax_histx.hist(x, bins=bins)
        ax_histy.hist(y, bins=bins, orientation='horizontal')
        ax_histx.axis('off')
        ax_histy.axis('off')

        # no labels
        ax_histx.tick_params(axis="x", labelbottom=False)
        ax_histy.tick_params(axis="y", labelleft=False)

        # the scatter plot:
        sc = ax.scatter(x, y, c=color_)
        cbar = plt.colorbar(sc)
        cbar.set_label(grouping_variable, rotation=270, fontsize=fontsize_)

        xmax = int(df[variable_x].max())
        ymax = int(df[variable_y].max())

        ax.set_xlabel(variable_x + ' ' + units_[variable_x], fontsize=fontsize_)
        ax.set_ylabel(variable_y + ' ' + units_[variable_y], fontsize=fontsize_)

        ax.tick_params(axis='both', labelsize=fontsize_)
        ax.set_xticks = list(np.arange(0, xmax, int(xmax / 10)))
        ax.set_yticks = list(np.arange(0, ymax, int(ymax / 10)))
        ax.set_xticklabels = list(np.arange(0, xmax, int(xmax / 10)))
        ax.set_yticklabels = list(np.arange(0, ymax, int(ymax / 10)))

        # now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")

        plt.title('{}'.format(road), fontsize=fontsize_)
        plt.show()
        plt.savefig('figures\\{}_scatterplot_with_histogram_{}_{}.png'.format(road, variable_y, variable_x),
                    bbox_inches='tight', pad_inches=0)
        plt.close()


    if 1 == 0:
        df_aux = filter_by_std(df, 'speed', 2)
        plot_scatterplot_with_histogram(df_aux, 'volume', 'speed', 'hour', road)
        plot_scatterplot_with_histogram(df_aux, 'occupancy', 'speed', 'hour', road)

        plot_scatterplot_with_histogram(df, 'volume', 'occupancy', 'hour', road)

    """
    """
    """  ==================== Time series =======================
    """
    """
    """


    def plot_time_series_mpl(df, start_date, end_date, road, variable):
        """
        Plot time series with matplotlib, one variable per plot,
        segmented by section, for a specific data range
        The plot is saved in png format
        :param df:
        :param start_date:
        :param end_date:
        :param road:
        :return:
        """
        fontsize_ = 18

        fig = plt.figure(figsize=(25, 10), dpi=150, facecolor='w', edgecolor='w')
        ax = fig.add_subplot(1, 1, 1)

        for section_ in df['ID'].unique():
            df_aux = df[df['ID'] == section_]
            y = df_aux.loc[start_date:end_date, [variable]]
            ax.plot(y, '-o', label='section:{}'.format(section_))

        ax.set_xlabel('time (15 min)', fontsize=fontsize_)
        ax.set_ylabel(variable + units_[variable], fontsize=fontsize_)
        ax.tick_params(axis='both', labelsize=fontsize_)
        ax.legend(fontsize=fontsize_)
        ax.set_title('{} - Time Series - {} to {}'.format(road, start_date, end_date), fontsize=fontsize_)
        # now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
        plt.subplots_adjust(wspace=0.2, hspace=0.4)
        plt.show()
        plt.savefig('figures\\{}_timeseries_{}.png'.format(road, variable), bbox_inches='tight', pad_inches=0)
        plt.close()


    if 1 == 0:
        # select dates
        start_date = '2019-03-15'
        end_date = '2019-03-16'

        plot_time_series_mpl(df, start_date, end_date, road, 'volume')
        plot_time_series_mpl(df, start_date, end_date, road, 'speed')
        plot_time_series_mpl(df, start_date, end_date, road, 'occupancy')
        plot_time_series_mpl(df, start_date, end_date, road, 'meanlength')
        plot_time_series_mpl(df, start_date, end_date, road, 'gap')


    def plot_time_series(df, road, variable, start_date, end_date):

        df_aux = df[(df.index > start_date) & (df.index < end_date)]

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=df_aux[variable].index,
                                 y=df_aux[variable],
                                 mode='markers',
                                 name=variable + ' ' + units_[variable]))

        fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
        fig.update_layout(showlegend=True)
        fig.update_yaxes(title_text=variable + ' ' + units_[variable])
        fig.update_xaxes(title_text='Time (15 min)')
        fig.update_layout(title='{}  - Time series'.format(road))

        # Add range slider
        fig.update_layout(
            xaxis=go.layout.XAxis(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1,
                             label="1 day",
                             step="day",
                             stepmode="backward"),
                        dict(count=3,
                             label="3 days",
                             step="day",
                             stepmode="backward"),
                        dict(count=7,
                             label="1 week",
                             step="day",
                             stepmode="backward")
                    ])
                ),
                rangeslider=dict(
                    visible=True
                ),
                type="date"
            )
        )
        now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
        fig.write_html("figures\\{}_timeseries_slider_{}.html".format(road, variable))
        fig.show()


    if 0 == 1:
        start_date = '2019-03-02'
        end_date = '2019-03-16'
        plot_time_series_with_slider(df, road, 'volume', start_date, end_date)


    def plot_time_series_segmented(df, start_date, end_date, road, variable, group_variable):
        """
        Plot time series with plotly, one variable per plot,
        segmented by section, for a specific data range
        The plot is saved in html format
        :param df:
        :param start_date:
        :param end_date:
        :param road:
        :return:
        """

        fig = go.Figure()

        for group in df[group_variable].unique():
            df_aux = df[df[group_variable] == group]
            df_aux = df_aux.loc[start_date:end_date]

            t = df_aux.index
            y = df_aux[variable]

            fig.add_trace(go.Scatter(x=t,
                                     y=y,
                                     mode='markers',
                                     name=group,
                                     opacity=0.6))

        # fig.update_xaxes(title_text='Time (15 min)')
        fig.update_yaxes(title_text=variable + ' ' + units_[variable])

        fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
        fig.update_layout(title='{}  - Time series - {} to {}'.format(road, start_date, end_date))
        fig.update_layout(showlegend=True)

        # Add range slider
        fig.update_layout(
            xaxis=go.layout.XAxis(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1,
                             label="1 day",
                             step="day",
                             stepmode="backward"),
                        dict(count=3,
                             label="3 days",
                             step="day",
                             stepmode="backward"),
                        dict(count=7,
                             label="1 week",
                             step="day",
                             stepmode="backward")
                    ])
                ),
                rangeslider=dict(
                    visible=True
                ),
                type="date"
            )
        )

        # now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
        fig.write_html("figures\\{}_timeseries_segmented_{}.html".format(road, variable))
        # fig.show()


    if 1 == 0:
        start_date = '2019-03-02'
        end_date = '2019-03-16'
        plot_time_series_segmented(df, start_date, end_date, road, 'volume', 'ID')

    """ Time series plotly with time splider
    """


    def plot_time_series_by_section(df, road, variable, start_date, end_date):

        df_aux = df[(df.index > start_date) & (df.index < end_date)]
        section_ = df_aux['ID'].unique()[0]

        df_aux = df_aux[df_aux['ID'] == section_]
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=df_aux[variable].index,
                                 y=df_aux[variable],
                                 mode='markers',
                                 name='{}-Section:{}'.format(variable, str(section_))))

        fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
        fig.update_layout(showlegend=True)
        fig.update_yaxes(title_text=variable + ' ' + units_[variable])
        fig.update_xaxes(title_text='Time (15 min)')
        fig.update_layout(title='{}  - Time series'.format(road))

        # Add range slider
        fig.update_layout(
            xaxis=go.layout.XAxis(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1,
                             label="1 day",
                             step="day",
                             stepmode="backward"),
                        dict(count=3,
                             label="3 days",
                             step="day",
                             stepmode="backward"),
                        dict(count=7,
                             label="1 week",
                             step="day",
                             stepmode="backward")
                    ])
                ),
                rangeslider=dict(
                    visible=True
                ),
                type="date"
            )
        )
        # now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
        fig.write_html("figures\\{}_timeseries_by_section_{}.html".format(road, variable))
        # fig.show()


    if 1 == 0:
        start_date = '2019-03-01'
        end_date = '2019-06-01'
        plot_time_series_by_section(df, road, 'volume', start_date, end_date)


    def plot_time_series_by_speed_bandwith(df, road, variable, start_date, end_date):

        df_aux = df[(df.index > start_date) & (df.index < end_date)]

        # segment_ = df_aux['segment'].unique()[0]
        # df_aux = df_aux[df_aux['segment'] == segment_]
        # section_ = df_aux['ID'].unique()[0]

        # Plot all sections

        df_aux1 = df_aux[(df_aux['speed'] < 85) & (df_aux['speed'] >= 75)]
        df_aux2 = df_aux[(df_aux['speed'] < 75) & (df_aux['speed'] >= 65)]
        df_aux3 = df_aux[(df_aux['speed'] < 65) & (df_aux['speed'] >= 55)]

        print(df_aux1[variable].describe())
        print(df_aux2[variable].describe())
        print(df_aux3[variable].describe())

        t1 = df_aux1[variable].index
        t2 = df_aux2[variable].index
        t3 = df_aux3[variable].index

        y1 = df_aux1[variable]
        y2 = df_aux2[variable]
        y3 = df_aux3[variable]

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=t1,
                                 y=y1,
                                 mode='markers',
                                 marker=dict(color='blue'),
                                 name='{} with speed in (85-75)'.format(variable)))

        fig.add_trace(go.Scatter(x=t2,
                                 y=y2,
                                 mode='markers',
                                 marker=dict(color='gray'),
                                 name='{} with speed in (75-65)'.format(variable)))

        fig.add_trace(go.Scatter(x=t3,
                                 y=y3,
                                 mode='markers',
                                 marker=dict(color='orange'),
                                 name='{} with speed in (65-55)'.format(variable)))

        fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
        fig.update_layout(showlegend=True)
        fig.update_yaxes(title_text=variable + ' ' + units_[variable])
        fig.update_xaxes(title_text='Time (15 min)')
        fig.update_layout(title='{}  - Time series '.format(road))
        # fig.update_layout(title='{}  - Time series - section: {} - segment:{}'.format(road, section_, segment_))

        # Add range slider
        fig.update_layout(
            xaxis=go.layout.XAxis(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1,
                             label="1 day",
                             step="day",
                             stepmode="backward"),
                        dict(count=3,
                             label="3 days",
                             step="day",
                             stepmode="backward"),
                        dict(count=7,
                             label="1 week",
                             step="day",
                             stepmode="backward")
                    ])
                ),
                rangeslider=dict(
                    visible=True
                ),
                type="date"
            )
        )
        # now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
        fig.write_html("figures\\{}_timeseries_by_spped_bandwith_{}.html".format(road, variable))
        fig.show()


    if 1 == 0:
        start_date = '2019-04-10 00:00:00'
        end_date = '2019-04-15 00:00:00'

        df_aux = filter_by_std(df, 'speed', 2)
        plot_time_series_by_speed_bandwith(df_aux, road, 'volume', start_date, end_date)


    def plot_time_series_smooth(df, road, variable, start_date, end_date, smooth):

        df_aux = df[(df.index > start_date) & (df.index < end_date)]

        segment_ = df_aux['segment'].unique()[0]
        df_aux = df_aux[df_aux['segment'] == segment_]

        print(df_aux[variable].describe())

        t = df_aux[variable].index
        y = df_aux[variable]

        if smooth == 0:
            print('no smooth')

        if smooth == 1:
            smooth_label = 'Savgol filter'
            from scipy.signal import savgol_filter
            yhat = savgol_filter(y, 15, 2)  # window size 51, polynomial order 3

        if smooth == 2:
            smooth_label = 'Convolute'

            def smooth(y, box_pts):
                box = np.ones(box_pts) / box_pts
                y_smooth = np.convolve(y, box, mode='same')
                return y_smooth

            yhat = smooth(y, 3)

        if smooth == 3:
            smooth_label = 'Rolling'
            yhat = y.rolling(window=5, min_periods=5, center=True).median()

        if smooth == 4:
            smooth_label = 'Interpolate'
            from scipy.interpolate import interp1d

            x = np.arange(0, len(t))
            xx = np.linspace(x, x, 1000)

            # interpolate + smooth
            itp = interp1d(t, y.values, kind='linear')

            window_size, poly_order = 101, 3
            yhat = savgol_filter(itp(xx), window_size, poly_order)

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=t,
                                 y=y,
                                 mode='markers',
                                 marker=dict(color='blue'),
                                 name='{} data'.format(variable)))
        if smooth != 0:
            fig.add_trace(go.Scatter(x=t,
                                     y=yhat,
                                     mode='lines',
                                     marker=dict(color='gray'),
                                     name='{} smooth {}'.format(variable, smooth_label)))

        fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
        fig.update_layout(showlegend=True)
        fig.update_yaxes(title_text=variable + ' ' + units_[variable])
        fig.update_xaxes(title_text='Time (15 min)')
        fig.update_layout(title='{}  - Time series'.format(road))

        # Add range slider
        fig.update_layout(
            xaxis=go.layout.XAxis(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1,
                             label="1 day",
                             step="day",
                             stepmode="backward"),
                        dict(count=3,
                             label="3 days",
                             step="day",
                             stepmode="backward"),
                        dict(count=7,
                             label="1 week",
                             step="day",
                             stepmode="backward")
                    ])
                ),
                rangeslider=dict(
                    visible=True
                ),
                type="date"
            )
        )
        # now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
        fig.write_html("figures\\{}_timeseries_smooth_{}.html".format(road, variable))
        fig.show()


    if 1 == 0:
        start_date = '2019-03-10 00:00:00'
        end_date = '2019-04-15 00:00:00'

        df_aux = filter_by_std(df, 'speed', 2)
        plot_time_series_smooth(df_aux, road, 'volume', start_date, end_date, 1)

"""
"""
"""
"""
""" ===================== Reference Pattern ========================
"""
"""
"""
"""
"""
if 1 == 1:

    def plot_reference_daily_pattern(df, road, variable):

        df_mean = df.groupby(by=['hour', 'minute'])[variable].mean()

        df_std = df.groupby(by=['hour', 'minute'])[variable].std()

        t = pd.date_range('2020-05-25 00:00:00', '2020-05-25 23:45:00', freq='15min')
        y_mean = df_mean.values

        y_upper_std = df_mean.values + df_std.values
        y_lower_std = df_mean.values - df_std.values
        y_lower_std[y_lower_std < 0] = 0

        y_upper_std2 = df_mean.values + 2 * df_std.values
        y_lower_std2 = df_mean.values - 2 * df_std.values
        y_lower_std2[y_lower_std2 < 0] = 0

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t,
                                 y=y_mean,
                                 mode='markers+lines',
                                 name='Mean Value',
                                 marker=dict(color='blue'),
                                 opacity=0.6))

        fig.add_trace(go.Scatter(x=t,
                                 y=y_upper_std,
                                 name='Mean Value + 1 Standard deviation',
                                 marker=dict(color='red'),
                                 mode='lines',
                                 opacity=0.6))

        fig.add_trace(go.Scatter(x=t,
                                 y=y_lower_std,
                                 name='Mean Value - 1 Standard deviation',
                                 marker=dict(color='red'),
                                 mode='lines',
                                 opacity=0.6))

        fig.add_trace(go.Scatter(x=t,
                                 y=y_upper_std2,
                                 name='Mean Value + 2 Standard deviation',
                                 marker=dict(color='darkred'),
                                 mode='lines',
                                 opacity=0.6))

        fig.add_trace(go.Scatter(x=t,
                                 y=y_lower_std2,
                                 name='Mean Value - 2 Standard deviation',
                                 marker=dict(color='darkred'),
                                 mode='lines',
                                 opacity=0.6))

        index_ = df.index.dayofyear.unique()

        if 1 == 0:
            for days_ in index_:
                df_aux = df[df.index.dayofyear == days_]

                df_aux = df_aux[df_aux['section'] == df_aux['section'].unique()[0]]

                fig.add_trace(go.Scatter(x=t,
                                         y=df_aux[variable].values,

                                         name='Data',
                                         marker=dict(color='gray'),
                                         mode='markers',
                                         opacity=0.6))

        fig.update_yaxes(title_text=variable + ' ' + units_[variable])
        fig.update_xaxes(title_text='time (15 min)')

        fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
        fig.update_layout(title='{}  - Time series - (Reference Pattern)'.format(road))
        fig.update_layout(showlegend=True)
        # now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
        fig.write_html("figures\\{}_reference_pattern_{}.html".format(road, variable))
        fig.show()


    if 0 == 1:
        plot_reference_daily_pattern(df, road, 'volume')


    def plot_reference_daily_pattern_by_weekend(df, road, variable):

        t = pd.date_range('2020-05-25 00:00:00', '2020-05-25 23:45:00', freq='15min')

        fig = go.Figure()

        weekend_dict = {1: 'weekend',
                        0: 'weekday'}

        for week_day_ in df['weekend'].unique():
            df_aux = df[df['weekend'] == week_day_]

            df_mean = df_aux.groupby(by=['hour', 'minute'])[variable].mean()
            df_std = df_aux.groupby(by=['hour', 'minute'])[variable].std()
            y_mean = df_mean.values
            y_upper_std1 = df_mean.values + df_std.values
            y_lower_std1 = df_mean.values - df_std.values
            y_lower_std1[y_lower_std1 < 0] = 0

            y_upper_std2 = df_mean.values + 2 * df_std.values
            y_lower_std2 = df_mean.values - 2 * df_std.values
            y_lower_std2[y_lower_std2 < 0] = 0

            fig.add_trace(go.Scatter(x=t,
                                     y=y_mean,
                                     mode='markers+lines',
                                     marker=dict(color='darkred'),
                                     name='Mean Value ({})'.format(weekend_dict[week_day_]),
                                     opacity=0.6))

            fig.add_trace(go.Scatter(x=t,
                                     y=y_upper_std1,
                                     marker=dict(color='blue'),
                                     name='Mean Value + 1 Standard deviation ({})'.format(weekend_dict[week_day_]),
                                     mode='lines',
                                     opacity=0.6))

            fig.add_trace(go.Scatter(x=t,
                                     y=y_lower_std1,
                                     marker=dict(color='blue'),
                                     name='Mean Value - 1 Standard deviation ({})'.format(weekend_dict[week_day_]),
                                     mode='lines',
                                     opacity=0.6))

            fig.add_trace(go.Scatter(x=t,
                                     y=y_upper_std2,
                                     marker=dict(color='darkblue'),
                                     name='Mean Value + 2 Standard deviation ({})'.format(weekend_dict[week_day_]),
                                     mode='lines',
                                     opacity=0.6))

            fig.add_trace(go.Scatter(x=t,
                                     y=y_lower_std2,
                                     marker=dict(color='darkblue'),
                                     name='Mean Value - 2 Standard deviation ({})'.format(weekend_dict[week_day_]),
                                     mode='lines',
                                     opacity=0.6))

        fig.update_yaxes(title_text=variable + ' ' + units_[variable])
        fig.update_xaxes(title_text='time (15 min)')

        fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
        fig.update_layout(title='{}  - Time series - (Reference Pattern)'.format(road))
        fig.update_layout(showlegend=True)
        now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
        # fig.write_html("figures\\{}_timeseries_{}_{}.html".format(road, variable, now, ))
        fig.show()


    if 1 == 0:
        plot_reference_daily_pattern_by_weekend(df, road, 'volume')


    def plot_reference_daily_pattern_by_day(df, road, variable):

        fig = go.Figure()

        days_dict = {0: 'Monday',
                     1: 'Thuesday',
                     2: 'Wednesday',
                     3: 'Thursday',
                     4: 'Friday',
                     5: 'Saturday',
                     6: 'Sunday'}

        t = pd.date_range('2020-05-25 00:00:00', '2020-05-25 23:45:00', freq='15min')
        for day_ in df['weekday'].unique():
            df_aux = df[df['weekday'] == day_]

            coef = 2
            df_mean = df_aux.groupby(by=['hour', 'minute'])[variable].mean()
            df_std = df_aux.groupby(by=['hour', 'minute'])[variable].std()
            y_mean = df_mean.values
            y_upper_std = df_mean.values + coef * df_std.values
            y_lower_std = df_mean.values - coef * df_std.values
            y_lower_std[y_lower_std < 0] = 0

            fig.add_trace(go.Scatter(x=t,
                                     y=y_mean,
                                     mode='markers+lines',
                                     marker=dict(color='darkred'),
                                     name='Mean Value (day:{})'.format(days_dict[day_]),
                                     opacity=0.6))

            fig.add_trace(go.Scatter(x=t,
                                     y=y_upper_std,
                                     mode='lines',
                                     marker=dict(color='blue'),
                                     name='Mean Value + 1 Standard deviation (day:{})'.format(days_dict[day_]),
                                     opacity=0.6))

            fig.add_trace(go.Scatter(x=t,
                                     y=y_lower_std,
                                     mode='lines',
                                     marker=dict(color='blue'),
                                     name='Mean Value - 1 Standard deviation (day:{})'.format(days_dict[day_]),
                                     opacity=0.6))

        fig.update_yaxes(title_text=variable + ' ' + units_[variable])
        fig.update_xaxes(title_text='time (15 min)')

        fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
        fig.update_layout(title='{}  - Time series - (Reference Pattern)'.format(road))
        fig.update_layout(showlegend=True)
        #  now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
        fig.write_html("figures\\{}_reference_pattern_by_day_{}.html".format(road, variable))
        fig.show()


    if 0 == 1:
        plot_reference_daily_pattern_by_day(df, road, 'volume')


    def plot_reference_daily_pattern_by_speed_bandwith(df, road, variable):

        fig = go.Figure()

        df_aux = df[(df.index > start_date) & (df.index < end_date)]

        # segment_ = df_aux['segment'].unique()[0]
        # df_aux = df_aux[df_aux['segment'] == segment_]
        # section_ = df_aux['ID'].unique()[0]

        # Plot all sections
        colors_ = [('red', 'darkred'), ('blue', 'darkblue')]

        speed_limits = [(85, 75), (75, 65)]
        t = pd.date_range('2020-05-25 00:00:00', '2020-05-25 23:45:00', freq='15min')

        for i in range(0, len(speed_limits)):

            print(i)
            df_aux1 = df_aux[(df_aux['speed'] < speed_limits[i][0]) & (df_aux['speed'] >= speed_limits[i][1])]

            df_mean = df_aux1.groupby(by=['hour', 'minute'])[variable].mean()
            df_std = df_aux1.groupby(by=['hour', 'minute'])[variable].std()
            y_mean = df_mean.values
            y_upper_1std = df_mean.values + df_std.values
            y_lower_1std = df_mean.values - df_std.values
            y_upper_2std = df_mean.values + 2 * df_std.values
            y_lower_2std = df_mean.values - 2 * df_std.values
            y_lower_1std[y_lower_1std < 0] = 0
            y_lower_2std[y_lower_2std < 0] = 0


            fig.add_trace(go.Scatter(x=t,
                                     y=y_mean,
                                     mode='markers+lines',
                                     marker=dict(color=colors_[0][i]),
                                     name='Mean Value (speed:{}-{})'.format(speed_limits[i][0], speed_limits[i][1]),
                                     opacity=0.6))

            if 0 == 1:
                fig.add_trace(go.Scatter(x=t,
                                         y=y_upper_1std,
                                         mode='lines',
                                         marker=dict(color='blue'),
                                         name='Mean Value + 1 Standard deviation (speed:{}-{})'.format(speed_limits[i][0], speed_limits[i][1]),
                                         opacity=0.6))

                fig.add_trace(go.Scatter(x=t,
                                         y=y_lower_1std,
                                         mode='markers',
                                         marker=dict(color='blue'),
                                         name='Mean Value - 1 Standard deviation (speed:{}-{})'.format(speed_limits[i][0], speed_limits[i][1]),
                                         opacity=0.6))

            fig.add_trace(go.Scatter(x=t,
                                     y=y_upper_2std,
                                     mode='markers',
                                     marker=dict(color=colors_[1][i]),
                                     name='Mean Value + 2 Standard deviation (speed:{}-{})'.format(speed_limits[i][0], speed_limits[i][1]),
                                     opacity=0.6))

            fig.add_trace(go.Scatter(x=t,
                                     y=y_lower_2std,
                                     mode='markers',
                                     marker=dict(color=colors_[1][i]),
                                     name='Mean Value - 2 Standard deviation (speed:{}-{})'.format(speed_limits[i][0], speed_limits[i][1]),
                                     opacity=0.6))

        fig.update_yaxes(title_text=variable + ' ' + units_[variable])
        fig.update_xaxes(title_text='time (15 min)')

        fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
        fig.update_layout(title='{}  - Time series - (Reference Pattern)'.format(road))
        fig.update_layout(showlegend=True)
        #now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
        fig.write_html("figures\\{}_reference_pattern_by_speed_bandwith_{}.html".format(road, variable))
        fig.show()


    if 0 == 1:
        df_aux = filter_by_std(df, 'speed', 2)
        plot_reference_daily_pattern_by_speed_bandwith(df_aux, road, 'volume')



"""
    This script produce the following function plots:

    lineplots
    barplots
    histograms
    congestion matrix (heat map)
    correlation matrix (heat map)
    boxplots
    scatterplots
    timeseries
    reference patterns

    @ Jose Angel Velasco (javelacor@indra.es)
    (C) Indra Digital Labs 2020
"""





""" ===================== line plots ===================
"""


def plot_line(df, var_x, var_y, var_color, road):
    """ Plot sinmple time series volume with color per algorithm
    :param df:
    :param var_x:
    :param var_y:
    :param var_color:
    :return:
    """
    import plotly.express as px
    from dl_ia_cla_settings import config_plotly
    config_plotly()

    show = True
    print_ = True

    fig = px.line(df,
                  x=var_x,
                  y=var_y,
                  color=var_color)
    if print_:
        fig.write_html("figures\\plot_line_{}_{}_{}.html".format(road, var_x, var_y))
    if show:
        fig.show()





""" ===================== Bar plots ===================
"""



def plot_bar_df(df, variable):
    """ produce a bar plot with pandas function
    :param df:
    :param variable:
    :return:
    """
    df.sort_values(by=[variable], ascending=False, inplace=True)
    df.plot(kind='bar',
            ylim=(0, 100))


def plot_bar(df, variable, road, var_group):
    """ produce a barplot with plotly

    :param df:
    :param road:
    :return:
    """
    import plotly.express as px
    from dl_ia_cla_settings import config_plotly
    config_plotly()

    print_ = True
    show=  True
    error = 0

    try:
        fig = px.bar(df, x=var_group, y=variable)
        fig.update_yaxes(title_text='')
        fig.update_xaxes(title_text='Variables')

        fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
        fig.update_layout(title='Bar plot - {}'.format(road))
        fig.update_layout(showlegend=True)

        if print_:
            fig.write_html("figures\\{}\\plot_bar_{}.html".format(road, variable))
        if show:
            fig.show()

        return error

    except Exception as exception_msg:

        ECLA_comm('(!) Error in plot_bar: ' + str(exception_msg))
        error = 1
        return error



def plot_bar_frequency(df, road):
    """ Description
    :param df:
    :param variable:
    :return:
    """
    import plotly.graph_objects as go
    from dl_ia_cla_settings import create_time_array

    t = create_time_array('00:00:00', '23:59:59', '15min')

    df_count = df.groupby(by=['hour', 'minute'])['volume'].sum()



    volume_relative = 100 * df_count.values / df_count.values.sum()


    fig = go.Figure([go.Bar(x=t, y=volume_relative)])
    fig.update_yaxes(title_text=' Freq (%)')
    fig.update_xaxes(title_text='Time steps (15 min)')

    fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
    fig.update_layout(title='{}  - Freq. distribution - Volume'.format(road))
    fig.update_layout(showlegend=True)
    fig.write_html("figures\\{}_freq_barplot_volume.html".format(road))
    fig.show()




""" ===================== Histograms ===================
"""



def plot_histogram(df, variable, n_bins_, road):
    """ plot a histogram using plotly from a vairable in a dataframe

    :param df: Data frame with the variable
    :param variable: name of the variable (column name)
    :param n_bins_: number of bins of the histogram
    :param road: name of the road asset
    :return error: error code 0: everything ok, 1: something happened
    """

    import plotly.express as px
    from dl_ia_cla_settings import units_, config_plotly, CLA_comm, dl_ia_cla_settings_check_folder
    import numpy as np
    config_plotly()

    print_ = True
    show = True
    fontsize_ = 18
    error = 0

    try:
        max_value = int(df[variable].max())
        x_axis = np.arange(0, max_value, int(max_value / 20))

        fig = px.histogram(df, x = variable, nbins = n_bins_, marginal="box", hover_data=df.columns)
        fig.update_xaxes(title_text=variable + ' ' + units_[variable])
        fig.update_layout(font=dict(family="Courier New, monospace", size=fontsize_, color="#7f7f7f"))
        fig.update_layout(title='Histogram - {} - {}'.format(road, variable))
        fig.update_layout(showlegend=True)
        fig.update_traces(opacity=0.9)
        fig.update_layout(bargap=0.2)  # gap between bars of adjacent location coordinates
        fig.update_xaxes(ticktext=x_axis, tickvals=x_axis)
        # now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
        if print_:
            dl_ia_cla_settings_check_folder('figures\\{}'.format(road))
            fig.write_html("figures\\{}\\plot_histogram_{}.html".format(road, variable))

        if show:
            fig.show()
        return error
    except Exception as exception_msg:
        CLA_comm('(!) Error in plot_histogram: ' + str(exception_msg))
        error = 1
        return error


def plot_distribution(df, variable, road):
    """ plot the distribution using plotly from a vairable in a dataframe

    :param df: Data frame with the variable
    :param variable: name of the variable (column name)
    :param road: name of the road asset
    :return error: error code 0: everything ok, 1: something happened
    """

    import plotly.figure_factory as ff
    from dl_ia_cla_settings import units_, config_plotly, CLA_comm
    import numpy as np
    config_plotly()

    print_ = True
    show = True
    fontsize_ = 18
    error = 0

    try:
        max_value = int(df[variable].max())
        x_axis = np.arange(0, max_value, int(max_value / 20))

        fig = ff.create_distplot([list(df[variable].values)], variable)
        fig.update_xaxes(title_text=variable + ' ' + units_[variable])
        fig.update_layout(font=dict(family="Courier New, monospace", size=fontsize_, color="#7f7f7f"))
        fig.update_layout(title='Distribution - {} - {}'.format(road, variable))
        fig.update_layout(showlegend=True)
        fig.update_traces(opacity=0.9)
        fig.update_layout(bargap=0.2)  # gap between bars of adjacent location coordinates
        fig.update_xaxes(ticktext=x_axis, tickvals=x_axis)

        if print_:
            fig.write_html("figures\\{}\\plot_distribution_{}.html".format(road, variable))
        if show:
            fig.show()
        return error
    except Exception as exception_msg:
        CLA_comm('(!) Error in plot_distribution: ' + str(exception_msg))
        error = 1
        return error

def plot_histogram_mpl(df, variable, n_bins, road):
    """

    :param df: Data frame with the vairable to plot
    :param variable: name of the variable (column name)
    :param n_bins_: number of bins of the histogram
    :return:
    """

    import matplotlib.pyplot as plt
    from dl_ia_cla_settings import units_
    from datetime import datetime
    import numpy as np

    print_ = True
    show = True
    fontsize_ = 18

    fig = plt.figure(figsize=(10, 8), dpi=150, facecolor='w', edgecolor='w')
    ax = fig.add_subplot(1, 1, 1)

    # fig, ax = plt.subplots(1, 1, sharey=True, tight_layout=True)
    n = ax.hist(df[variable], bins=n_bins, edgecolor='b')
    ax.set_xlabel(variable + ' ' + units_[variable], fontsize=fontsize_)
    ax.set_ylabel('Count', fontsize=fontsize_)
    xmax = df[variable].max()
    x_range = list(np.arange(0, xmax, int(xmax / 20)))
    # ax.set_xticks = x_range
    # ax.set_xticklabels = x_range
    ax.tick_params(axis='both', labelsize=fontsize_)
    now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
    plt.title('{} - Histogram - {} '.format(road, variable), fontsize=fontsize_)
    if show:
        plt.show()

    if print_:
        plt.savefig('figures\\{}_histogram_{}.png'.format(road, variable),
                    bbox_inches='tight', pad_inches=0)




"""  =================== Congestion matrix ====================
"""



def plot_congestion_matrix(df_congestions):
    """

    :param df: Data frame with the vairable to plot
    :param variable: name of the variable (column name)
    :param n_bins_: number of bins of the histogram
    :return:
    """
    congestion = df_congestions.values
    r = round(congestion, 2)

    import plotly.figure_factory as ff

    print_ = True
    show = True

    fig = ff.create_annotated_heatmap(congestion,
                                      colorscale='Viridis')
    fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
    fig.update_layout(title='Congestion Matrix - Brisbane')
    fig.update_layout(showlegend=True)

    if print_:
        fig.write_html("figures\\plot_congestion_matrix_{}.html".format(road))
    if show:
        fig.show()




"""  =================== Correlation matrix ====================
"""



def plot_correlation_matrix(df, road):
    """ This function calcualte the correlation coefficient between the columns of a dataframe and plot
        in a heat map using plotly

    :param df: dataframe with the columns being the variables
    :param road: road of the variables
    :return R: dataframe with the correlations
    """

    import plotly.figure_factory as ff
    from dl_ia_cla_settings import CLA_comm, dl_ia_cla_settings_check_folder
    from dl_ia_cla_settings import config_plotly
    config_plotly()

    error = 0
    print_ = True
    show = True

    x_ = df.columns
    y_ = df.columns

    R = df.corr()
    r = round(R, 2)

    try:
        fig = ff.create_annotated_heatmap(r.values,
                                          x = list(r.columns),
                                          y = list(r.columns),
                                          colorscale='Viridis')
        fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
        fig.update_layout(title='{} - Correlation Matrix (Pearson correlation coefficient)'.format(road))
        fig.update_layout(showlegend=True)
        # now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
        if show:
            fig.show()
        if print_:
            dl_ia_cla_settings_check_folder('figures\\{}'.format(road))
            fig.write_html("figures\\{}\\correlation_matrix.html".format(road))

        return error, R

    except Exception as exception_msg:
        CLA_comm('(!) Error in plot_correlation_matrix: ' + str(exception_msg))
        error = 1
        R = 0
        return error, R








"""  =========================== Boxplots ===========================
"""



def boxplot_plotly(df, var, var_group, var_color, road):
    """ This function do this-

    :param df:
    :param road:
    :param columns_:
    :return:
    """
    import plotly.express as px
    from dl_ia_cla_settings import units_

    print_ = True
    show = True

    fig = px.box(df, y=var, x=var_group, color=var_color)
    fig.update_xaxes(title_text=var_group)
    fig.update_yaxes(title_text='{}'.format(var + ' ' + units_[variable]))
    fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
    fig.update_layout(title='Boxplot - {} - {}'.format(road, variable))
    fig.update_layout(showlegend=True)

    # now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")

    if print_:
        fig.write_html(
        "figures\\{}\\boxplot_{}_{}_{}.html".format(road, var, var_group, var_color))
    if show:
        fig.show()




""" ================= Scatter plots ============================== 
"""





def plot_scatterplot(df, var_x, var_y, var_color, var_size, road):
    """ Produce a simple scatter plot with plotly

    :param df: dataframe that contains the variables
    :param variable_x: variable to plot in x axis
    :param variable_y: variable to plot in y axis
    :param variable_to_color: variable to use as color
    :param variable_to_color: variable to use as size
    :return:
    """
    import plotly.express as px
    from dl_ia_cla_settings import units_, CLA_comm
    from dl_ia_cla_settings import config_plotly, dl_ia_cla_settings_check_folder
    config_plotly()

    print_ = True
    show = True
    error = 0

    try:
        fig = px.scatter(df,
                         x = var_x,
                         y = var_y,  # marker = dict(color='blue')
                         size = var_size,
                         color = var_color)

        fig.update_xaxes(title_text=var_x + ' ' + units_[var_x])
        fig.update_yaxes(title_text=var_y + ' ' + units_[var_y])
        fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
        fig.update_layout(title='Scatterplot - {}'.format(road))
        fig.update_layout(showlegend=True)
        # now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")

        if print_:
            dl_ia_cla_settings_check_folder('figures\\{}'.format(road))
            fig.write_html("figures\\{}\\scatterplot_{}_{}.html".format(road, var_x, var_y))
        if show:
            fig.show()

        return error


    except Exception as exception_msg:

        CLA_comm('(!) Error in plot_scatterplot: ' + str(exception_msg))
        error = 1
        return error





def plot_scatterplot_mpl(df, variable_x, variable_y, group_variable, road):
    """
    Scatterplot matplotlib individual with color by variable

    :param df:
    :param variable_x:
    :param variable_y:
    :param group_variable:
    :param road:
    :return:
    """

    import matplotlib.pyplot as plt
    from dl_ia_cla_settings import units_
    from datetime import datetime

    fig = plt.figure(figsize=(12, 8), dpi=150, facecolor='w', edgecolor='w')
    ax = fig.add_subplot(1, 1, 1)

    for var in df[group_variable].unique():
        df_aux = df[df[group_variable] == var]
        x = df_aux[variable_x].values
        y = df_aux[variable_y].values
        ax.scatter(x, y, label=var, alpha=0.8, edgecolor='k')

    fontsize_ = 12
    ax.set_xlabel(variable_x + ' ' + units_[variable_x], fontsize=fontsize_)
    ax.set_ylabel(variable_y + ' ' + units_[variable_y], fontsize=fontsize_)
    ax.tick_params(axis='both', labelsize=fontsize_)
    plt.legend()

    now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
    plt.title('{}'.format(road), fontsize=fontsize_)
    plt.show()

    plt.savefig('figures\\{}_scatterplot_{}_{}.png'.format(road, variable_y, variable_x),
                bbox_inches='tight',
                pad_inches=0)
    plt.close()




def plot_scatterplot_mpl2(df):
    """ This function do this-

    :param df:
    :param road:
    :param columns_:
    :return:
    """
    fontsize_ = 12

    import matplotlib.pyplot as plt
    from dl_ia_cla_settings import units_

    print_ = True
    show = True

    fig = plt.figure(figsize=(12, 6), dpi=90, facecolor='w', edgecolor='k')

    ax1 = fig.add_subplot(1, 1, 1)
    var_x = 'flow'
    var_y = 'speed'
    ax1.scatter(df[var_x], df[var_y], marker='o', s=3, c='b')
    ax1.set_ylabel(units_[var_y], fontsize=fontsize_)
    ax1.set_xlabel(units_[var_x], fontsize=fontsize_)

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.scatter(df['occupancy'], df['speed'])
    ax2.set_ylabel(units_['speed'], fontsize=fontsize_)
    ax2.set_xlabel(units_['occupancy'], fontsize=fontsize_)

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.scatter(df['flow'], df['occupancy'])
    ax3.set_ylabel(units_['speed'], fontsize=fontsize_)
    ax3.set_xlabel(units_['occupancy'], fontsize=fontsize_)

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.scatter(df['occupancy'], df['volume'])
    ax4.set_ylabel(units_['speed'], fontsize=fontsize_)
    ax4.set_xlabel(units_['occupancy'], fontsize=fontsize_)

    if show:
        plt.show()
    if print_:
        plt.savefig('figures\\plot_scatterplot_matplotlib.pdf', bbox_inches='tight', pad_inches=0.1)
    # plt.close()


def plot_scatterplot_mpl_multi(df, variable_x, variable_y, group_var, road):
    """
        Scatterplot matplotlib (4 plots)

    :param df:
    :param variable_x:
    :param variable_y:
    :param group_var:
    :param road:
    :return:
    """
    fontsize_ = 12
    marker_size_ = 10

    import matplotlib.pyplot as plt
    from dl_ia_cla_settings import units_
    from datetime import datetime
    import numpy as np

    fig = plt.figure(figsize=(12, 8), dpi=150, facecolor='w', edgecolor='w')

    for i in np.arange(0, len(variable_x)):
        ax1 = fig.add_subplot(2, 2, i + 1)
        sc1 = ax1.scatter(df[variable_x[i]],
                          df[variable_y[i]], marker='o', s=marker_size_, alpha=0.8, c=df[group_var])
        cbar1 = plt.colorbar(sc1)
        cbar1.set_label(group_var, rotation=270, fontsize=fontsize_)
        ax1.set_xlabel(variable_x[i] + ' ' + units_[variable_x[i]], fontsize=fontsize_)
        ax1.set_ylabel(variable_y[i] + ' ' + units_[variable_y[i]], fontsize=fontsize_)

        y_min = df[variable_y[i]].min()
        y_max = df[variable_y[i]].max()
        x_min = df[variable_x[i]].min()
        x_max = df[variable_x[i]].max()
        y_range = np.arange(y_min, y_max, round(y_max / 5))
        x_range = np.arange(x_min, x_max, round(x_max / 5))
        ax1.set_yticks(y_range)
        ax1.set_yticklabels(y_range)
        ax1.set_xticks(x_range)
        ax1.set_xticklabels(x_range)
        ax1.tick_params(axis='both', labelsize=fontsize_)

    now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
    plt.title('{}'.format(road), fontsize=fontsize_)
    plt.subplots_adjust(wspace=0.2, hspace=0.4)
    plt.show()
    plt.savefig('figures\\{}_scatterplot_multi_{}.png'.format(road, now),
                bbox_inches='tight', pad_inches=0)

    plt.close()




def plot_scatterplot_by_color(df, variable_x, variable_y, color_variable, road):
    """
    Statter plot plotly individual all variables segmented by schedule (!) ALL COMBINATIONS (6x7=42 plots)

    :param df:
    :param variable_x:
    :param variable_y:
    :param color_variable:
    :param road:
    :return:
    """
    import plotly.express as px
    from dl_ia_cla_settings import units_

    fig = px.scatter(df,
                     x=variable_x,
                     y=variable_y,
                     color=color_variable,
                     opacity=0.8,
                     hover_data=[color_variable])

    fig.update_xaxes(title_text=variable_x + ' ' + units_[variable_x])
    fig.update_yaxes(title_text=variable_y + ' ' + units_[variable_y])
    fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
    fig.update_layout(title='{} - Scatterplots'.format(road))
    fig.update_layout(showlegend=True)
    # now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
    # fig.show()
    fig.write_html(
        "figures\\{}_scatterplot_by_color_{}_{}.html".format(road, variable_x, variable_y))




def plot_scatterplot_by_color_multi(df, variable_x, variable_y, road):
    """
    Plotly scatterplot multiple with all values and segmented by schedule time



    :param df:
    :param variable_x:
    :param variable_y:
    :param road:
    :return:
    """

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import numpy as np

    fig = make_subplots(rows=2, cols=3)

    index = {'0-6': (1, 1),
             '7-12': (1, 2),
             '13-17': (1, 3),
             '18-21': (2, 1),
             '22-23': (2, 2),
             }

    colors_ = {'0-6': 'darkblue',
               '7-12': 'darkgreen',
               '13-17': 'darkred',
               '18-21': 'darkviolet',
               '22-23': 'darkgrey',
               }

    for schedule_ in df['schedule_def'].unique():
        df_aux = df.loc[df['schedule_def'] == schedule_]

        fig.add_trace(go.Scatter(x=df_aux[variable_x].values,
                                 y=df_aux[variable_y].values,
                                 name=schedule_,
                                 mode='markers',
                                 marker_color=colors_[schedule_]), row=index[schedule_][0], col=index[schedule_][1])

        x_axis = np.arange(0, int(df[variable_x].max()), int(df[variable_x].max() / 10))
        y_axis = np.arange(0, int(df[variable_y].max()), int(df[variable_y].max() / 10))
        fig.update_yaxes(title_text=variable_y,
                         tickfont=dict(size=14),
                         ticktext=y_axis,
                         tickvals=y_axis, row=index[schedule_][0], col=index[schedule_][1])

        fig.update_xaxes(title_text=variable_x,
                         tickfont=dict(size=14),
                         ticktext=x_axis,
                         tickvals=x_axis, row=index[schedule_][0], col=index[schedule_][1])

    fig.update_layout(showlegend=True)
    fig.update_layout(font=dict(family="Courier New, monospace", size=14, color="#7f7f7f"))
    fig.update_layout(title_text='{} - Scatterplots '.format(road))
    # now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
    # fig.show()
    fig.write_html("figures\\{}_scatterplot_multi_{}_{}_by_schedule_{}.html".format(road, variable_x, variable_y))




def plot_scatterplot_by_schedule_multi(df, variable_x, variable_y, road):
    """
    Plotly scatterplot individual with values segmented by schedule time

    :param df:
    :param variable_x:
    :param variable_y:
    :param road:
    :return:
    """

    from dl_ia_cla_settings import colors_, units_
    import plotly.graph_objects as go
    from datetime import datetime
    import numpy as np

    for schedule_ in df['schedule_def'].unique():
        df_aux = df.loc[df['schedule_def'] == schedule_]

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=df[variable_x].values,
                                 y=df[variable_y].values,
                                 name='All',
                                 mode='markers',
                                 opacity=0.9,
                                 marker_color='lightgrey'))

        fig.add_trace(go.Scatter(x=df_aux[variable_x].values,
                                 y=df_aux[variable_y].values,
                                 name=schedule_,
                                 mode='markers',
                                 marker_color=colors_[schedule_]))

        x_axis = np.arange(0, int(df[variable_x].max()), int(df[variable_x].max() / 10))
        y_axis = np.arange(0, int(df[variable_y].max()), int(df[variable_y].max() / 10))
        fig.update_yaxes(title_text=variable_y + units_[variable_y],
                         tickfont=dict(size=14),
                         ticktext=y_axis,
                         tickvals=y_axis)

        fig.update_xaxes(title_text=variable_x + units_[variable_x],
                         tickfont=dict(size=14),
                         ticktext=x_axis,
                         tickvals=x_axis)

        fig.update_layout(showlegend=True)
        fig.update_layout(font=dict(family="Courier New, monospace", size=14, color="#7f7f7f"))
        fig.update_layout(title_text='{} - Scatterplot '.format(road))
        now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
        fig.write_html(
            "figures\\{}_scatterplot_multi_{}_{}_schedule_{}_{}.html".format(road, variable_x, variable_y,
                                                                             schedule_, now))
        # fig.show()



def plot_scatterplot_with_histogram(df, variable_y, variable_x, grouping_variable, road):
    """

    :param variable_y:
    :param variable_x:
    :param grouping_variable:
    :param road:
    :return:
    """
    ""

    import matplotlib.pyplot as plt
    from dl_ia_cla_settings import units_
    import numpy as np

    # definitions for the axes
    left, width = 0.1, 0.4
    bottom, height = 0.1, 0.4
    spacing = 0.005

    fontsize_ = 16

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]

    # start with a square Figure
    fig = plt.figure(figsize=(12, 8), dpi=150, facecolor='w', edgecolor='w')

    ax = fig.add_axes(rect_scatter)
    ax_histx = fig.add_axes(rect_histx, sharex=ax)
    ax_histy = fig.add_axes(rect_histy, sharey=ax)

    x = df[variable_x].values
    y = df[variable_y].values

    color_ = df[grouping_variable].values

    # now determine nice limits by hand:
    binwidth = 0.5
    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    lim = (int(xymax / binwidth) + 1) * binwidth
    bins = np.arange(0, lim + binwidth, binwidth)

    ax_histx.hist(x, bins=bins)
    ax_histy.hist(y, bins=bins, orientation='horizontal')
    ax_histx.axis('off')
    ax_histy.axis('off')

    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    sc = ax.scatter(x, y, c=color_)
    cbar = plt.colorbar(sc)
    cbar.set_label(grouping_variable, rotation=270, fontsize=fontsize_)

    xmax = int(df[variable_x].max())
    ymax = int(df[variable_y].max())

    ax.set_xlabel(variable_x + ' ' + units_[variable_x], fontsize=fontsize_)
    ax.set_ylabel(variable_y + ' ' + units_[variable_y], fontsize=fontsize_)

    ax.tick_params(axis='both', labelsize=fontsize_)
    ax.set_xticks = list(np.arange(0, xmax, int(xmax / 10)))
    ax.set_yticks = list(np.arange(0, ymax, int(ymax / 10)))
    ax.set_xticklabels = list(np.arange(0, xmax, int(xmax / 10)))
    ax.set_yticklabels = list(np.arange(0, ymax, int(ymax / 10)))

    # now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")

    plt.title('{}'.format(road), fontsize=fontsize_)
    plt.show()
    plt.savefig('figures\\{}_scatterplot_with_histogram_{}_{}.png'.format(road, variable_y, variable_x),
                bbox_inches='tight', pad_inches=0)
    plt.close()





"""  ==================== Time series =======================
"""



def plot_time_series_mpl(df, start_date, end_date, road, variable):
    """
    Plot time series with matplotlib, one variable per plot,
    segmented by section, for a specific data range
    The plot is saved in png format

    :param df:
    :param start_date:
    :param end_date:
    :param road:
    :return:
    """
    fontsize_ = 18

    import matplotlib.pyplot as plt
    from dl_ia_cla_settings import units_

    fig = plt.figure(figsize=(25, 10), dpi=150, facecolor='w', edgecolor='w')
    ax = fig.add_subplot(1, 1, 1)

    for section_ in df['sub_section'].unique():
        df_aux = df[df['sub_section'] == section_]
        y = df_aux.loc[start_date:end_date, [variable]]
        ax.plot(y, '-o', label='section:{}'.format(section_))

    ax.set_xlabel('time (15 min)', fontsize=fontsize_)
    ax.set_ylabel(variable + units_[variable], fontsize=fontsize_)
    ax.tick_params(axis='both', labelsize=fontsize_)
    ax.legend(fontsize=fontsize_)
    ax.set_title('{} - Time Series - {} to {}'.format(road, start_date, end_date), fontsize=fontsize_)
    # now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
    plt.subplots_adjust(wspace=0.2, hspace=0.4)
    plt.show()
    plt.savefig('figures\\{}_timeseries_{}.png'.format(road, variable), bbox_inches='tight', pad_inches=0)
    plt.close()



def plot_timeseries_with_slider(df, var_x, var_y, road):
    """ Plot simple time series with time slider

    :param df: dataframe with timestamp index and variable
    :param variable: name of the variable
    :param road: name of the road/asset
    :return:
    """

    import plotly.graph_objects as go
    from dl_ia_cla_settings import units_, CLA_comm, dl_ia_cla_settings_check_folder

    print_ = True
    show = True
    error = 0

    try:
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=df[var_x],
                                 y=df[var_y],
                                 mode='markers+lines',
                                 marker=dict(color='red'),
                                 name='{}'.format(var_y)))

        # Add range slider
        fig.update_layout(
            xaxis=go.layout.XAxis(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1,
                             label="1 day",
                             step="day",
                             stepmode="backward"),
                        dict(count=3,
                             label="3 days",
                             step="day",
                             stepmode="backward"),
                        dict(count=7,
                             label="1 week",
                             step="day",
                             stepmode="backward"),
                    ])
                ),
                rangeslider=dict(
                    visible=True
                ),
                type="date"
            )
        )

        fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
        fig.update_layout(showlegend=True)
        fig.update_yaxes(title_text=var_y + ' ' + units_[var_y])
        fig.update_xaxes(title_text='Time')
        fig.update_layout(title='Timeseries')

        if print_:
            dl_ia_cla_settings_check_folder('figures\\{}'.format(road))
            fig.write_html("figures\\{}\\timeseries_with_slider_{}.html".format(road, var_y))
        if show:
            fig.show()

    except Exception as exception_msg:
        CLA_comm('(!) Error in plot_timeseries_with_slider: ' + str(exception_msg))
        error = 1
        return error




def plot_timeseries(df, var_x, var_y, road):
    """ Plot simple time series with time slider

    :param df: dataframe with timestamp index and variable
    :param variable: name of the variable
    :param road: name of the road/asset
    :return:
    """

    import plotly.graph_objects as go
    from dl_ia_cla_settings import units_, CLA_comm, dl_ia_cla_settings_check_folder

    print_ = True
    show = True
    error = 0

    try:
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=df[var_x],
                                 y=df[var_y],
                                 mode='markers+lines',
                                 marker=dict(color='red'),
                                 name='{}'.format(var_y)))


        fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
        fig.update_layout(showlegend=True)
        fig.update_yaxes(title_text=variable + ' ' + units_[var_y])
        fig.update_xaxes(title_text='Time')
        fig.update_layout(title='Timeseries - {} - {}'.format(road, var_y))

        if print_:
            dl_ia_cla_settings_check_folder('figures')
            fig.write_html("figures\\{}\\timeseries_with_slider_{}.html".format(road, var_y))
        if show:
            fig.show()

    except Exception as exception_msg:

        CLA_comm('(!) Error in plot_timeseries: ' + str(exception_msg))
        error = 1
        return error


def plot_time_Series(df, variable, road):
    """ Produce the time series plot

    :param df: data frame with the variable to plot, the time index, and the columns of section and lanes
    :param variable: name of the road traffic variable to plot
    """

    import plotly.graph_objects as go
    from dl_ia_cla_settings import units_
    import numpy as np

    fig = go.Figure()

    for section_ in np.sort(df['section'].unique()):
        df_aux = df[df['section'] == section_]
        for lane_ in np.sort(df_aux['lanes'].unique()):
            df_aux2 = df_aux[df_aux['lanes'] == lane_]
            df_aux2.sort_values(by=['time'], inplace=True)

            fig.add_trace(go.Scatter(x=df_aux2['time'],
                                     y=df_aux2[variable],
                                     line=dict(width=2, dash='dot'),
                                     mode='markers+lines',
                                     name=variable + ' ' + units_[variable] + '{}'.format(section_) + '{}'.format(lane_)))

    fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
    fig.update_layout(
        title='{} - timeseries '.format(road),
        xaxis_title='Time',
        yaxis_title='',
        showlegend=True
    )

    # Add range slider
    fig.update_layout(
        xaxis=go.layout.XAxis(
            rangeselector=dict(
                buttons=list([
                    dict(count=1,
                         label="1 month",
                         step="month",
                         stepmode="backward"),
                    dict(count=6,
                         label="6 month",
                         step="month",
                         stepmode="backward"),
                    dict(count=7,
                         label="1 week",
                         step="day",
                         stepmode="backward"),
                    dict(count=1,
                         label="1 year",
                         step="year",
                         stepmode="backward"),
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            type="date"
        )
    )
    fig.show()
    fig.write_html("figures\\{}\\plot_time_Series_{}.html".format(road, road))


def time_series_comparison(time_index, y1, y2, label1, label2):
    """ Plot two time series with the same time index

    :param time_index:
    :param y1:
    :param y2:
    :param label1:
    :param label2:
    :return:
    """

    import plotly.graph_objects as go
    from dl_ia_cla_settings import units_, config_plotly
    config_plotly()

    print_ = True
    show = True
    title = 'time_series_comparison'

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_index,
                             y=y1,
                             mode='markers',
                             marker=dict(color='red'),
                             name='{}'.format(label1)))

    fig.add_trace(go.Scatter(x=time_index,
                             y=y2,
                             mode='markers+lines',
                             marker=dict(color='blue'),
                             name='{}'.format(label2)))

    fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
    fig.update_layout(showlegend=True)
    fig.update_yaxes(title_text=label1 + ' '+ units_[label1])
    fig.update_xaxes(title_text='Time')
    fig.update_layout(title='Time series comparison')

    # Add range slider
    fig.update_layout(
        xaxis=go.layout.XAxis(
            rangeselector=dict(
                buttons=list([
                    dict(count=1,
                         label="1 day",
                         step="day",
                         stepmode="backward"),
                    dict(count=3,
                         label="3 day",
                         step="day",
                         stepmode="backward"),
                    dict(count=7,
                         label="1 week",
                         step="day",
                         stepmode="backward"),
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            type="date"
        )
    )
    if print_:
        fig.write_html("figures\\{}_{}_{}.html".format(title, label1, label2))
    if show:
        fig.show()





def plot_time_Series_slider(df, variable, road):
    """ plot a time series with a time slider

    :param df: dataframe with the data
    :return:
    """
    from dl_ia_cla_settings import units_
    import plotly.graph_objects as go


    subsection_ = df['sub_section'].unique()[0]
    df_aux = df[df['sub_section'] == subsection_]

    lane_ = df_aux[df_aux['sub_section'] == subsection_]['lanes'].unique()[0]
    section_ = df_aux[df_aux['sub_section'] == subsection_]['section'].unique()[0]

    df_aux['date'] = df_aux.index


    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df_aux['date'],
                             y=df_aux[variable],
                             name=variable + ' ' + units_[variable]))

    fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
    fig.update_layout(
        title='{} - timeseries - Section:{}-Lane:{}'.format(road, int(section_), int(lane_)),
        xaxis_title='Time',
        yaxis_title='',
        showlegend=True
    )

    # Add range slider
    fig.update_layout(
        xaxis=go.layout.XAxis(
            rangeselector=dict(
                buttons=list([
                    dict(count=1,
                         label="1 month",
                         step="month",
                         stepmode="backward"),
                    dict(count=6,
                         label="6 month",
                         step="month",
                         stepmode="backward"),
                    dict(count=7,
                         label="1 week",
                         step="day",
                         stepmode="backward"),
                    dict(count=1,
                         label="1 year",
                         step="year",
                         stepmode="backward"),
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            type="date"
        )
    )
    fig.show()


def plot_time_series_by_date(df, variable, start_date, end_date):
    """

    :param df:
    :param variable:
    :param start_date:
    :param end_date:
    :return:
    """

    from dl_ia_cla_settings import units_
    import plotly.graph_objects as go

    # filter by dates
    df_aux = df[(df.index > start_date) & (df.index < end_date)]

    # filter by sectopm
    section_ = 1
    df_aux = df_aux[(df_aux['section'] == section_)]

    # filter by lanes
    df_aux = df_aux[(df['lanes'] == 1)]

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df_aux[variable].index,
                             y=df_aux[variable],
                             mode='markers',
                             name=variable + ' ' + units_[variable]))

    fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
    fig.update_layout(showlegend=True)
    fig.update_yaxes(title_text=variable + ' ' + units_[variable])
    fig.update_xaxes(title_text='Time (15 min)')
    fig.update_layout(title='{}  - Time series dates: {} to {} section:{}'.format(road, start_date, end_date, section_))

    # Add range slider
    fig.update_layout(
        xaxis=go.layout.XAxis(
            rangeselector=dict(
                buttons=list([
                    dict(count=1,
                         label="1 day",
                         step="day",
                         stepmode="backward"),
                    dict(count=3,
                         label="3 days",
                         step="day",
                         stepmode="backward"),
                    dict(count=7,
                         label="1 week",
                         step="day",
                         stepmode="backward")
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            type="date"
        )
    )
    # now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
    fig.write_html("figures\\{}_timeseries_{}.html".format(road, variable))
    # fig.show()



def plot_time_series_segmented(df, start_date, end_date, road, variable, group_variable):
    """
        definition

    :param df:
    :param start_date:
    :param end_date:
    :param road:
    :return:
    """

    import plotly.graph_objects as go
    from dl_ia_cla_settings import units_

    fig = go.Figure()

    for group in df[group_variable].unique():
        df_aux = df[df[group_variable] == group]
        df_aux = df_aux.loc[start_date:end_date]

        t = df_aux.index
        y = df_aux[variable]

        fig.add_trace(go.Scatter(x=t,
                                 y=y,
                                 mode='markers',
                                 name=group,
                                 opacity=0.6))

    # fig.update_xaxes(title_text='Time (15 min)')
    fig.update_yaxes(title_text=variable + ' ' + units_[variable])

    fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
    fig.update_layout(title='{}  - Time series - {} to {}'.format(road, start_date, end_date))
    fig.update_layout(showlegend=True)

    # Add range slider
    fig.update_layout(
        xaxis=go.layout.XAxis(
            rangeselector=dict(
                buttons=list([
                    dict(count=1,
                         label="1 day",
                         step="day",
                         stepmode="backward"),
                    dict(count=3,
                         label="3 days",
                         step="day",
                         stepmode="backward"),
                    dict(count=7,
                         label="1 week",
                         step="day",
                         stepmode="backward")
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            type="date"
        )
    )

    # now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
    fig.write_html("figures\\{}_timeseries_segmented_{}.html".format(road, variable))
    # fig.show()




def plot_time_series_by_section(df, road, variable, start_date, end_date):
    """
        definition

    :param df:
    :param start_date:
    :param end_date:
    :param road:
    :return:
    """

    df_aux = df[(df.index > start_date) & (df.index < end_date)]
    section_ = df_aux['section'].unique()[0]

    df_aux = df_aux[df_aux['section'] == section_]


    import plotly.graph_objects as go
    from dl_ia_cla_settings import units_

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df_aux[variable].index,
                             y=df_aux[variable],
                             mode='markers',
                             name='{}-Section:{}'.format(variable, str(section_))))

    fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
    fig.update_layout(showlegend=True)
    fig.update_yaxes(title_text=variable + ' ' + units_[variable])
    fig.update_xaxes(title_text='Time (15 min)')
    fig.update_layout(title='{}  - Time series'.format(road))

    # Add range slider
    fig.update_layout(
        xaxis=go.layout.XAxis(
            rangeselector=dict(
                buttons=list([
                    dict(count=1,
                         label="1 day",
                         step="day",
                         stepmode="backward"),
                    dict(count=3,
                         label="3 days",
                         step="day",
                         stepmode="backward"),
                    dict(count=7,
                         label="1 week",
                         step="day",
                         stepmode="backward")
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            type="date"
        )
    )
    # now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
    fig.write_html("figures\\{}_timeseries_by_section_{}.html".format(road, variable))
    # fig.show()



def plot_time_series_by_speed_bandwith(df, road, variable, start_date, end_date):
    """
        definition

    :param df:
    :param start_date:
    :param end_date:
    :param road:
    :return:
    """

    df_aux = df[(df.index > start_date) & (df.index < end_date)]

    # Plot all sections

    df_aux1 = df_aux[(df_aux['speed'] < 80) & (df_aux['speed'] >= 70)]
    df_aux2 = df_aux[(df_aux['speed'] < 70) & (df_aux['speed'] >= 50)]

    print(df_aux1[variable].describe())
    print(df_aux2[variable].describe())

    t1 = df_aux1[variable].index
    t2 = df_aux2[variable].index

    y1 = df_aux1[variable]
    y2 = df_aux2[variable]

    import plotly.graph_objects as go
    from dl_ia_cla_settings import units_

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=t1,
                             y=y1,
                             mode='markers',
                             marker=dict(color='blue'),
                             name='{} with speed in (80-70)'.format(variable)))

    fig.add_trace(go.Scatter(x=t2,
                             y=y2,
                             mode='markers',
                             marker=dict(color='gray'),
                             name='{} with speed in (70-50)'.format(variable)))

    fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
    fig.update_layout(showlegend=True)
    fig.update_yaxes(title_text=variable + ' ' + units_[variable])
    fig.update_xaxes(title_text='Time (15 min)')
    fig.update_layout(title='{}  - Time series '.format(road))

    # Add range slider
    fig.update_layout(
        xaxis=go.layout.XAxis(
            rangeselector=dict(
                buttons=list([
                    dict(count=1,
                         label="1 day",
                         step="day",
                         stepmode="backward"),
                    dict(count=3,
                         label="3 days",
                         step="day",
                         stepmode="backward"),
                    dict(count=7,
                         label="1 week",
                         step="day",
                         stepmode="backward")
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            type="date"
        )
    )
    # now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
    fig.write_html("figures\\{}_timeseries_by_speed_bandwith_{}.html".format(road, variable))
    fig.show()





def plot_time_series_congestion_by_day_by_lane(df, df_congestions):
    """
        definition

    :param df:
    :param start_date:
    :param end_date:
    :param road:
    :return:
    """



    import plotly.graph_objects as go
    from dl_ia_cla_settings import create_time_array

    t = create_time_array('00:00:00', '23:59:59', '15min')

    fig = go.Figure()

    for day in ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']:
        y_congestion = df_congestions['Congestion_{}'.format(day)]

        fig.add_trace(go.Scatter(x=t,
                                 y=y_congestion,
                                 mode='markers+lines',
                                 name='({}) Avg. Congestion (%)'.format(day)))

        # to plot volume and gap

        df_aux = df[df['weekday'] == dict_days[day]]

        subsection_ = df['sub_section'].unique()[0]

        df_aux = df_aux[df_aux['sub_section'] == subsection_]

        lane_ = df_aux['lanes'].unique()
        section_ = df_aux['section'].unique()[0]

        y_volume = df_aux.groupby(['hour', 'minute'])['volume'].mean()
        fig.add_trace(go.Scatter(x=t,
                                 y=y_volume.values,
                                 mode='markers+lines',
                                 name='({}) Avg. Volume (Veh./15min)'.format(day)))

        if 1 == 1:
            y_speed = df_aux.groupby(['hour', 'minute'])['speed'].mean()
            fig.add_trace(go.Scatter(x=t,
                                     y=y_speed.values,
                                     mode='markers+lines',
                                     name='({}) Avg. speed (km/h)'.format(day)))

        if 1 == 0:
            y_gap = df_aux.groupby(['hour', 'minute'])['gap'].mean()
            fig.add_trace(go.Scatter(x=t,
                                     y=y_gap.values * 0.1,
                                     mode='markers+lines',
                                     name='({}) Avg. Gap (1/10 sec)'.format(day)))

    fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
    fig.update_layout(showlegend=True)
    fig.update_yaxes(title_text=' ')
    fig.update_xaxes(title_text='Time (15 min)')
    fig.update_layout(title='{} - Section:{}  -Lane:{}  '.format(road, section_, int(lane_)))

    # now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
    fig.write_html("figures\\{}_congestion.html".format(road))
    fig.show()



def plot_time_series_congestion_by_day_by_section(df, df_congestions):
    """
        definition

    :param df:
    :param start_date:
    :param end_date:
    :param road:
    :return:
    """



    import plotly.graph_objects as go
    from dl_ia_cla_settings import create_time_array

    t = create_time_array('00:00:00', '23:59:59', '15min')

    fig = go.Figure()

    for day in ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']:
        y_congestion = df_congestions['Congestion_{}'.format(day)]

        fig.add_trace(go.Scatter(x=t,
                                 y=y_congestion,
                                 mode='markers+lines',
                                 name='({}) Avg. Congestion (%)'.format(day)))

        # to plot volume and gap

        df_aux = df[df['weekday'] == dict_days[day]]

        section_ = df['section'].unique()[0]

        df_aux = df_aux[df_aux['section'] == section_]

        y_volume = df_aux.groupby(['hour', 'minute'])['volume'].mean()
        fig.add_trace(go.Scatter(x=t,
                                 y=y_volume.values,
                                 mode='markers+lines',
                                 name='({}) Avg. Volume (Veh./15min)'.format(day)))

        if 1 == 1:
            y_speed = df_aux.groupby(['hour', 'minute'])['speed'].mean()
            fig.add_trace(go.Scatter(x=t,
                                     y=y_speed.values,
                                     mode='markers+lines',
                                     name='({}) Avg. speed (km/h)'.format(day)))

        if 1 == 0:
            y_gap = df_aux.groupby(['hour', 'minute'])['gap'].mean()
            fig.add_trace(go.Scatter(x=t,
                                     y=y_gap.values * 0.1,
                                     mode='markers+lines',
                                     name='({}) Avg. Gap (1/10 sec)'.format(day)))

    fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
    fig.update_layout(showlegend=True)
    fig.update_yaxes(title_text=' ')
    fig.update_xaxes(title_text='Time (15 min)')
    fig.update_layout(title='{} - Section:{} '.format(road, section_))

    # now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
    fig.write_html("figures\\{}_congestion_by_day_by_section_{}.html".format(road, section_))
    fig.show()



def plot_time_series_congestion_total_by_section(df, df_congestions):
    """
        definition

    :param df:
    :param start_date:
    :param end_date:
    :param road:
    :return:
    """



    import plotly.graph_objects as go
    import numpy as np
    from dl_ia_cla_settings import create_time_array

    t = create_time_array('00:00:00', '23:59:59', '15min')

    fig = go.Figure()

    Y_congestion = []
    for i in range(0, len(df_congestions)):
        Y_congestion.append(df_congestions.iloc[i, :].mean())

    fig.add_trace(go.Scatter(x=t,
                             y=Y_congestion,
                             mode='markers+lines',
                             name='Avg. Congestion Brisbane 2019 (%)'))

    # to plot volume and gap

    color_1 = {1: 'brown', 2: 'crimson', 3: 'darkred', 4: 'tomato'}
    color_2 = {1: 'sandybrown', 2: 'salmon', 3: 'orange', 4: 'goldenrod'}

    for section_ in np.sort(df['section'].unique()):
        df_aux = df[df['section'] == section_]

        y_volume = df_aux.groupby(['hour', 'minute'])['volume'].mean()

        fig.add_trace(go.Scatter(x=t,
                                 y=y_volume.values,
                                 mode='markers+lines',
                                 marker=dict(color=color_1[section_]),
                                 name='Section:{} Avg. Volume (Veh./15min)'.format(section_)))

        y_speed = df_aux.groupby(['hour', 'minute'])['speed'].mean()
        fig.add_trace(go.Scatter(x=t,
                                 y=y_speed.values,
                                 mode='markers+lines',
                                 marker=dict(color=color_2[section_]),
                                 name='Section:{} Avg. speed (km/h)'.format(section_)))

    fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
    fig.update_layout(showlegend=True)
    fig.update_yaxes(title_text=' ')
    fig.update_xaxes(title_text='Time (15 min)')
    fig.update_layout(title='{} -'.format(road))

    # now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
    fig.write_html("figures\\{}_congestion_total_by_section.html".format(road))
    fig.show()



def plot_time_series_congestion_total_by_lane(df, df_congestions):
    """
        definition

    :param df:
    :param start_date:
    :param end_date:
    :param road:
    :return:
    """



    import plotly.graph_objects as go
    import numpy as np
    from dl_ia_cla_settings import create_time_array

    t = create_time_array('00:00:00', '23:59:59', '15min')

    fig = go.Figure()

    Y_congestion = []
    for i in range(0, len(df_congestions)):
        Y_congestion.append(df_congestions.iloc[i, :].mean())

    fig.add_trace(go.Scatter(x=t,
                             y=Y_congestion,
                             mode='markers+lines',
                             name='Avg. Congestion Brisbane 2019 (%)'))

    # to plot volume and gap

    color_1 = {1: 'brown', 2: 'crimson', 3: 'darkred', 4: 'tomato'}
    color_2 = {1: 'sandybrown', 2: 'salmon', 3: 'orange', 4: 'goldenrod'}

    for subsection_ in np.sort(df['sub_section'].unique()):
        df_aux = df[df['sub_section'] == subsection_]

        lane_ = df_aux['lanes'].unique()
        section_ = df_aux['section'].unique()[0]

        y_volume = df_aux.groupby(['hour', 'minute'])['volume'].mean()

        fig.add_trace(go.Scatter(x=t,
                                 y=y_volume.values,
                                 mode='markers+lines',
                                 marker=dict(color=color_1[section_]),
                                 name='Section:{} Lane:{} Avg. Volume (Veh./15min)'.format(section_, int(lane_))))

        y_speed = df_aux.groupby(['hour', 'minute'])['speed'].mean()
        fig.add_trace(go.Scatter(x=t,
                                 y=y_speed.values,
                                 mode='markers+lines',
                                 marker=dict(color=color_2[section_]),
                                 name='Section:{} Lane:{} Avg. speed (km/h)'.format(section_, int(lane_))))

    fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
    fig.update_layout(showlegend=True)
    fig.update_yaxes(title_text=' ')
    fig.update_xaxes(title_text='Time (15 min)')
    fig.update_layout(title='{} -'.format(road))

    # now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
    fig.write_html("figures\\{}_congestion_total_by_lane.html".format(road))
    fig.show()




def plot_time_series_congestion_total(df, df_congestions):
    """
        definition

    :param df:
    :param start_date:
    :param end_date:
    :param road:
    :return:
    """



    import plotly.graph_objects as go
    from dl_ia_cla_settings import create_time_array

    t = create_time_array('00:00:00', '23:59:59', '15min')

    fig = go.Figure()

    Y_congestion = []
    for i in range(0, len(df_congestions)):
        Y_congestion.append(df_congestions.iloc[i, :].mean())

    fig.add_trace(go.Scatter(x=t,
                             y=Y_congestion,
                             mode='markers+lines',
                             name='Avg. Congestion Brisbane 2019 (%)'))

    y_volume = df.groupby(['hour', 'minute'])['volume'].mean()

    fig.add_trace(go.Scatter(x=t,
                             y=y_volume.values,
                             mode='markers+lines',
                             name='Avg. Volume (Veh./15min)'))

    y_speed = df.groupby(['hour', 'minute'])['speed'].mean()
    fig.add_trace(go.Scatter(x=t,
                             y=y_speed.values,
                             mode='markers+lines',
                             name='Avg. speed (km/h)'))

    fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
    fig.update_layout(showlegend=True)
    fig.update_yaxes(title_text=' ')
    fig.update_xaxes(title_text='Time (15 min)')
    fig.update_layout(title='{}  '.format(road))

    fig.write_html("figures\\{}_congestion_total.html".format(road))
    fig.show()




def plot_time_series_smooth(df, road, variable, start_date, end_date, smooth):
    """
        definition

    :param df:
    :param start_date:
    :param end_date:
    :param road:
    :return:
    """
    from scipy.signal import savgol_filter
    import numpy as np
    import plotly.graph_objects as go
    from dl_ia_cla_settings import units_

    df_aux = df[(df.index > start_date) & (df.index < end_date)]

    subsection_ = df_aux['sub_section'].unique()[0]
    df_aux = df_aux[df_aux['sub_section'] == subsection_]

    print(df_aux[variable].describe())

    t = df_aux[variable].index
    y = df_aux[variable]

    if smooth == 0:
        print('no smooth')

    if smooth == 1:
        smooth_label = 'Savgol filter'

        yhat = savgol_filter(y, 15, 2)  # window size 51, polynomial order 3

    if smooth == 2:
        smooth_label = 'Convolute'

        def smooth(y, box_pts):
            box = np.ones(box_pts) / box_pts
            y_smooth = np.convolve(y, box, mode='same')
            return y_smooth

        yhat = smooth(y, 3)

    if smooth == 3:
        smooth_label = 'Rolling'
        yhat = y.rolling(window=5, min_periods=5, center=True).median()

    if smooth == 4:
        smooth_label = 'Interpolate'
        from scipy.interpolate import interp1d

        x = np.arange(0, len(t))
        xx = np.linspace(x, x, 1000)

        # interpolate + smooth
        itp = interp1d(t, y.values, kind='linear')

        window_size, poly_order = 101, 3
        yhat = savgol_filter(itp(xx), window_size, poly_order)

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=t,
                             y=y,
                             mode='markers',
                             marker=dict(color='blue'),
                             name='{} data'.format(variable)))
    if smooth != 0:
        fig.add_trace(go.Scatter(x=t,
                                 y=yhat,
                                 mode='lines',
                                 marker=dict(color='gray'),
                                 name='{} smooth {}'.format(variable, smooth_label)))

    fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
    fig.update_layout(showlegend=True)
    fig.update_yaxes(title_text=variable + ' ' + units_[variable])
    fig.update_xaxes(title_text='Time (15 min)')
    fig.update_layout(title='{}  - Time series'.format(road))

    # Add range slider
    fig.update_layout(
        xaxis=go.layout.XAxis(
            rangeselector=dict(
                buttons=list([
                    dict(count=1,
                         label="1 day",
                         step="day",
                         stepmode="backward"),
                    dict(count=3,
                         label="3 days",
                         step="day",
                         stepmode="backward"),
                    dict(count=7,
                         label="1 week",
                         step="day",
                         stepmode="backward")
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            type="date"
        )
    )
    # now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
    fig.write_html("figures\\{}_timeseries_smooth_{}.html".format(road, variable))
    fig.show()




""" ===================== Reference Pattern ========================
"""



def plot_reference_daily_pattern(df, road, variable):
    """ This function do this-

    :param df:
    :param road:
    :param columns_:
    :return:
    """
    import plotly.graph_objects as go
    from dl_ia_cla_settings import units_
    from dl_ia_cla_settings import create_time_array

    t = create_time_array('00:00:00', '23:59:59', '15min')

    df_mean = df.groupby(by=['hour', 'minute'])[variable].mean()

    df_std = df.groupby(by=['hour', 'minute'])[variable].std()


    y_mean = df_mean.values

    y_upper_std = df_mean.values + df_std.values
    y_lower_std = df_mean.values - df_std.values
    y_lower_std[y_lower_std < 0] = 0

    y_upper_std2 = df_mean.values + 2 * df_std.values
    y_lower_std2 = df_mean.values - 2 * df_std.values
    y_lower_std2[y_lower_std2 < 0] = 0

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t,
                             y=y_mean,
                             mode='markers+lines',
                             name='Mean Value',
                             marker=dict(color='blue'),
                             opacity=0.6))

    fig.add_trace(go.Scatter(x=t,
                             y=y_upper_std,
                             name='Mean Value + 1 Standard deviation',
                             marker=dict(color='red'),
                             mode='lines',
                             opacity=0.6))

    fig.add_trace(go.Scatter(x=t,
                             y=y_lower_std,
                             name='Mean Value - 1 Standard deviation',
                             marker=dict(color='red'),
                             mode='lines',
                             opacity=0.6))

    fig.add_trace(go.Scatter(x=t,
                             y=y_upper_std2,
                             name='Mean Value + 2 Standard deviation',
                             marker=dict(color='darkred'),
                             mode='lines',
                             opacity=0.6))

    fig.add_trace(go.Scatter(x=t,
                             y=y_lower_std2,
                             name='Mean Value - 2 Standard deviation',
                             marker=dict(color='darkred'),
                             mode='lines',
                             opacity=0.6))

    index_ = df.index.dayofyear.unique()

    if 1 == 0:
        for days_ in index_:
            df_aux = df[df.index.dayofyear == days_]

            df_aux = df_aux[df_aux['section'] == df_aux['section'].unique()[0]]

            fig.add_trace(go.Scatter(x=t,
                                     y=df_aux[variable].values,

                                     name='Data',
                                     marker=dict(color='gray'),
                                     mode='markers',
                                     opacity=0.6))

    fig.update_yaxes(title_text=variable + ' ' + units_[variable])
    fig.update_xaxes(title_text='time (15 min)')

    fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
    fig.update_layout(title='{}  - Time series - (Reference Pattern)'.format(road))
    fig.update_layout(showlegend=True)
    # now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
    fig.write_html("figures\\{}_reference_pattern_{}.html".format(road, variable))
    fig.show()




def plot_reference_daily_pattern_by_weekend(df, road, variable):
    """ This function do this-

    :param df:
    :param road:
    :param columns_:
    :return:
    """


    import plotly.graph_objects as go
    from datetime import datetime
    from dl_ia_cla_settings import create_time_array

    t = create_time_array('00:00:00', '23:59:59', '15min')

    fig = go.Figure()

    weekend_dict = {1: 'weekend',
                    0: 'weekday'}

    for week_day_ in df['weekend'].unique():
        df_aux = df[df['weekend'] == week_day_]

        df_mean = df_aux.groupby(by=['hour', 'minute'])[variable].mean()
        df_std = df_aux.groupby(by=['hour', 'minute'])[variable].std()
        y_mean = df_mean.values
        y_upper_std1 = df_mean.values + df_std.values
        y_lower_std1 = df_mean.values - df_std.values
        y_lower_std1[y_lower_std1 < 0] = 0

        y_upper_std2 = df_mean.values + 2 * df_std.values
        y_lower_std2 = df_mean.values - 2 * df_std.values
        y_lower_std2[y_lower_std2 < 0] = 0

        fig.add_trace(go.Scatter(x=t,
                                 y=y_mean,
                                 mode='markers+lines',
                                 marker=dict(color='darkred'),
                                 name='Mean Value ({})'.format(weekend_dict[week_day_]),
                                 opacity=0.6))

        fig.add_trace(go.Scatter(x=t,
                                 y=y_upper_std1,
                                 marker=dict(color='blue'),
                                 name='Mean Value + 1 Standard deviation ({})'.format(weekend_dict[week_day_]),
                                 mode='lines',
                                 opacity=0.6))

        fig.add_trace(go.Scatter(x=t,
                                 y=y_lower_std1,
                                 marker=dict(color='blue'),
                                 name='Mean Value - 1 Standard deviation ({})'.format(weekend_dict[week_day_]),
                                 mode='lines',
                                 opacity=0.6))

        fig.add_trace(go.Scatter(x=t,
                                 y=y_upper_std2,
                                 marker=dict(color='darkblue'),
                                 name='Mean Value + 2 Standard deviation ({})'.format(weekend_dict[week_day_]),
                                 mode='lines',
                                 opacity=0.6))

        fig.add_trace(go.Scatter(x=t,
                                 y=y_lower_std2,
                                 marker=dict(color='darkblue'),
                                 name='Mean Value - 2 Standard deviation ({})'.format(weekend_dict[week_day_]),
                                 mode='lines',
                                 opacity=0.6))

    fig.update_yaxes(title_text=variable + ' ' + units_[variable])
    fig.update_xaxes(title_text='time (15 min)')

    fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
    fig.update_layout(title='{}  - Time series - (Reference Pattern)'.format(road))
    fig.update_layout(showlegend=True)
    now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
    # fig.write_html("figures\\{}_timeseries_{}_{}.html".format(road, variable, now, ))
    fig.show()





def plot_reference_daily_pattern_by_speed_bandwith(df, road, variable):
    """ This function do this-

    :param df:
    :param road:
    :param columns_:
    :return:
    """

    import plotly.graph_objects as go
    from dl_ia_cla_settings import units_
    from dl_ia_cla_settings import create_time_array

    t = create_time_array('00:00:00', '23:59:59', '15min')

    fig = go.Figure()

    df_aux = df

    # Plot all sections
    colors_ = [('red', 'darkred'), ('blue', 'darkblue')]

    speed_limits = [(80, 70), (70, 60)]


    for i in range(0, len(speed_limits)):

        # print(i)

        df_aux1 = df_aux[(df_aux['speed'] < speed_limits[i][0]) & (df_aux['speed'] >= speed_limits[i][1])]

        print(df_aux1.describe())

        df_mean = df_aux1.groupby(by=['hour', 'minute'])[variable].mean()
        df_std = df_aux1.groupby(by=['hour', 'minute'])[variable].std()
        y_mean = df_mean.values
        y_upper_1std = df_mean.values + df_std.values
        y_lower_1std = df_mean.values - df_std.values
        y_upper_2std = df_mean.values + 2 * df_std.values
        y_lower_2std = df_mean.values - 2 * df_std.values
        y_lower_1std[y_lower_1std < 0] = 0
        y_lower_2std[y_lower_2std < 0] = 0

        fig.add_trace(go.Scatter(x=t,
                                 y=y_mean,
                                 mode='markers+lines',
                                 marker=dict(color=colors_[0][i]),
                                 name='Mean Value (speed:{}-{})'.format(speed_limits[i][0], speed_limits[i][1]),
                                 opacity=0.6))

        if 0 == 1:
            fig.add_trace(go.Scatter(x=t,
                                     y=y_upper_1std,
                                     mode='lines',
                                     marker=dict(color='blue'),
                                     name='Mean Value + 1 Standard deviation (speed:{}-{})'.format(
                                         speed_limits[i][0], speed_limits[i][1]),
                                     opacity=0.6))

            fig.add_trace(go.Scatter(x=t,
                                     y=y_lower_1std,
                                     mode='markers',
                                     marker=dict(color='blue'),
                                     name='Mean Value - 1 Standard deviation (speed:{}-{})'.format(
                                         speed_limits[i][0], speed_limits[i][1]),
                                     opacity=0.6))

        fig.add_trace(go.Scatter(x=t,
                                 y=y_upper_2std,
                                 mode='markers',
                                 marker=dict(color=colors_[1][i]),
                                 name='Mean Value + 2 Standard deviation (speed:{}-{})'.format(speed_limits[i][0],
                                                                                               speed_limits[i][1]),
                                 opacity=0.6))

        fig.add_trace(go.Scatter(x=t,
                                 y=y_lower_2std,
                                 mode='markers',
                                 marker=dict(color=colors_[1][i]),
                                 name='Mean Value - 2 Standard deviation (speed:{}-{})'.format(speed_limits[i][0],
                                                                                               speed_limits[i][1]),
                                 opacity=0.6))

    fig.update_yaxes(title_text=variable + ' ' + units_[variable])
    fig.update_xaxes(title_text='time (15 min)')

    fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
    fig.update_layout(title='{}  - Time series - (Reference Pattern)'.format(road))
    fig.update_layout(showlegend=True)
    # now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
    fig.write_html("figures\\{}_reference_pattern_by_speed_bandwith_{}.html".format(road, variable))
    fig.show()



def plot_reference_daily_pattern_by_speed_bandwith_per_section(df, road, variable):
    """ This function do this-

    :param df:
    :param road:
    :param columns_:
    :return:
    """

    import plotly.graph_objects as go
    from dl_ia_cla_settings import units_, segments_dict_LW
    from dl_ia_cla_settings import create_time_array

    t = create_time_array('00:00:00', '23:59:59', '15min')

    fig = go.Figure()

    colors_ = [('red', 'darkred'), ('blue', 'darkblue')]
    speed_limits = [(80, 70), (70, 0)]

    df_aux0 = df


    for subsection_ in df_aux0['sub_section'].unique():

        df_aux = df_aux0[df_aux0['sub_section'] == subsection_]

        for i in range(0, len(speed_limits)):
            # print(i)

            df_aux1 = df_aux[(df_aux['speed'] < speed_limits[i][0]) & (df_aux['speed'] >= speed_limits[i][1])]

            print(df_aux1.describe())

            df_mean = df_aux1.groupby(by=['hour', 'minute'])[variable].mean()
            df_std = df_aux1.groupby(by=['hour', 'minute'])[variable].std()
            y_mean = df_mean.values
            # y_upper_1std = df_mean.values + df_std.values
            # y_lower_1std = df_mean.values - df_std.values
            y_upper_2std = df_mean.values + 2 * df_std.values
            y_lower_2std = df_mean.values - 2 * df_std.values
            # y_lower_1std[y_lower_1std < 0] = 0
            y_lower_2std[y_lower_2std < 0] = 0

            fig.add_trace(go.Scatter(x=t,
                                     y=y_mean,
                                     mode='markers+lines',
                                     marker=dict(color=colors_[0][i]),
                                     name='Reference Pattern (speed:{}-{}) section:{}'.format(
                                         speed_limits[i][0],
                                         speed_limits[i][1],
                                         segments_dict_LW[subsection_]),
                                     opacity=0.6))
            if 1 == 0:
                fig.add_trace(go.Scatter(x=t,
                                         y=y_upper_2std,
                                         mode='markers',
                                         marker=dict(color=colors_[1][i]),
                                         name='Upper limit (speed:{}-{}) section:{}'.format(
                                             speed_limits[i][0],
                                             speed_limits[i][1],
                                             segments_dict_LW[subsection_]),
                                         opacity=0.6))

                fig.add_trace(go.Scatter(x=t,
                                         y=y_lower_2std,
                                         mode='markers',
                                         marker=dict(color=colors_[1][i]),
                                         name='Lower Limit (speed:{}-{}) section:{}'.format(
                                             speed_limits[i][0],
                                             speed_limits[i][1],
                                             segments_dict_LW[subsection_]),
                                         opacity=0.6))

    fig.update_yaxes(title_text=variable + ' ' + units_[variable])
    fig.update_xaxes(title_text='time (15 min)')

    fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
    fig.update_layout(title='{}  - Reference Pattern'.format(road))
    fig.update_layout(showlegend=True)
    # now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
    fig.write_html("figures\\{}_reference_pattern_by_speed_bandwith_{}_per_section.html".format(road, variable))
    fig.show()



def plot_gaps_betwen_referece_patterns_by_speed_bandwidth(df, road, variable):
    """ This function do this-

    :param df:
    :param road:
    :param columns_:
    :return:
    """

    import plotly.graph_objects as go
    from dl_ia_cla_settings import units_, segments_dict_LW
    from dl_ia_cla_settings import create_time_array

    t = create_time_array('00:00:00', '23:59:59', '15min')

    fig = go.Figure()

    colors_ = [('red', 'darkred'), ('blue', 'darkblue')]
    speed_limits = [(80, 70), (70, 60)]

    df_aux0 = df



    subsection_ = 2

    df_aux = df_aux0[df_aux0['sub_section'] == subsection_]

    df_aux1 = df_aux[(df_aux['speed'] < speed_limits[0][0]) & (df_aux['speed'] >= speed_limits[0][1])]
    df_mean1 = df_aux1.groupby(by=['hour', 'minute'])[variable].mean()
    y_mean1 = df_mean1.values

    df_aux2 = df_aux[(df_aux['speed'] < speed_limits[1][0]) & (df_aux['speed'] >= speed_limits[1][1])]
    df_mean2 = df_aux2.groupby(by=['hour', 'minute'])[variable].mean()
    y_mean2 = df_mean2.values

    diff = y_mean2 - y_mean1

    fig.add_trace(go.Scatter(x=t,
                             y=diff,
                             mode='markers+lines',
                             marker=dict(color=colors_[0][0]),
                             name=' $ \Delta V_{i,1,t}^{SP0} \hspace{3cm} $',
                             opacity=0.6))

    subsection_ = 3

    df_aux = df_aux0[df_aux0['sub_section'] == subsection_]

    df_aux1 = df_aux[(df_aux['speed'] < speed_limits[0][0]) & (df_aux['speed'] >= speed_limits[0][1])]
    df_mean1 = df_aux1.groupby(by=['hour', 'minute'])[variable].mean()
    y_mean1 = df_mean1.values

    df_aux2 = df_aux[(df_aux['speed'] < speed_limits[1][0]) & (df_aux['speed'] >= speed_limits[1][1])]
    df_mean2 = df_aux2.groupby(by=['hour', 'minute'])[variable].mean()
    y_mean2 = df_mean2.values

    diff = y_mean2 - y_mean1

    fig.add_trace(go.Scatter(x=t,
                             y=diff,
                             mode='markers+lines',
                             marker=dict(color=colors_[0][1]),
                             name=' $ \Delta V_{i,2,t}^{SP0} \hspace{3cm} $',
                             opacity=0.6))

    fig.update_yaxes(title_text=variable + ' ' + units_[variable])
    fig.update_xaxes(title_text='time (15 min)')

    fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
    fig.update_layout(title='{}  - Reference Pattern - {}'.format(road, segments_dict_LW[subsection_]))
    fig.update_layout(showlegend=True)
    # now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
    fig.write_html("figures\\{}_reference_pattern_by_speed_bandwith_{}_per_section.html".format(road, variable))
    fig.show()




def plot_two_timeserie(y1, y2, var_1, var_2, road):
    """ This plots two time series from a dataframe

    :param df: dataframe with the two time series to plot and a column timestamp
    :param var_1: first variable to plot (str)
    :param var_2: second variarble to plot (str)
    :param road: name of the road from which the variables belongs
    :return:
    """

    import plotly.graph_objects as go

    print_ = True
    show = True

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=y1.index,
                             y=y1,
                             mode='markers+lines',
                             marker=dict(color='red'),
                             name='{}'.format(name1)))

    fig.add_trace(go.Scatter(x=y2.index,
                             y=y2,
                             mode='markers+lines',
                             marker=dict(color='blue'),
                             name='{}'.format(name2)))

    fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
    fig.update_layout(showlegend=True)
    fig.update_yaxes(title_text=' ')
    fig.update_xaxes(title_text='Time')
    fig.update_layout(title='Timeseries plot - {}'.format(road))
    if show:
        fig.show()
    if print_:
        fig.write_html("figures\\{}_congestion.html".format(road))





def plot_daily_patterns(y1, y2, name1, name2):
    """ This function do this-

    :param df:
    :param road:
    :param columns_:
    :return:
    """

    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y1.index,
                             y=y1,
                             mode='markers+lines',
                             marker=dict(color='red'),
                             name='{} Section:{} Lane:{} Day:{} '.format(name1)))
    fig.add_trace(go.Scatter(x=y2.index,
                             y=y2,
                             mode='markers+lines',
                             marker=dict(color='blue'),
                             name='{} Section:{} Lane:{} Day:{} '.format(name2)))

    fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
    fig.update_layout(showlegend=True)
    fig.update_yaxes(title_text=' ')
    fig.update_xaxes(title_text='Time (15 min)')
    fig.update_layout(title='{} -  '.format(road))
    # fig.write_html("figures\\{}_congestion.html".format(road))
    fig.show()


def plot_scatter_plotly(df, variable_x, variable_y):
    """ This function do this-

    :param df:
    :param road:
    :param columns_:
    :return:
    """
    import plotly.graph_objects as go
    from dl_ia_cla_settings import units_

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[variable_x],
                             y=df[variable_y],
                             mode='markers',
                             marker=dict(color='blue')))

    fig.update_xaxes(title_text=variable_x + ' ' + units_[variable_y])
    fig.update_yaxes(title_text=variable_y + ' ' + units_[variable])

    fig.update_layout(font=dict(family="Courier New, monospace", size=12, color="#7f7f7f"))
    fig.update_layout(title=road)
    fig.update_layout(showlegend=False)
    # fig.show()


def plot_scatter_mpl(df, variable):
    """ This function do this-

    :param df:
    :param road:
    :param columns_:
    :return:
    """

    import matplotlib.pyplot as plt

    print_ = True
    show = True

    fig = plt.figure()

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.scatter(df['occupancy'],
                df[variable])

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.scatter(df['speed'],
                df[variable])

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.scatter(df['headway'],
                df[variable])

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.scatter(df['meanlength'],
                df[variable])

    if print_:
        plt.show()
    if show:
        plt.savefig('figures\\plot_scatter_mpl.png')

def plot_scatter_together(df):
    """ This function do this-

    :param df:
    :param road:
    :param columns_:
    :return:
    """

    import plotly.graph_objects as go
    from dl_ia_cla_settings import units_
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=2, cols=2)

    fig.add_trace(go.Scatter(x=df['occupancy'],
                             y=df['flow'],
                             mode='markers',
                             name='Occupacy-flow',
                             marker=dict(color='blue')), row=1, col=1)

    fig.add_trace(go.Scatter(x=df['speed'],
                             y=df['flow'],
                             mode='markers',
                             name='speed-flow',
                             marker=dict(color='blue')), row=2, col=1)

    fig.add_trace(go.Scatter(x=df['headway'],
                             y=df['flow'],
                             mode='markers',
                             name='headway-flow',
                             marker=dict(color='blue')), row=1, col=2)

    fig.add_trace(go.Scatter(x=df['meanlength'],
                             y=df['flow'],
                             mode='markers',
                             name='meanlength-flow',
                             marker=dict(color='blue')), row=2, col=2)

    fig.update_xaxes(title_text="occupancy" + ' ' + units_['occupancy'], row=1, col=1)
    fig.update_yaxes(title_text="flow" + ' ' + units_['flow'], row=1, col=1)

    fig.update_xaxes(title_text="speed" + ' ' + units_['speed'], row=2, col=1)
    fig.update_yaxes(title_text="flow" + ' ' + units_['flow'], row=2, col=1)

    fig.update_xaxes(title_text="headway" + ' ' + units_['headway'], row=1, col=2)
    fig.update_yaxes(title_text="flow" + ' ' + units_['flow'], row=1, col=2)

    fig.update_xaxes(title_text="meanlength" + ' ' + units_['meanlength'], row=2, col=2)
    fig.update_yaxes(title_text="flow" + ' ' + units_['flow'], row=2, col=2)

    fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
    fig.update_layout(title=road)
    fig.update_layout(showlegend=False)
    # fig.show()



def plot_simulation(df_sim):
    """ This function do this-

    :param df:
    :param road:
    :param columns_:
    :return:
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=4, cols=1,
                        subplot_titles=("Volume Closure Section: {} Direction:{}".format(section_to_close, direction_),
                                        "Speed Closure Section: {} Direction:{}".format(section_to_close, direction_),
                                        "Volume Previous Section: {} Direction:{}".format(prev_section, direction_),
                                        "Speed Previous Section: {} Direction:{}".format(prev_section, direction_)))

    fig.add_trace(go.Scatter(x=df_sim['volume_prev_lane_{}'.format(lane_to_close)].index,
                             y=df_sim['volume_prev_lane_{}'.format(lane_to_close)],
                             mode='markers+lines',
                             marker=dict(color='red'),
                             line=dict(width=3),
                             legendgroup="lane1",
                             name='{}'.format('lane 1')), row=3, col=1)

    fig.add_trace(go.Scatter(x=df_sim['volume_prev_lane_{}'.format(lane_free)].index,
                             y=df_sim['volume_prev_lane_{}'.format(lane_free)],
                             mode='markers+lines',
                             line=dict(width=3),
                             legendgroup="lane2",
                             marker=dict(color='darkred'),
                             name='{}'.format('lane 2')), row=3, col=1)

    fig.add_trace(go.Scatter(x=df_sim['volume_prev_lane_{}_modify'.format(lane_to_close)].index,
                             y=df_sim['volume_prev_lane_{}_modify'.format(lane_to_close)],
                             mode='markers+lines',
                             line=dict(width=2, dash='dash'),
                             marker=dict(color='red'),
                             legendgroup="lane1_after",
                             name='{}'.format('lane 1 sim.')), row=3, col=1)

    fig.add_trace(go.Scatter(x=df_sim['volume_prev_lane_{}_modify'.format(lane_free)].index,
                             y=df_sim['volume_prev_lane_{}_modify'.format(lane_free)],
                             mode='markers+lines',
                             line=dict(width=2, dash='dash'),
                             legendgroup="lane2_after",
                             marker=dict(color='darkred'),
                             name='{}'.format('lane 2 sim. ')), row=3, col=1)

    fig.add_trace(go.Scatter(x=df_sim['volume_actual_lane_{}_modify'.format(lane_to_close)].index,
                             y=df_sim['volume_actual_lane_{}_modify'.format(lane_to_close)],
                             mode='markers+lines',
                             legendgroup="lane1_after",
                             line=dict(width=2, dash='dash'),
                             marker=dict(color='red'),
                             showlegend=False,
                             name='{}'.format('lane 1 - sim.')), row=1, col=1)

    fig.add_trace(go.Scatter(x=df_sim['volume_actual_lane_{}_modify'.format(lane_free)].index,
                             y=df_sim['volume_actual_lane_{}_modify'.format(lane_free)],
                             mode='markers+lines',
                             legendgroup="lane2_after",
                             line=dict(width=2, dash='dash'),
                             marker=dict(color='darkred'),
                             showlegend=False,
                             name='{}'.format('lane 2 - sim.')), row=1, col=1)

    fig.add_trace(go.Scatter(x=df_sim['volume_actual_lane_{}'.format(lane_to_close)].index,
                             y=df_sim['volume_actual_lane_{}'.format(lane_to_close)],
                             mode='markers+lines',
                             legendgroup="lane1",
                             line=dict(width=3),
                             marker=dict(color='red'),
                             showlegend=False,
                             name='{}'.format('lane 1')), row=1, col=1)

    fig.add_trace(go.Scatter(x=df_sim['volume_actual_lane_{}'.format(lane_free)].index,
                             y=df_sim['volume_actual_lane_{}'.format(lane_free)],
                             mode='markers+lines',
                             legendgroup="lane2",
                             line=dict(width=3),
                             marker=dict(color='darkred'),
                             showlegend=False,
                             name='{}'.format('lane 1')), row=1, col=1)

    fig.add_trace(go.Scatter(x=df_sim['speed_prev_lane_{}'.format(lane_to_close)].index,
                             y=df_sim['speed_prev_lane_{}'.format(lane_to_close)],
                             mode='markers+lines',
                             legendgroup="lane1",
                             line=dict(width=3),
                             marker=dict(color='red'),
                             showlegend=False,
                             name='{}'.format('Lane 1')), row=4, col=1)

    fig.add_trace(go.Scatter(x=df_sim['speed_prev_lane_{}'.format(lane_free)].index,
                             y=df_sim['speed_prev_lane_{}'.format(lane_free)],
                             mode='markers+lines',
                             legendgroup="lane2",
                             line=dict(width=3),
                             marker=dict(color='darkred'),
                             showlegend=False,
                             name='{}'.format('Lane 2')), row=4, col=1)

    fig.add_trace(go.Scatter(x=df_sim['speed_prev_lane_{}_modify'.format(lane_to_close)].index,
                             y=df_sim['speed_prev_lane_{}_modify'.format(lane_to_close)],
                             mode='markers+lines',
                             line=dict(width=2, dash='dash'),
                             legendgroup="lane1_after",
                             marker=dict(color='red'),
                             showlegend=False,
                             name='{}'.format('Lane 1')), row=4, col=1)

    fig.add_trace(go.Scatter(x=df_sim['speed_prev_lane_{}_modify'.format(lane_free)].index,
                             y=df_sim['speed_prev_lane_{}_modify'.format(lane_free)],
                             mode='markers+lines',
                             line=dict(width=2, dash='dash'),
                             legendgroup="lane2_after",
                             marker=dict(color='darkred'),
                             showlegend=False,
                             name='{}'.format('Lane 2')), row=4, col=1)

    fig.add_trace(go.Scatter(x=df_sim['speed_actual_lane_{}_modify'.format(lane_to_close)].index,
                             y=df_sim['speed_actual_lane_{}_modify'.format(lane_to_close)],
                             mode='markers+lines',
                             legendgroup="lane1_after",
                             line=dict(width=2, dash='dash'),
                             marker=dict(color='red'),
                             showlegend=False,
                             name='{}'.format('Lane 1')), row=2, col=1)

    fig.add_trace(go.Scatter(x=df_sim['speed_actual_lane_{}_modify'.format(lane_free)].index,
                             y=df_sim['speed_actual_lane_{}_modify'.format(lane_free)],
                             mode='markers+lines',
                             legendgroup="lane2_after",
                             marker=dict(color='darkred'),
                             line=dict(width=2, dash='dash'),
                             showlegend=False,
                             name='{}'.format('Lane 2')), row=2, col=1)

    fig.add_trace(go.Scatter(x=df_sim['speed_actual_lane_{}'.format(lane_to_close)].index,
                             y=df_sim['speed_actual_lane_{}'.format(lane_to_close)],
                             mode='markers+lines',
                             legendgroup="lane1",
                             line=dict(width=3),
                             marker=dict(color='red'),
                             showlegend=False,
                             name='{}'.format('Lane 1')), row=2, col=1)

    fig.add_trace(go.Scatter(x=df_sim['speed_actual_lane_{}'.format(lane_free)].index,
                             y=df_sim['speed_actual_lane_{}'.format(lane_free)],
                             mode='markers+lines',
                             legendgroup="lane2",
                             line=dict(width=3),
                             marker=dict(color='darkred'),
                             showlegend=False,
                             name='{}'.format('Lane 2')), row=2, col=1)

    fig.update_layout(font=dict(family="Courier New, monospace", size=11, color="#7f7f7f"))
    fig.update_layout(showlegend=True)
    fig.update_yaxes(title_text=' ')
    # fig.update_xaxes(title_text='Time (15 min)')
    fig.update_layout(title='{} - Road Traffic simulation - date:{} from {}am to {}am [{}] '.format(road, date_.date(), start_closure, end_closure, message))
    fig.write_html("figures\\{}\\road_traffic_simulation.html".format(road, section_to_close, lane_to_close))
    fig.show()


"""
    This script produce the following function plots:

    lineplots
    barplots
    histograms
    congestion matrix (heat map)
    correlation matrix (heat map)
    boxplots
    scatterplots
    timeseries
    reference patterns

    @ Jose Angel Velasco (javelacor@indra.es)
    (C) Indra Digital Labs 2020
"""





""" ===================== line plots ===================
"""


def plot_line(df, var_x, var_y, var_color, road):
    """ Plot sinmple time series volume with color per algorithm
    :param df:
    :param var_x:
    :param var_y:
    :param var_color:
    :return:
    """
    import plotly.express as px
    from dl_ia_cla_settings import config_plotly
    config_plotly()

    show = True
    print_ = True

    fig = px.line(df,
                  x=var_x,
                  y=var_y,
                  color=var_color)
    if print_:
        fig.write_html("figures\\plot_line_{}_{}_{}.html".format(road, var_x, var_y))
    if show:
        fig.show()





""" ===================== Bar plots ===================
"""



def plot_bar_df(df, variable):
    """ produce a bar plot with pandas function
    :param df:
    :param variable:
    :return:
    """
    df.sort_values(by=[variable], ascending=False, inplace=True)
    df.plot(kind='bar',
            ylim=(0, 100))


def plot_bar(df, variable, road, var_group):
    """ produce a barplot with plotly

    :param df:
    :param road:
    :return:
    """
    import plotly.express as px
    from dl_ia_cla_settings import config_plotly
    config_plotly()

    print_ = True
    show=  True
    error = 0

    try:
        fig = px.bar(df, x=var_group, y=variable)
        fig.update_yaxes(title_text='')
        fig.update_xaxes(title_text='Variables')

        fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
        fig.update_layout(title='Bar plot - {}'.format(road))
        fig.update_layout(showlegend=True)

        if print_:
            fig.write_html("figures\\{}\\plot_bar_{}.html".format(road, variable))
        if show:
            fig.show()

        return error

    except Exception as exception_msg:

        ECLA_comm('(!) Error in plot_bar: ' + str(exception_msg))
        error = 1
        return error



def plot_bar_frequency(df, road):
    """ Description
    :param df:
    :param variable:
    :return:
    """
    import plotly.graph_objects as go
    from dl_ia_cla_settings import create_time_array

    t = create_time_array('00:00:00', '23:59:59', '15min')

    df_count = df.groupby(by=['hour', 'minute'])['volume'].sum()



    volume_relative = 100 * df_count.values / df_count.values.sum()


    fig = go.Figure([go.Bar(x=t, y=volume_relative)])
    fig.update_yaxes(title_text=' Freq (%)')
    fig.update_xaxes(title_text='Time steps (15 min)')

    fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
    fig.update_layout(title='{}  - Freq. distribution - Volume'.format(road))
    fig.update_layout(showlegend=True)
    fig.write_html("figures\\{}_freq_barplot_volume.html".format(road))
    fig.show()




""" ===================== Histograms ===================
"""



def plot_histogram(df, variable, n_bins_, road):
    """ plot a histogram using plotly from a vairable in a dataframe

    :param df: Data frame with the variable
    :param variable: name of the variable (column name)
    :param n_bins_: number of bins of the histogram
    :param road: name of the road asset
    :return error: error code 0: everything ok, 1: something happened
    """

    import plotly.express as px
    from dl_ia_cla_settings import units_, config_plotly, CLA_comm, dl_ia_cla_settings_check_folder
    import numpy as np
    config_plotly()

    print_ = True
    show = True
    fontsize_ = 18
    error = 0

    try:
        max_value = int(df[variable].max())
        x_axis = np.arange(0, max_value, int(max_value / 20))

        fig = px.histogram(df, x = variable, nbins = n_bins_, marginal="box", hover_data=df.columns)
        fig.update_xaxes(title_text=variable + ' ' + units_[variable])
        fig.update_layout(font=dict(family="Courier New, monospace", size=fontsize_, color="#7f7f7f"))
        fig.update_layout(title='Histogram - {} - {}'.format(road, variable))
        fig.update_layout(showlegend=True)
        fig.update_traces(opacity=0.9)
        fig.update_layout(bargap=0.2)  # gap between bars of adjacent location coordinates
        fig.update_xaxes(ticktext=x_axis, tickvals=x_axis)
        # now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
        if print_:
            dl_ia_cla_settings_check_folder('figures\\{}'.format(road))
            fig.write_html("figures\\{}\\plot_histogram_{}.html".format(road, variable))

        if show:
            fig.show()
        return error
    except Exception as exception_msg:
        CLA_comm('(!) Error in plot_histogram: ' + str(exception_msg))
        error = 1
        return error


def plot_distribution(df, variable, road):
    """ plot the distribution using plotly from a vairable in a dataframe

    :param df: Data frame with the variable
    :param variable: name of the variable (column name)
    :param road: name of the road asset
    :return error: error code 0: everything ok, 1: something happened
    """

    import plotly.figure_factory as ff
    from dl_ia_cla_settings import units_, config_plotly, CLA_comm
    import numpy as np
    config_plotly()

    print_ = True
    show = True
    fontsize_ = 18
    error = 0

    try:
        max_value = int(df[variable].max())
        x_axis = np.arange(0, max_value, int(max_value / 20))

        fig = ff.create_distplot([list(df[variable].values)], variable)
        fig.update_xaxes(title_text=variable + ' ' + units_[variable])
        fig.update_layout(font=dict(family="Courier New, monospace", size=fontsize_, color="#7f7f7f"))
        fig.update_layout(title='Distribution - {} - {}'.format(road, variable))
        fig.update_layout(showlegend=True)
        fig.update_traces(opacity=0.9)
        fig.update_layout(bargap=0.2)  # gap between bars of adjacent location coordinates
        fig.update_xaxes(ticktext=x_axis, tickvals=x_axis)

        if print_:
            fig.write_html("figures\\{}\\plot_distribution_{}.html".format(road, variable))
        if show:
            fig.show()
        return error
    except Exception as exception_msg:
        CLA_comm('(!) Error in plot_distribution: ' + str(exception_msg))
        error = 1
        return error

def plot_histogram_mpl(df, variable, n_bins, road):
    """

    :param df: Data frame with the vairable to plot
    :param variable: name of the variable (column name)
    :param n_bins_: number of bins of the histogram
    :return:
    """

    import matplotlib.pyplot as plt
    from dl_ia_cla_settings import units_
    from datetime import datetime
    import numpy as np

    print_ = True
    show = True
    fontsize_ = 18

    fig = plt.figure(figsize=(10, 8), dpi=150, facecolor='w', edgecolor='w')
    ax = fig.add_subplot(1, 1, 1)

    # fig, ax = plt.subplots(1, 1, sharey=True, tight_layout=True)
    n = ax.hist(df[variable], bins=n_bins, edgecolor='b')
    ax.set_xlabel(variable + ' ' + units_[variable], fontsize=fontsize_)
    ax.set_ylabel('Count', fontsize=fontsize_)
    xmax = df[variable].max()
    x_range = list(np.arange(0, xmax, int(xmax / 20)))
    # ax.set_xticks = x_range
    # ax.set_xticklabels = x_range
    ax.tick_params(axis='both', labelsize=fontsize_)
    now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
    plt.title('{} - Histogram - {} '.format(road, variable), fontsize=fontsize_)
    if show:
        plt.show()

    if print_:
        plt.savefig('figures\\{}_histogram_{}.png'.format(road, variable),
                    bbox_inches='tight', pad_inches=0)




"""  =================== Congestion matrix ====================
"""



def plot_congestion_matrix(df_congestions):
    """

    :param df: Data frame with the vairable to plot
    :param variable: name of the variable (column name)
    :param n_bins_: number of bins of the histogram
    :return:
    """
    congestion = df_congestions.values
    r = round(congestion, 2)

    import plotly.figure_factory as ff

    print_ = True
    show = True

    fig = ff.create_annotated_heatmap(congestion,
                                      colorscale='Viridis')
    fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
    fig.update_layout(title='Congestion Matrix - Brisbane')
    fig.update_layout(showlegend=True)

    if print_:
        fig.write_html("figures\\plot_congestion_matrix_{}.html".format(road))
    if show:
        fig.show()




"""  =================== Correlation matrix ====================
"""



def plot_correlation_matrix(df, road):
    """ This function calcualte the correlation coefficient between the columns of a dataframe and plot
        in a heat map using plotly

    :param df: dataframe with the columns being the variables
    :param road: road of the variables
    :return R: dataframe with the correlations
    """

    import plotly.figure_factory as ff
    from dl_ia_cla_settings import CLA_comm, dl_ia_cla_settings_check_folder
    from dl_ia_cla_settings import config_plotly
    config_plotly()

    error = 0
    print_ = True
    show = True

    x_ = df.columns
    y_ = df.columns

    R = df.corr()
    r = round(R, 2)

    try:
        fig = ff.create_annotated_heatmap(r.values,
                                          x = list(r.columns),
                                          y = list(r.columns),
                                          colorscale='Viridis')
        fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
        fig.update_layout(title='{} - Correlation Matrix (Pearson correlation coefficient)'.format(road))
        fig.update_layout(showlegend=True)
        # now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
        if show:
            fig.show()
        if print_:
            dl_ia_cla_settings_check_folder('figures\\{}'.format(road))
            fig.write_html("figures\\{}\\correlation_matrix.html".format(road))

        return error, R

    except Exception as exception_msg:
        CLA_comm('(!) Error in plot_correlation_matrix: ' + str(exception_msg))
        error = 1
        R = 0
        return error, R








"""  =========================== Boxplots ===========================
"""



def boxplot_plotly(df, var, var_group, var_color, road):
    """ This function do this-

    :param df:
    :param road:
    :param columns_:
    :return:
    """
    import plotly.express as px
    from dl_ia_cla_settings import units_

    print_ = True
    show = True

    fig = px.box(df, y=var, x=var_group, color=var_color)
    fig.update_xaxes(title_text=var_group)
    fig.update_yaxes(title_text='{}'.format(var + ' ' + units_[variable]))
    fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
    fig.update_layout(title='Boxplot - {} - {}'.format(road, variable))
    fig.update_layout(showlegend=True)

    # now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")

    if print_:
        fig.write_html(
        "figures\\{}\\boxplot_{}_{}_{}.html".format(road, var, var_group, var_color))
    if show:
        fig.show()




""" ================= Scatter plots ============================== 
"""





def plot_scatterplot(df, var_x, var_y, var_color, var_size, road):
    """ Produce a simple scatter plot with plotly

    :param df: dataframe that contains the variables
    :param variable_x: variable to plot in x axis
    :param variable_y: variable to plot in y axis
    :param variable_to_color: variable to use as color
    :param variable_to_color: variable to use as size
    :return:
    """
    import plotly.express as px
    from dl_ia_cla_settings import units_, CLA_comm
    from dl_ia_cla_settings import config_plotly, dl_ia_cla_settings_check_folder
    config_plotly()

    print_ = True
    show = True
    error = 0

    try:
        fig = px.scatter(df,
                         x = var_x,
                         y = var_y,  # marker = dict(color='blue')
                         size = var_size,
                         color = var_color)

        fig.update_xaxes(title_text=var_x + ' ' + units_[var_x])
        fig.update_yaxes(title_text=var_y + ' ' + units_[var_y])
        fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
        fig.update_layout(title='Scatterplot - {}'.format(road))
        fig.update_layout(showlegend=True)
        # now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")

        if print_:
            dl_ia_cla_settings_check_folder('figures\\{}'.format(road))
            fig.write_html("figures\\{}\\scatterplot_{}_{}.html".format(road, var_x, var_y))
        if show:
            fig.show()

        return error


    except Exception as exception_msg:

        CLA_comm('(!) Error in plot_scatterplot: ' + str(exception_msg))
        error = 1
        return error





def plot_scatterplot_mpl(df, variable_x, variable_y, group_variable, road):
    """
    Scatterplot matplotlib individual with color by variable

    :param df:
    :param variable_x:
    :param variable_y:
    :param group_variable:
    :param road:
    :return:
    """

    import matplotlib.pyplot as plt
    from dl_ia_cla_settings import units_
    from datetime import datetime

    fig = plt.figure(figsize=(12, 8), dpi=150, facecolor='w', edgecolor='w')
    ax = fig.add_subplot(1, 1, 1)

    for var in df[group_variable].unique():
        df_aux = df[df[group_variable] == var]
        x = df_aux[variable_x].values
        y = df_aux[variable_y].values
        ax.scatter(x, y, label=var, alpha=0.8, edgecolor='k')

    fontsize_ = 12
    ax.set_xlabel(variable_x + ' ' + units_[variable_x], fontsize=fontsize_)
    ax.set_ylabel(variable_y + ' ' + units_[variable_y], fontsize=fontsize_)
    ax.tick_params(axis='both', labelsize=fontsize_)
    plt.legend()

    now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
    plt.title('{}'.format(road), fontsize=fontsize_)
    plt.show()

    plt.savefig('figures\\{}_scatterplot_{}_{}.png'.format(road, variable_y, variable_x),
                bbox_inches='tight',
                pad_inches=0)
    plt.close()




def plot_scatterplot_mpl2(df):
    """ This function do this-

    :param df:
    :param road:
    :param columns_:
    :return:
    """
    fontsize_ = 12

    import matplotlib.pyplot as plt
    from dl_ia_cla_settings import units_

    print_ = True
    show = True

    fig = plt.figure(figsize=(12, 6), dpi=90, facecolor='w', edgecolor='k')

    ax1 = fig.add_subplot(1, 1, 1)
    var_x = 'flow'
    var_y = 'speed'
    ax1.scatter(df[var_x], df[var_y], marker='o', s=3, c='b')
    ax1.set_ylabel(units_[var_y], fontsize=fontsize_)
    ax1.set_xlabel(units_[var_x], fontsize=fontsize_)

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.scatter(df['occupancy'], df['speed'])
    ax2.set_ylabel(units_['speed'], fontsize=fontsize_)
    ax2.set_xlabel(units_['occupancy'], fontsize=fontsize_)

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.scatter(df['flow'], df['occupancy'])
    ax3.set_ylabel(units_['speed'], fontsize=fontsize_)
    ax3.set_xlabel(units_['occupancy'], fontsize=fontsize_)

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.scatter(df['occupancy'], df['volume'])
    ax4.set_ylabel(units_['speed'], fontsize=fontsize_)
    ax4.set_xlabel(units_['occupancy'], fontsize=fontsize_)

    if show:
        plt.show()
    if print_:
        plt.savefig('figures\\plot_scatterplot_matplotlib.pdf', bbox_inches='tight', pad_inches=0.1)
    # plt.close()


def plot_scatterplot_mpl_multi(df, variable_x, variable_y, group_var, road):
    """
        Scatterplot matplotlib (4 plots)

    :param df:
    :param variable_x:
    :param variable_y:
    :param group_var:
    :param road:
    :return:
    """
    fontsize_ = 12
    marker_size_ = 10

    import matplotlib.pyplot as plt
    from dl_ia_cla_settings import units_
    from datetime import datetime
    import numpy as np

    fig = plt.figure(figsize=(12, 8), dpi=150, facecolor='w', edgecolor='w')

    for i in np.arange(0, len(variable_x)):
        ax1 = fig.add_subplot(2, 2, i + 1)
        sc1 = ax1.scatter(df[variable_x[i]],
                          df[variable_y[i]], marker='o', s=marker_size_, alpha=0.8, c=df[group_var])
        cbar1 = plt.colorbar(sc1)
        cbar1.set_label(group_var, rotation=270, fontsize=fontsize_)
        ax1.set_xlabel(variable_x[i] + ' ' + units_[variable_x[i]], fontsize=fontsize_)
        ax1.set_ylabel(variable_y[i] + ' ' + units_[variable_y[i]], fontsize=fontsize_)

        y_min = df[variable_y[i]].min()
        y_max = df[variable_y[i]].max()
        x_min = df[variable_x[i]].min()
        x_max = df[variable_x[i]].max()
        y_range = np.arange(y_min, y_max, round(y_max / 5))
        x_range = np.arange(x_min, x_max, round(x_max / 5))
        ax1.set_yticks(y_range)
        ax1.set_yticklabels(y_range)
        ax1.set_xticks(x_range)
        ax1.set_xticklabels(x_range)
        ax1.tick_params(axis='both', labelsize=fontsize_)

    now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
    plt.title('{}'.format(road), fontsize=fontsize_)
    plt.subplots_adjust(wspace=0.2, hspace=0.4)
    plt.show()
    plt.savefig('figures\\{}_scatterplot_multi_{}.png'.format(road, now),
                bbox_inches='tight', pad_inches=0)

    plt.close()




def plot_scatterplot_by_color(df, variable_x, variable_y, color_variable, road):
    """
    Statter plot plotly individual all variables segmented by schedule (!) ALL COMBINATIONS (6x7=42 plots)

    :param df:
    :param variable_x:
    :param variable_y:
    :param color_variable:
    :param road:
    :return:
    """
    import plotly.express as px
    from dl_ia_cla_settings import units_

    fig = px.scatter(df,
                     x=variable_x,
                     y=variable_y,
                     color=color_variable,
                     opacity=0.8,
                     hover_data=[color_variable])

    fig.update_xaxes(title_text=variable_x + ' ' + units_[variable_x])
    fig.update_yaxes(title_text=variable_y + ' ' + units_[variable_y])
    fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
    fig.update_layout(title='{} - Scatterplots'.format(road))
    fig.update_layout(showlegend=True)
    # now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
    # fig.show()
    fig.write_html(
        "figures\\{}_scatterplot_by_color_{}_{}.html".format(road, variable_x, variable_y))




def plot_scatterplot_by_color_multi(df, variable_x, variable_y, road):
    """
    Plotly scatterplot multiple with all values and segmented by schedule time



    :param df:
    :param variable_x:
    :param variable_y:
    :param road:
    :return:
    """

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import numpy as np

    fig = make_subplots(rows=2, cols=3)

    index = {'0-6': (1, 1),
             '7-12': (1, 2),
             '13-17': (1, 3),
             '18-21': (2, 1),
             '22-23': (2, 2),
             }

    colors_ = {'0-6': 'darkblue',
               '7-12': 'darkgreen',
               '13-17': 'darkred',
               '18-21': 'darkviolet',
               '22-23': 'darkgrey',
               }

    for schedule_ in df['schedule_def'].unique():
        df_aux = df.loc[df['schedule_def'] == schedule_]

        fig.add_trace(go.Scatter(x=df_aux[variable_x].values,
                                 y=df_aux[variable_y].values,
                                 name=schedule_,
                                 mode='markers',
                                 marker_color=colors_[schedule_]), row=index[schedule_][0], col=index[schedule_][1])

        x_axis = np.arange(0, int(df[variable_x].max()), int(df[variable_x].max() / 10))
        y_axis = np.arange(0, int(df[variable_y].max()), int(df[variable_y].max() / 10))
        fig.update_yaxes(title_text=variable_y,
                         tickfont=dict(size=14),
                         ticktext=y_axis,
                         tickvals=y_axis, row=index[schedule_][0], col=index[schedule_][1])

        fig.update_xaxes(title_text=variable_x,
                         tickfont=dict(size=14),
                         ticktext=x_axis,
                         tickvals=x_axis, row=index[schedule_][0], col=index[schedule_][1])

    fig.update_layout(showlegend=True)
    fig.update_layout(font=dict(family="Courier New, monospace", size=14, color="#7f7f7f"))
    fig.update_layout(title_text='{} - Scatterplots '.format(road))
    # now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
    # fig.show()
    fig.write_html("figures\\{}_scatterplot_multi_{}_{}_by_schedule_{}.html".format(road, variable_x, variable_y))




def plot_scatterplot_by_schedule_multi(df, variable_x, variable_y, road):
    """
    Plotly scatterplot individual with values segmented by schedule time

    :param df:
    :param variable_x:
    :param variable_y:
    :param road:
    :return:
    """

    from dl_ia_cla_settings import colors_, units_
    import plotly.graph_objects as go
    from datetime import datetime
    import numpy as np

    for schedule_ in df['schedule_def'].unique():
        df_aux = df.loc[df['schedule_def'] == schedule_]

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=df[variable_x].values,
                                 y=df[variable_y].values,
                                 name='All',
                                 mode='markers',
                                 opacity=0.9,
                                 marker_color='lightgrey'))

        fig.add_trace(go.Scatter(x=df_aux[variable_x].values,
                                 y=df_aux[variable_y].values,
                                 name=schedule_,
                                 mode='markers',
                                 marker_color=colors_[schedule_]))

        x_axis = np.arange(0, int(df[variable_x].max()), int(df[variable_x].max() / 10))
        y_axis = np.arange(0, int(df[variable_y].max()), int(df[variable_y].max() / 10))
        fig.update_yaxes(title_text=variable_y + units_[variable_y],
                         tickfont=dict(size=14),
                         ticktext=y_axis,
                         tickvals=y_axis)

        fig.update_xaxes(title_text=variable_x + units_[variable_x],
                         tickfont=dict(size=14),
                         ticktext=x_axis,
                         tickvals=x_axis)

        fig.update_layout(showlegend=True)
        fig.update_layout(font=dict(family="Courier New, monospace", size=14, color="#7f7f7f"))
        fig.update_layout(title_text='{} - Scatterplot '.format(road))
        now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
        fig.write_html(
            "figures\\{}_scatterplot_multi_{}_{}_schedule_{}_{}.html".format(road, variable_x, variable_y,
                                                                             schedule_, now))
        # fig.show()



def plot_scatterplot_with_histogram(df, variable_y, variable_x, grouping_variable, road):
    """

    :param variable_y:
    :param variable_x:
    :param grouping_variable:
    :param road:
    :return:
    """
    ""

    import matplotlib.pyplot as plt
    from dl_ia_cla_settings import units_
    import numpy as np

    # definitions for the axes
    left, width = 0.1, 0.4
    bottom, height = 0.1, 0.4
    spacing = 0.005

    fontsize_ = 16

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]

    # start with a square Figure
    fig = plt.figure(figsize=(12, 8), dpi=150, facecolor='w', edgecolor='w')

    ax = fig.add_axes(rect_scatter)
    ax_histx = fig.add_axes(rect_histx, sharex=ax)
    ax_histy = fig.add_axes(rect_histy, sharey=ax)

    x = df[variable_x].values
    y = df[variable_y].values

    color_ = df[grouping_variable].values

    # now determine nice limits by hand:
    binwidth = 0.5
    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    lim = (int(xymax / binwidth) + 1) * binwidth
    bins = np.arange(0, lim + binwidth, binwidth)

    ax_histx.hist(x, bins=bins)
    ax_histy.hist(y, bins=bins, orientation='horizontal')
    ax_histx.axis('off')
    ax_histy.axis('off')

    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    sc = ax.scatter(x, y, c=color_)
    cbar = plt.colorbar(sc)
    cbar.set_label(grouping_variable, rotation=270, fontsize=fontsize_)

    xmax = int(df[variable_x].max())
    ymax = int(df[variable_y].max())

    ax.set_xlabel(variable_x + ' ' + units_[variable_x], fontsize=fontsize_)
    ax.set_ylabel(variable_y + ' ' + units_[variable_y], fontsize=fontsize_)

    ax.tick_params(axis='both', labelsize=fontsize_)
    ax.set_xticks = list(np.arange(0, xmax, int(xmax / 10)))
    ax.set_yticks = list(np.arange(0, ymax, int(ymax / 10)))
    ax.set_xticklabels = list(np.arange(0, xmax, int(xmax / 10)))
    ax.set_yticklabels = list(np.arange(0, ymax, int(ymax / 10)))

    # now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")

    plt.title('{}'.format(road), fontsize=fontsize_)
    plt.show()
    plt.savefig('figures\\{}_scatterplot_with_histogram_{}_{}.png'.format(road, variable_y, variable_x),
                bbox_inches='tight', pad_inches=0)
    plt.close()





"""  ==================== Time series =======================
"""



def plot_time_series_mpl(df, start_date, end_date, road, variable):
    """
    Plot time series with matplotlib, one variable per plot,
    segmented by section, for a specific data range
    The plot is saved in png format

    :param df:
    :param start_date:
    :param end_date:
    :param road:
    :return:
    """
    fontsize_ = 18

    import matplotlib.pyplot as plt
    from dl_ia_cla_settings import units_

    fig = plt.figure(figsize=(25, 10), dpi=150, facecolor='w', edgecolor='w')
    ax = fig.add_subplot(1, 1, 1)

    for section_ in df['sub_section'].unique():
        df_aux = df[df['sub_section'] == section_]
        y = df_aux.loc[start_date:end_date, [variable]]
        ax.plot(y, '-o', label='section:{}'.format(section_))

    ax.set_xlabel('time (15 min)', fontsize=fontsize_)
    ax.set_ylabel(variable + units_[variable], fontsize=fontsize_)
    ax.tick_params(axis='both', labelsize=fontsize_)
    ax.legend(fontsize=fontsize_)
    ax.set_title('{} - Time Series - {} to {}'.format(road, start_date, end_date), fontsize=fontsize_)
    # now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
    plt.subplots_adjust(wspace=0.2, hspace=0.4)
    plt.show()
    plt.savefig('figures\\{}_timeseries_{}.png'.format(road, variable), bbox_inches='tight', pad_inches=0)
    plt.close()



def plot_timeseries_with_slider(df, var_x, var_y, road):
    """ Plot simple time series with time slider

    :param df: dataframe with timestamp index and variable
    :param variable: name of the variable
    :param road: name of the road/asset
    :return:
    """

    import plotly.graph_objects as go
    from dl_ia_cla_settings import units_, CLA_comm, dl_ia_cla_settings_check_folder

    print_ = True
    show = True
    error = 0

    try:
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=df[var_x],
                                 y=df[var_y],
                                 mode='markers+lines',
                                 marker=dict(color='red'),
                                 name='{}'.format(var_y)))

        # Add range slider
        fig.update_layout(
            xaxis=go.layout.XAxis(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1,
                             label="1 day",
                             step="day",
                             stepmode="backward"),
                        dict(count=3,
                             label="3 days",
                             step="day",
                             stepmode="backward"),
                        dict(count=7,
                             label="1 week",
                             step="day",
                             stepmode="backward"),
                    ])
                ),
                rangeslider=dict(
                    visible=True
                ),
                type="date"
            )
        )

        fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
        fig.update_layout(showlegend=True)
        fig.update_yaxes(title_text=var_y + ' ' + units_[var_y])
        fig.update_xaxes(title_text='Time')
        fig.update_layout(title='Timeseries')

        if print_:
            dl_ia_cla_settings_check_folder('figures\\{}'.format(road))
            fig.write_html("figures\\{}\\timeseries_with_slider_{}.html".format(road, var_y))
        if show:
            fig.show()

    except Exception as exception_msg:
        CLA_comm('(!) Error in plot_timeseries_with_slider: ' + str(exception_msg))
        error = 1
        return error




def plot_timeseries(df, var_x, var_y, road):
    """ Plot simple time series with time slider

    :param df: dataframe with timestamp index and variable
    :param variable: name of the variable
    :param road: name of the road/asset
    :return:
    """

    import plotly.graph_objects as go
    from dl_ia_cla_settings import units_, CLA_comm, dl_ia_cla_settings_check_folder

    print_ = True
    show = True
    error = 0

    try:
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=df[var_x],
                                 y=df[var_y],
                                 mode='markers+lines',
                                 marker=dict(color='red'),
                                 name='{}'.format(var_y)))


        fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
        fig.update_layout(showlegend=True)
        fig.update_yaxes(title_text=variable + ' ' + units_[var_y])
        fig.update_xaxes(title_text='Time')
        fig.update_layout(title='Timeseries - {} - {}'.format(road, var_y))

        if print_:
            dl_ia_cla_settings_check_folder('figures')
            fig.write_html("figures\\{}\\timeseries_with_slider_{}.html".format(road, var_y))
        if show:
            fig.show()

    except Exception as exception_msg:

        CLA_comm('(!) Error in plot_timeseries: ' + str(exception_msg))
        error = 1
        return error


def plot_time_Series(df, variable, road):
    """ Produce the time series plot

    :param df: data frame with the variable to plot, the time index, and the columns of section and lanes
    :param variable: name of the road traffic variable to plot
    """

    import plotly.graph_objects as go
    from dl_ia_cla_settings import units_
    import numpy as np

    fig = go.Figure()

    for section_ in np.sort(df['section'].unique()):
        df_aux = df[df['section'] == section_]
        for lane_ in np.sort(df_aux['lanes'].unique()):
            df_aux2 = df_aux[df_aux['lanes'] == lane_]
            df_aux2.sort_values(by=['time'], inplace=True)

            fig.add_trace(go.Scatter(x=df_aux2['time'],
                                     y=df_aux2[variable],
                                     line=dict(width=2, dash='dot'),
                                     mode='markers+lines',
                                     name=variable + ' ' + units_[variable] + '{}'.format(section_) + '{}'.format(lane_)))

    fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
    fig.update_layout(
        title='{} - timeseries '.format(road),
        xaxis_title='Time',
        yaxis_title='',
        showlegend=True
    )

    # Add range slider
    fig.update_layout(
        xaxis=go.layout.XAxis(
            rangeselector=dict(
                buttons=list([
                    dict(count=1,
                         label="1 month",
                         step="month",
                         stepmode="backward"),
                    dict(count=6,
                         label="6 month",
                         step="month",
                         stepmode="backward"),
                    dict(count=7,
                         label="1 week",
                         step="day",
                         stepmode="backward"),
                    dict(count=1,
                         label="1 year",
                         step="year",
                         stepmode="backward"),
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            type="date"
        )
    )
    fig.show()
    fig.write_html("figures\\{}\\plot_time_Series_{}.html".format(road, road))


def time_series_comparison(time_index, y1, y2, label1, label2):
    """ Plot two time series with the same time index

    :param time_index:
    :param y1:
    :param y2:
    :param label1:
    :param label2:
    :return:
    """

    import plotly.graph_objects as go
    from dl_ia_cla_settings import units_, config_plotly
    config_plotly()

    print_ = True
    show = True
    title = 'time_series_comparison'

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_index,
                             y=y1,
                             mode='markers',
                             marker=dict(color='red'),
                             name='{}'.format(label1)))

    fig.add_trace(go.Scatter(x=time_index,
                             y=y2,
                             mode='markers+lines',
                             marker=dict(color='blue'),
                             name='{}'.format(label2)))

    fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
    fig.update_layout(showlegend=True)
    fig.update_yaxes(title_text=label1 + ' '+ units_[label1])
    fig.update_xaxes(title_text='Time')
    fig.update_layout(title='Time series comparison')

    # Add range slider
    fig.update_layout(
        xaxis=go.layout.XAxis(
            rangeselector=dict(
                buttons=list([
                    dict(count=1,
                         label="1 day",
                         step="day",
                         stepmode="backward"),
                    dict(count=3,
                         label="3 day",
                         step="day",
                         stepmode="backward"),
                    dict(count=7,
                         label="1 week",
                         step="day",
                         stepmode="backward"),
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            type="date"
        )
    )
    if print_:
        fig.write_html("figures\\{}_{}_{}.html".format(title, label1, label2))
    if show:
        fig.show()





def plot_time_Series_slider(df, variable, road):
    """ plot a time series with a time slider

    :param df: dataframe with the data
    :return:
    """
    from dl_ia_cla_settings import units_
    import plotly.graph_objects as go


    subsection_ = df['sub_section'].unique()[0]
    df_aux = df[df['sub_section'] == subsection_]

    lane_ = df_aux[df_aux['sub_section'] == subsection_]['lanes'].unique()[0]
    section_ = df_aux[df_aux['sub_section'] == subsection_]['section'].unique()[0]

    df_aux['date'] = df_aux.index


    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df_aux['date'],
                             y=df_aux[variable],
                             name=variable + ' ' + units_[variable]))

    fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
    fig.update_layout(
        title='{} - timeseries - Section:{}-Lane:{}'.format(road, int(section_), int(lane_)),
        xaxis_title='Time',
        yaxis_title='',
        showlegend=True
    )

    # Add range slider
    fig.update_layout(
        xaxis=go.layout.XAxis(
            rangeselector=dict(
                buttons=list([
                    dict(count=1,
                         label="1 month",
                         step="month",
                         stepmode="backward"),
                    dict(count=6,
                         label="6 month",
                         step="month",
                         stepmode="backward"),
                    dict(count=7,
                         label="1 week",
                         step="day",
                         stepmode="backward"),
                    dict(count=1,
                         label="1 year",
                         step="year",
                         stepmode="backward"),
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            type="date"
        )
    )
    fig.show()


def plot_time_series_by_date(df, variable, start_date, end_date):
    """

    :param df:
    :param variable:
    :param start_date:
    :param end_date:
    :return:
    """

    from dl_ia_cla_settings import units_
    import plotly.graph_objects as go

    # filter by dates
    df_aux = df[(df.index > start_date) & (df.index < end_date)]

    # filter by sectopm
    section_ = 1
    df_aux = df_aux[(df_aux['section'] == section_)]

    # filter by lanes
    df_aux = df_aux[(df['lanes'] == 1)]

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df_aux[variable].index,
                             y=df_aux[variable],
                             mode='markers',
                             name=variable + ' ' + units_[variable]))

    fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
    fig.update_layout(showlegend=True)
    fig.update_yaxes(title_text=variable + ' ' + units_[variable])
    fig.update_xaxes(title_text='Time (15 min)')
    fig.update_layout(title='{}  - Time series dates: {} to {} section:{}'.format(road, start_date, end_date, section_))

    # Add range slider
    fig.update_layout(
        xaxis=go.layout.XAxis(
            rangeselector=dict(
                buttons=list([
                    dict(count=1,
                         label="1 day",
                         step="day",
                         stepmode="backward"),
                    dict(count=3,
                         label="3 days",
                         step="day",
                         stepmode="backward"),
                    dict(count=7,
                         label="1 week",
                         step="day",
                         stepmode="backward")
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            type="date"
        )
    )
    # now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
    fig.write_html("figures\\{}_timeseries_{}.html".format(road, variable))
    # fig.show()



def plot_time_series_segmented(df, start_date, end_date, road, variable, group_variable):
    """
        definition

    :param df:
    :param start_date:
    :param end_date:
    :param road:
    :return:
    """

    import plotly.graph_objects as go
    from dl_ia_cla_settings import units_

    fig = go.Figure()

    for group in df[group_variable].unique():
        df_aux = df[df[group_variable] == group]
        df_aux = df_aux.loc[start_date:end_date]

        t = df_aux.index
        y = df_aux[variable]

        fig.add_trace(go.Scatter(x=t,
                                 y=y,
                                 mode='markers',
                                 name=group,
                                 opacity=0.6))

    # fig.update_xaxes(title_text='Time (15 min)')
    fig.update_yaxes(title_text=variable + ' ' + units_[variable])

    fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
    fig.update_layout(title='{}  - Time series - {} to {}'.format(road, start_date, end_date))
    fig.update_layout(showlegend=True)

    # Add range slider
    fig.update_layout(
        xaxis=go.layout.XAxis(
            rangeselector=dict(
                buttons=list([
                    dict(count=1,
                         label="1 day",
                         step="day",
                         stepmode="backward"),
                    dict(count=3,
                         label="3 days",
                         step="day",
                         stepmode="backward"),
                    dict(count=7,
                         label="1 week",
                         step="day",
                         stepmode="backward")
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            type="date"
        )
    )

    # now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
    fig.write_html("figures\\{}_timeseries_segmented_{}.html".format(road, variable))
    # fig.show()




def plot_time_series_by_section(df, road, variable, start_date, end_date):
    """
        definition

    :param df:
    :param start_date:
    :param end_date:
    :param road:
    :return:
    """

    df_aux = df[(df.index > start_date) & (df.index < end_date)]
    section_ = df_aux['section'].unique()[0]

    df_aux = df_aux[df_aux['section'] == section_]


    import plotly.graph_objects as go
    from dl_ia_cla_settings import units_

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df_aux[variable].index,
                             y=df_aux[variable],
                             mode='markers',
                             name='{}-Section:{}'.format(variable, str(section_))))

    fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
    fig.update_layout(showlegend=True)
    fig.update_yaxes(title_text=variable + ' ' + units_[variable])
    fig.update_xaxes(title_text='Time (15 min)')
    fig.update_layout(title='{}  - Time series'.format(road))

    # Add range slider
    fig.update_layout(
        xaxis=go.layout.XAxis(
            rangeselector=dict(
                buttons=list([
                    dict(count=1,
                         label="1 day",
                         step="day",
                         stepmode="backward"),
                    dict(count=3,
                         label="3 days",
                         step="day",
                         stepmode="backward"),
                    dict(count=7,
                         label="1 week",
                         step="day",
                         stepmode="backward")
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            type="date"
        )
    )
    # now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
    fig.write_html("figures\\{}_timeseries_by_section_{}.html".format(road, variable))
    # fig.show()



def plot_time_series_by_speed_bandwith(df, road, variable, start_date, end_date):
    """
        definition

    :param df:
    :param start_date:
    :param end_date:
    :param road:
    :return:
    """

    df_aux = df[(df.index > start_date) & (df.index < end_date)]

    # Plot all sections

    df_aux1 = df_aux[(df_aux['speed'] < 80) & (df_aux['speed'] >= 70)]
    df_aux2 = df_aux[(df_aux['speed'] < 70) & (df_aux['speed'] >= 50)]

    print(df_aux1[variable].describe())
    print(df_aux2[variable].describe())

    t1 = df_aux1[variable].index
    t2 = df_aux2[variable].index

    y1 = df_aux1[variable]
    y2 = df_aux2[variable]

    import plotly.graph_objects as go
    from dl_ia_cla_settings import units_

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=t1,
                             y=y1,
                             mode='markers',
                             marker=dict(color='blue'),
                             name='{} with speed in (80-70)'.format(variable)))

    fig.add_trace(go.Scatter(x=t2,
                             y=y2,
                             mode='markers',
                             marker=dict(color='gray'),
                             name='{} with speed in (70-50)'.format(variable)))

    fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
    fig.update_layout(showlegend=True)
    fig.update_yaxes(title_text=variable + ' ' + units_[variable])
    fig.update_xaxes(title_text='Time (15 min)')
    fig.update_layout(title='{}  - Time series '.format(road))

    # Add range slider
    fig.update_layout(
        xaxis=go.layout.XAxis(
            rangeselector=dict(
                buttons=list([
                    dict(count=1,
                         label="1 day",
                         step="day",
                         stepmode="backward"),
                    dict(count=3,
                         label="3 days",
                         step="day",
                         stepmode="backward"),
                    dict(count=7,
                         label="1 week",
                         step="day",
                         stepmode="backward")
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            type="date"
        )
    )
    # now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
    fig.write_html("figures\\{}_timeseries_by_speed_bandwith_{}.html".format(road, variable))
    fig.show()





def plot_time_series_congestion_by_day_by_lane(df, df_congestions):
    """
        definition

    :param df:
    :param start_date:
    :param end_date:
    :param road:
    :return:
    """



    import plotly.graph_objects as go
    from dl_ia_cla_settings import create_time_array

    t = create_time_array('00:00:00', '23:59:59', '15min')

    fig = go.Figure()

    for day in ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']:
        y_congestion = df_congestions['Congestion_{}'.format(day)]

        fig.add_trace(go.Scatter(x=t,
                                 y=y_congestion,
                                 mode='markers+lines',
                                 name='({}) Avg. Congestion (%)'.format(day)))

        # to plot volume and gap

        df_aux = df[df['weekday'] == dict_days[day]]

        subsection_ = df['sub_section'].unique()[0]

        df_aux = df_aux[df_aux['sub_section'] == subsection_]

        lane_ = df_aux['lanes'].unique()
        section_ = df_aux['section'].unique()[0]

        y_volume = df_aux.groupby(['hour', 'minute'])['volume'].mean()
        fig.add_trace(go.Scatter(x=t,
                                 y=y_volume.values,
                                 mode='markers+lines',
                                 name='({}) Avg. Volume (Veh./15min)'.format(day)))

        if 1 == 1:
            y_speed = df_aux.groupby(['hour', 'minute'])['speed'].mean()
            fig.add_trace(go.Scatter(x=t,
                                     y=y_speed.values,
                                     mode='markers+lines',
                                     name='({}) Avg. speed (km/h)'.format(day)))

        if 1 == 0:
            y_gap = df_aux.groupby(['hour', 'minute'])['gap'].mean()
            fig.add_trace(go.Scatter(x=t,
                                     y=y_gap.values * 0.1,
                                     mode='markers+lines',
                                     name='({}) Avg. Gap (1/10 sec)'.format(day)))

    fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
    fig.update_layout(showlegend=True)
    fig.update_yaxes(title_text=' ')
    fig.update_xaxes(title_text='Time (15 min)')
    fig.update_layout(title='{} - Section:{}  -Lane:{}  '.format(road, section_, int(lane_)))

    # now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
    fig.write_html("figures\\{}_congestion.html".format(road))
    fig.show()



def plot_time_series_congestion_by_day_by_section(df, df_congestions):
    """
        definition

    :param df:
    :param start_date:
    :param end_date:
    :param road:
    :return:
    """



    import plotly.graph_objects as go
    from dl_ia_cla_settings import create_time_array

    t = create_time_array('00:00:00', '23:59:59', '15min')

    fig = go.Figure()

    for day in ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']:
        y_congestion = df_congestions['Congestion_{}'.format(day)]

        fig.add_trace(go.Scatter(x=t,
                                 y=y_congestion,
                                 mode='markers+lines',
                                 name='({}) Avg. Congestion (%)'.format(day)))

        # to plot volume and gap

        df_aux = df[df['weekday'] == dict_days[day]]

        section_ = df['section'].unique()[0]

        df_aux = df_aux[df_aux['section'] == section_]

        y_volume = df_aux.groupby(['hour', 'minute'])['volume'].mean()
        fig.add_trace(go.Scatter(x=t,
                                 y=y_volume.values,
                                 mode='markers+lines',
                                 name='({}) Avg. Volume (Veh./15min)'.format(day)))

        if 1 == 1:
            y_speed = df_aux.groupby(['hour', 'minute'])['speed'].mean()
            fig.add_trace(go.Scatter(x=t,
                                     y=y_speed.values,
                                     mode='markers+lines',
                                     name='({}) Avg. speed (km/h)'.format(day)))

        if 1 == 0:
            y_gap = df_aux.groupby(['hour', 'minute'])['gap'].mean()
            fig.add_trace(go.Scatter(x=t,
                                     y=y_gap.values * 0.1,
                                     mode='markers+lines',
                                     name='({}) Avg. Gap (1/10 sec)'.format(day)))

    fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
    fig.update_layout(showlegend=True)
    fig.update_yaxes(title_text=' ')
    fig.update_xaxes(title_text='Time (15 min)')
    fig.update_layout(title='{} - Section:{} '.format(road, section_))

    # now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
    fig.write_html("figures\\{}_congestion_by_day_by_section_{}.html".format(road, section_))
    fig.show()



def plot_time_series_congestion_total_by_section(df, df_congestions):
    """
        definition

    :param df:
    :param start_date:
    :param end_date:
    :param road:
    :return:
    """



    import plotly.graph_objects as go
    import numpy as np
    from dl_ia_cla_settings import create_time_array

    t = create_time_array('00:00:00', '23:59:59', '15min')

    fig = go.Figure()

    Y_congestion = []
    for i in range(0, len(df_congestions)):
        Y_congestion.append(df_congestions.iloc[i, :].mean())

    fig.add_trace(go.Scatter(x=t,
                             y=Y_congestion,
                             mode='markers+lines',
                             name='Avg. Congestion Brisbane 2019 (%)'))

    # to plot volume and gap

    color_1 = {1: 'brown', 2: 'crimson', 3: 'darkred', 4: 'tomato'}
    color_2 = {1: 'sandybrown', 2: 'salmon', 3: 'orange', 4: 'goldenrod'}

    for section_ in np.sort(df['section'].unique()):
        df_aux = df[df['section'] == section_]

        y_volume = df_aux.groupby(['hour', 'minute'])['volume'].mean()

        fig.add_trace(go.Scatter(x=t,
                                 y=y_volume.values,
                                 mode='markers+lines',
                                 marker=dict(color=color_1[section_]),
                                 name='Section:{} Avg. Volume (Veh./15min)'.format(section_)))

        y_speed = df_aux.groupby(['hour', 'minute'])['speed'].mean()
        fig.add_trace(go.Scatter(x=t,
                                 y=y_speed.values,
                                 mode='markers+lines',
                                 marker=dict(color=color_2[section_]),
                                 name='Section:{} Avg. speed (km/h)'.format(section_)))

    fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
    fig.update_layout(showlegend=True)
    fig.update_yaxes(title_text=' ')
    fig.update_xaxes(title_text='Time (15 min)')
    fig.update_layout(title='{} -'.format(road))

    # now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
    fig.write_html("figures\\{}_congestion_total_by_section.html".format(road))
    fig.show()



def plot_time_series_congestion_total_by_lane(df, df_congestions):
    """
        definition

    :param df:
    :param start_date:
    :param end_date:
    :param road:
    :return:
    """



    import plotly.graph_objects as go
    import numpy as np
    from dl_ia_cla_settings import create_time_array

    t = create_time_array('00:00:00', '23:59:59', '15min')

    fig = go.Figure()

    Y_congestion = []
    for i in range(0, len(df_congestions)):
        Y_congestion.append(df_congestions.iloc[i, :].mean())

    fig.add_trace(go.Scatter(x=t,
                             y=Y_congestion,
                             mode='markers+lines',
                             name='Avg. Congestion Brisbane 2019 (%)'))

    # to plot volume and gap

    color_1 = {1: 'brown', 2: 'crimson', 3: 'darkred', 4: 'tomato'}
    color_2 = {1: 'sandybrown', 2: 'salmon', 3: 'orange', 4: 'goldenrod'}

    for subsection_ in np.sort(df['sub_section'].unique()):
        df_aux = df[df['sub_section'] == subsection_]

        lane_ = df_aux['lanes'].unique()
        section_ = df_aux['section'].unique()[0]

        y_volume = df_aux.groupby(['hour', 'minute'])['volume'].mean()

        fig.add_trace(go.Scatter(x=t,
                                 y=y_volume.values,
                                 mode='markers+lines',
                                 marker=dict(color=color_1[section_]),
                                 name='Section:{} Lane:{} Avg. Volume (Veh./15min)'.format(section_, int(lane_))))

        y_speed = df_aux.groupby(['hour', 'minute'])['speed'].mean()
        fig.add_trace(go.Scatter(x=t,
                                 y=y_speed.values,
                                 mode='markers+lines',
                                 marker=dict(color=color_2[section_]),
                                 name='Section:{} Lane:{} Avg. speed (km/h)'.format(section_, int(lane_))))

    fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
    fig.update_layout(showlegend=True)
    fig.update_yaxes(title_text=' ')
    fig.update_xaxes(title_text='Time (15 min)')
    fig.update_layout(title='{} -'.format(road))

    # now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
    fig.write_html("figures\\{}_congestion_total_by_lane.html".format(road))
    fig.show()




def plot_time_series_congestion_total(df, df_congestions):
    """
        definition

    :param df:
    :param start_date:
    :param end_date:
    :param road:
    :return:
    """



    import plotly.graph_objects as go
    from dl_ia_cla_settings import create_time_array

    t = create_time_array('00:00:00', '23:59:59', '15min')

    fig = go.Figure()

    Y_congestion = []
    for i in range(0, len(df_congestions)):
        Y_congestion.append(df_congestions.iloc[i, :].mean())

    fig.add_trace(go.Scatter(x=t,
                             y=Y_congestion,
                             mode='markers+lines',
                             name='Avg. Congestion Brisbane 2019 (%)'))

    y_volume = df.groupby(['hour', 'minute'])['volume'].mean()

    fig.add_trace(go.Scatter(x=t,
                             y=y_volume.values,
                             mode='markers+lines',
                             name='Avg. Volume (Veh./15min)'))

    y_speed = df.groupby(['hour', 'minute'])['speed'].mean()
    fig.add_trace(go.Scatter(x=t,
                             y=y_speed.values,
                             mode='markers+lines',
                             name='Avg. speed (km/h)'))

    fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
    fig.update_layout(showlegend=True)
    fig.update_yaxes(title_text=' ')
    fig.update_xaxes(title_text='Time (15 min)')
    fig.update_layout(title='{}  '.format(road))

    fig.write_html("figures\\{}_congestion_total.html".format(road))
    fig.show()




def plot_time_series_smooth(df, road, variable, start_date, end_date, smooth):
    """
        definition

    :param df:
    :param start_date:
    :param end_date:
    :param road:
    :return:
    """
    from scipy.signal import savgol_filter
    import numpy as np
    import plotly.graph_objects as go
    from dl_ia_cla_settings import units_

    df_aux = df[(df.index > start_date) & (df.index < end_date)]

    subsection_ = df_aux['sub_section'].unique()[0]
    df_aux = df_aux[df_aux['sub_section'] == subsection_]

    print(df_aux[variable].describe())

    t = df_aux[variable].index
    y = df_aux[variable]

    if smooth == 0:
        print('no smooth')

    if smooth == 1:
        smooth_label = 'Savgol filter'

        yhat = savgol_filter(y, 15, 2)  # window size 51, polynomial order 3

    if smooth == 2:
        smooth_label = 'Convolute'

        def smooth(y, box_pts):
            box = np.ones(box_pts) / box_pts
            y_smooth = np.convolve(y, box, mode='same')
            return y_smooth

        yhat = smooth(y, 3)

    if smooth == 3:
        smooth_label = 'Rolling'
        yhat = y.rolling(window=5, min_periods=5, center=True).median()

    if smooth == 4:
        smooth_label = 'Interpolate'
        from scipy.interpolate import interp1d

        x = np.arange(0, len(t))
        xx = np.linspace(x, x, 1000)

        # interpolate + smooth
        itp = interp1d(t, y.values, kind='linear')

        window_size, poly_order = 101, 3
        yhat = savgol_filter(itp(xx), window_size, poly_order)

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=t,
                             y=y,
                             mode='markers',
                             marker=dict(color='blue'),
                             name='{} data'.format(variable)))
    if smooth != 0:
        fig.add_trace(go.Scatter(x=t,
                                 y=yhat,
                                 mode='lines',
                                 marker=dict(color='gray'),
                                 name='{} smooth {}'.format(variable, smooth_label)))

    fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
    fig.update_layout(showlegend=True)
    fig.update_yaxes(title_text=variable + ' ' + units_[variable])
    fig.update_xaxes(title_text='Time (15 min)')
    fig.update_layout(title='{}  - Time series'.format(road))

    # Add range slider
    fig.update_layout(
        xaxis=go.layout.XAxis(
            rangeselector=dict(
                buttons=list([
                    dict(count=1,
                         label="1 day",
                         step="day",
                         stepmode="backward"),
                    dict(count=3,
                         label="3 days",
                         step="day",
                         stepmode="backward"),
                    dict(count=7,
                         label="1 week",
                         step="day",
                         stepmode="backward")
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            type="date"
        )
    )
    # now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
    fig.write_html("figures\\{}_timeseries_smooth_{}.html".format(road, variable))
    fig.show()




""" ===================== Reference Pattern ========================
"""



def plot_reference_daily_pattern(df, road, variable):
    """ This function do this-

    :param df:
    :param road:
    :param columns_:
    :return:
    """
    import plotly.graph_objects as go
    from dl_ia_cla_settings import units_
    from dl_ia_cla_settings import create_time_array

    t = create_time_array('00:00:00', '23:59:59', '15min')

    df_mean = df.groupby(by=['hour', 'minute'])[variable].mean()

    df_std = df.groupby(by=['hour', 'minute'])[variable].std()


    y_mean = df_mean.values

    y_upper_std = df_mean.values + df_std.values
    y_lower_std = df_mean.values - df_std.values
    y_lower_std[y_lower_std < 0] = 0

    y_upper_std2 = df_mean.values + 2 * df_std.values
    y_lower_std2 = df_mean.values - 2 * df_std.values
    y_lower_std2[y_lower_std2 < 0] = 0

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t,
                             y=y_mean,
                             mode='markers+lines',
                             name='Mean Value',
                             marker=dict(color='blue'),
                             opacity=0.6))

    fig.add_trace(go.Scatter(x=t,
                             y=y_upper_std,
                             name='Mean Value + 1 Standard deviation',
                             marker=dict(color='red'),
                             mode='lines',
                             opacity=0.6))

    fig.add_trace(go.Scatter(x=t,
                             y=y_lower_std,
                             name='Mean Value - 1 Standard deviation',
                             marker=dict(color='red'),
                             mode='lines',
                             opacity=0.6))

    fig.add_trace(go.Scatter(x=t,
                             y=y_upper_std2,
                             name='Mean Value + 2 Standard deviation',
                             marker=dict(color='darkred'),
                             mode='lines',
                             opacity=0.6))

    fig.add_trace(go.Scatter(x=t,
                             y=y_lower_std2,
                             name='Mean Value - 2 Standard deviation',
                             marker=dict(color='darkred'),
                             mode='lines',
                             opacity=0.6))

    index_ = df.index.dayofyear.unique()

    if 1 == 0:
        for days_ in index_:
            df_aux = df[df.index.dayofyear == days_]

            df_aux = df_aux[df_aux['section'] == df_aux['section'].unique()[0]]

            fig.add_trace(go.Scatter(x=t,
                                     y=df_aux[variable].values,

                                     name='Data',
                                     marker=dict(color='gray'),
                                     mode='markers',
                                     opacity=0.6))

    fig.update_yaxes(title_text=variable + ' ' + units_[variable])
    fig.update_xaxes(title_text='time (15 min)')

    fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
    fig.update_layout(title='{}  - Time series - (Reference Pattern)'.format(road))
    fig.update_layout(showlegend=True)
    # now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
    fig.write_html("figures\\{}_reference_pattern_{}.html".format(road, variable))
    fig.show()




def plot_reference_daily_pattern_by_weekend(df, road, variable):
    """ This function do this-

    :param df:
    :param road:
    :param columns_:
    :return:
    """


    import plotly.graph_objects as go
    from datetime import datetime
    from dl_ia_cla_settings import create_time_array

    t = create_time_array('00:00:00', '23:59:59', '15min')

    fig = go.Figure()

    weekend_dict = {1: 'weekend',
                    0: 'weekday'}

    for week_day_ in df['weekend'].unique():
        df_aux = df[df['weekend'] == week_day_]

        df_mean = df_aux.groupby(by=['hour', 'minute'])[variable].mean()
        df_std = df_aux.groupby(by=['hour', 'minute'])[variable].std()
        y_mean = df_mean.values
        y_upper_std1 = df_mean.values + df_std.values
        y_lower_std1 = df_mean.values - df_std.values
        y_lower_std1[y_lower_std1 < 0] = 0

        y_upper_std2 = df_mean.values + 2 * df_std.values
        y_lower_std2 = df_mean.values - 2 * df_std.values
        y_lower_std2[y_lower_std2 < 0] = 0

        fig.add_trace(go.Scatter(x=t,
                                 y=y_mean,
                                 mode='markers+lines',
                                 marker=dict(color='darkred'),
                                 name='Mean Value ({})'.format(weekend_dict[week_day_]),
                                 opacity=0.6))

        fig.add_trace(go.Scatter(x=t,
                                 y=y_upper_std1,
                                 marker=dict(color='blue'),
                                 name='Mean Value + 1 Standard deviation ({})'.format(weekend_dict[week_day_]),
                                 mode='lines',
                                 opacity=0.6))

        fig.add_trace(go.Scatter(x=t,
                                 y=y_lower_std1,
                                 marker=dict(color='blue'),
                                 name='Mean Value - 1 Standard deviation ({})'.format(weekend_dict[week_day_]),
                                 mode='lines',
                                 opacity=0.6))

        fig.add_trace(go.Scatter(x=t,
                                 y=y_upper_std2,
                                 marker=dict(color='darkblue'),
                                 name='Mean Value + 2 Standard deviation ({})'.format(weekend_dict[week_day_]),
                                 mode='lines',
                                 opacity=0.6))

        fig.add_trace(go.Scatter(x=t,
                                 y=y_lower_std2,
                                 marker=dict(color='darkblue'),
                                 name='Mean Value - 2 Standard deviation ({})'.format(weekend_dict[week_day_]),
                                 mode='lines',
                                 opacity=0.6))

    fig.update_yaxes(title_text=variable + ' ' + units_[variable])
    fig.update_xaxes(title_text='time (15 min)')

    fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
    fig.update_layout(title='{}  - Time series - (Reference Pattern)'.format(road))
    fig.update_layout(showlegend=True)
    now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
    # fig.write_html("figures\\{}_timeseries_{}_{}.html".format(road, variable, now, ))
    fig.show()





def plot_reference_daily_pattern_by_speed_bandwith(df, road, variable):
    """ This function do this-

    :param df:
    :param road:
    :param columns_:
    :return:
    """

    import plotly.graph_objects as go
    from dl_ia_cla_settings import units_
    from dl_ia_cla_settings import create_time_array

    t = create_time_array('00:00:00', '23:59:59', '15min')

    fig = go.Figure()

    df_aux = df

    # Plot all sections
    colors_ = [('red', 'darkred'), ('blue', 'darkblue')]

    speed_limits = [(80, 70), (70, 60)]


    for i in range(0, len(speed_limits)):

        # print(i)

        df_aux1 = df_aux[(df_aux['speed'] < speed_limits[i][0]) & (df_aux['speed'] >= speed_limits[i][1])]

        print(df_aux1.describe())

        df_mean = df_aux1.groupby(by=['hour', 'minute'])[variable].mean()
        df_std = df_aux1.groupby(by=['hour', 'minute'])[variable].std()
        y_mean = df_mean.values
        y_upper_1std = df_mean.values + df_std.values
        y_lower_1std = df_mean.values - df_std.values
        y_upper_2std = df_mean.values + 2 * df_std.values
        y_lower_2std = df_mean.values - 2 * df_std.values
        y_lower_1std[y_lower_1std < 0] = 0
        y_lower_2std[y_lower_2std < 0] = 0

        fig.add_trace(go.Scatter(x=t,
                                 y=y_mean,
                                 mode='markers+lines',
                                 marker=dict(color=colors_[0][i]),
                                 name='Mean Value (speed:{}-{})'.format(speed_limits[i][0], speed_limits[i][1]),
                                 opacity=0.6))

        if 0 == 1:
            fig.add_trace(go.Scatter(x=t,
                                     y=y_upper_1std,
                                     mode='lines',
                                     marker=dict(color='blue'),
                                     name='Mean Value + 1 Standard deviation (speed:{}-{})'.format(
                                         speed_limits[i][0], speed_limits[i][1]),
                                     opacity=0.6))

            fig.add_trace(go.Scatter(x=t,
                                     y=y_lower_1std,
                                     mode='markers',
                                     marker=dict(color='blue'),
                                     name='Mean Value - 1 Standard deviation (speed:{}-{})'.format(
                                         speed_limits[i][0], speed_limits[i][1]),
                                     opacity=0.6))

        fig.add_trace(go.Scatter(x=t,
                                 y=y_upper_2std,
                                 mode='markers',
                                 marker=dict(color=colors_[1][i]),
                                 name='Mean Value + 2 Standard deviation (speed:{}-{})'.format(speed_limits[i][0],
                                                                                               speed_limits[i][1]),
                                 opacity=0.6))

        fig.add_trace(go.Scatter(x=t,
                                 y=y_lower_2std,
                                 mode='markers',
                                 marker=dict(color=colors_[1][i]),
                                 name='Mean Value - 2 Standard deviation (speed:{}-{})'.format(speed_limits[i][0],
                                                                                               speed_limits[i][1]),
                                 opacity=0.6))

    fig.update_yaxes(title_text=variable + ' ' + units_[variable])
    fig.update_xaxes(title_text='time (15 min)')

    fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
    fig.update_layout(title='{}  - Time series - (Reference Pattern)'.format(road))
    fig.update_layout(showlegend=True)
    # now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
    fig.write_html("figures\\{}_reference_pattern_by_speed_bandwith_{}.html".format(road, variable))
    fig.show()



def plot_reference_daily_pattern_by_speed_bandwith_per_section(df, road, variable):
    """ This function do this-

    :param df:
    :param road:
    :param columns_:
    :return:
    """

    import plotly.graph_objects as go
    from dl_ia_cla_settings import units_, segments_dict_LW
    from dl_ia_cla_settings import create_time_array

    t = create_time_array('00:00:00', '23:59:59', '15min')

    fig = go.Figure()

    colors_ = [('red', 'darkred'), ('blue', 'darkblue')]
    speed_limits = [(80, 70), (70, 0)]

    df_aux0 = df


    for subsection_ in df_aux0['sub_section'].unique():

        df_aux = df_aux0[df_aux0['sub_section'] == subsection_]

        for i in range(0, len(speed_limits)):
            # print(i)

            df_aux1 = df_aux[(df_aux['speed'] < speed_limits[i][0]) & (df_aux['speed'] >= speed_limits[i][1])]

            print(df_aux1.describe())

            df_mean = df_aux1.groupby(by=['hour', 'minute'])[variable].mean()
            df_std = df_aux1.groupby(by=['hour', 'minute'])[variable].std()
            y_mean = df_mean.values
            # y_upper_1std = df_mean.values + df_std.values
            # y_lower_1std = df_mean.values - df_std.values
            y_upper_2std = df_mean.values + 2 * df_std.values
            y_lower_2std = df_mean.values - 2 * df_std.values
            # y_lower_1std[y_lower_1std < 0] = 0
            y_lower_2std[y_lower_2std < 0] = 0

            fig.add_trace(go.Scatter(x=t,
                                     y=y_mean,
                                     mode='markers+lines',
                                     marker=dict(color=colors_[0][i]),
                                     name='Reference Pattern (speed:{}-{}) section:{}'.format(
                                         speed_limits[i][0],
                                         speed_limits[i][1],
                                         segments_dict_LW[subsection_]),
                                     opacity=0.6))
            if 1 == 0:
                fig.add_trace(go.Scatter(x=t,
                                         y=y_upper_2std,
                                         mode='markers',
                                         marker=dict(color=colors_[1][i]),
                                         name='Upper limit (speed:{}-{}) section:{}'.format(
                                             speed_limits[i][0],
                                             speed_limits[i][1],
                                             segments_dict_LW[subsection_]),
                                         opacity=0.6))

                fig.add_trace(go.Scatter(x=t,
                                         y=y_lower_2std,
                                         mode='markers',
                                         marker=dict(color=colors_[1][i]),
                                         name='Lower Limit (speed:{}-{}) section:{}'.format(
                                             speed_limits[i][0],
                                             speed_limits[i][1],
                                             segments_dict_LW[subsection_]),
                                         opacity=0.6))

    fig.update_yaxes(title_text=variable + ' ' + units_[variable])
    fig.update_xaxes(title_text='time (15 min)')

    fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
    fig.update_layout(title='{}  - Reference Pattern'.format(road))
    fig.update_layout(showlegend=True)
    # now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
    fig.write_html("figures\\{}_reference_pattern_by_speed_bandwith_{}_per_section.html".format(road, variable))
    fig.show()



def plot_gaps_betwen_referece_patterns_by_speed_bandwidth(df, road, variable):
    """ This function do this-

    :param df:
    :param road:
    :param columns_:
    :return:
    """

    import plotly.graph_objects as go
    from dl_ia_cla_settings import units_, segments_dict_LW
    from dl_ia_cla_settings import create_time_array

    t = create_time_array('00:00:00', '23:59:59', '15min')

    fig = go.Figure()

    colors_ = [('red', 'darkred'), ('blue', 'darkblue')]
    speed_limits = [(80, 70), (70, 60)]

    df_aux0 = df



    subsection_ = 2

    df_aux = df_aux0[df_aux0['sub_section'] == subsection_]

    df_aux1 = df_aux[(df_aux['speed'] < speed_limits[0][0]) & (df_aux['speed'] >= speed_limits[0][1])]
    df_mean1 = df_aux1.groupby(by=['hour', 'minute'])[variable].mean()
    y_mean1 = df_mean1.values

    df_aux2 = df_aux[(df_aux['speed'] < speed_limits[1][0]) & (df_aux['speed'] >= speed_limits[1][1])]
    df_mean2 = df_aux2.groupby(by=['hour', 'minute'])[variable].mean()
    y_mean2 = df_mean2.values

    diff = y_mean2 - y_mean1

    fig.add_trace(go.Scatter(x=t,
                             y=diff,
                             mode='markers+lines',
                             marker=dict(color=colors_[0][0]),
                             name=' $ \Delta V_{i,1,t}^{SP0} \hspace{3cm} $',
                             opacity=0.6))

    subsection_ = 3

    df_aux = df_aux0[df_aux0['sub_section'] == subsection_]

    df_aux1 = df_aux[(df_aux['speed'] < speed_limits[0][0]) & (df_aux['speed'] >= speed_limits[0][1])]
    df_mean1 = df_aux1.groupby(by=['hour', 'minute'])[variable].mean()
    y_mean1 = df_mean1.values

    df_aux2 = df_aux[(df_aux['speed'] < speed_limits[1][0]) & (df_aux['speed'] >= speed_limits[1][1])]
    df_mean2 = df_aux2.groupby(by=['hour', 'minute'])[variable].mean()
    y_mean2 = df_mean2.values

    diff = y_mean2 - y_mean1

    fig.add_trace(go.Scatter(x=t,
                             y=diff,
                             mode='markers+lines',
                             marker=dict(color=colors_[0][1]),
                             name=' $ \Delta V_{i,2,t}^{SP0} \hspace{3cm} $',
                             opacity=0.6))

    fig.update_yaxes(title_text=variable + ' ' + units_[variable])
    fig.update_xaxes(title_text='time (15 min)')

    fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
    fig.update_layout(title='{}  - Reference Pattern - {}'.format(road, segments_dict_LW[subsection_]))
    fig.update_layout(showlegend=True)
    # now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
    fig.write_html("figures\\{}_reference_pattern_by_speed_bandwith_{}_per_section.html".format(road, variable))
    fig.show()




def plot_two_timeserie(y1, y2, var_1, var_2, road):
    """ This plots two time series from a dataframe

    :param df: dataframe with the two time series to plot and a column timestamp
    :param var_1: first variable to plot (str)
    :param var_2: second variarble to plot (str)
    :param road: name of the road from which the variables belongs
    :return:
    """

    import plotly.graph_objects as go

    print_ = True
    show = True

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=y1.index,
                             y=y1,
                             mode='markers+lines',
                             marker=dict(color='red'),
                             name='{}'.format(name1)))

    fig.add_trace(go.Scatter(x=y2.index,
                             y=y2,
                             mode='markers+lines',
                             marker=dict(color='blue'),
                             name='{}'.format(name2)))

    fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
    fig.update_layout(showlegend=True)
    fig.update_yaxes(title_text=' ')
    fig.update_xaxes(title_text='Time')
    fig.update_layout(title='Timeseries plot - {}'.format(road))
    if show:
        fig.show()
    if print_:
        fig.write_html("figures\\{}_congestion.html".format(road))





def plot_daily_patterns(y1, y2, name1, name2):
    """ This function do this-

    :param df:
    :param road:
    :param columns_:
    :return:
    """

    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y1.index,
                             y=y1,
                             mode='markers+lines',
                             marker=dict(color='red'),
                             name='{} Section:{} Lane:{} Day:{} '.format(name1)))
    fig.add_trace(go.Scatter(x=y2.index,
                             y=y2,
                             mode='markers+lines',
                             marker=dict(color='blue'),
                             name='{} Section:{} Lane:{} Day:{} '.format(name2)))

    fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
    fig.update_layout(showlegend=True)
    fig.update_yaxes(title_text=' ')
    fig.update_xaxes(title_text='Time (15 min)')
    fig.update_layout(title='{} -  '.format(road))
    # fig.write_html("figures\\{}_congestion.html".format(road))
    fig.show()


def plot_scatter_plotly(df, variable_x, variable_y):
    """ This function do this-

    :param df:
    :param road:
    :param columns_:
    :return:
    """
    import plotly.graph_objects as go
    from dl_ia_cla_settings import units_

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[variable_x],
                             y=df[variable_y],
                             mode='markers',
                             marker=dict(color='blue')))

    fig.update_xaxes(title_text=variable_x + ' ' + units_[variable_y])
    fig.update_yaxes(title_text=variable_y + ' ' + units_[variable])

    fig.update_layout(font=dict(family="Courier New, monospace", size=12, color="#7f7f7f"))
    fig.update_layout(title=road)
    fig.update_layout(showlegend=False)
    # fig.show()


def plot_scatter_mpl(df, variable):
    """ This function do this-

    :param df:
    :param road:
    :param columns_:
    :return:
    """

    import matplotlib.pyplot as plt

    print_ = True
    show = True

    fig = plt.figure()

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.scatter(df['occupancy'],
                df[variable])

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.scatter(df['speed'],
                df[variable])

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.scatter(df['headway'],
                df[variable])

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.scatter(df['meanlength'],
                df[variable])

    if print_:
        plt.show()
    if show:
        plt.savefig('figures\\plot_scatter_mpl.png')

def plot_scatter_together(df):
    """ This function do this-

    :param df:
    :param road:
    :param columns_:
    :return:
    """

    import plotly.graph_objects as go
    from dl_ia_cla_settings import units_
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=2, cols=2)

    fig.add_trace(go.Scatter(x=df['occupancy'],
                             y=df['flow'],
                             mode='markers',
                             name='Occupacy-flow',
                             marker=dict(color='blue')), row=1, col=1)

    fig.add_trace(go.Scatter(x=df['speed'],
                             y=df['flow'],
                             mode='markers',
                             name='speed-flow',
                             marker=dict(color='blue')), row=2, col=1)

    fig.add_trace(go.Scatter(x=df['headway'],
                             y=df['flow'],
                             mode='markers',
                             name='headway-flow',
                             marker=dict(color='blue')), row=1, col=2)

    fig.add_trace(go.Scatter(x=df['meanlength'],
                             y=df['flow'],
                             mode='markers',
                             name='meanlength-flow',
                             marker=dict(color='blue')), row=2, col=2)

    fig.update_xaxes(title_text="occupancy" + ' ' + units_['occupancy'], row=1, col=1)
    fig.update_yaxes(title_text="flow" + ' ' + units_['flow'], row=1, col=1)

    fig.update_xaxes(title_text="speed" + ' ' + units_['speed'], row=2, col=1)
    fig.update_yaxes(title_text="flow" + ' ' + units_['flow'], row=2, col=1)

    fig.update_xaxes(title_text="headway" + ' ' + units_['headway'], row=1, col=2)
    fig.update_yaxes(title_text="flow" + ' ' + units_['flow'], row=1, col=2)

    fig.update_xaxes(title_text="meanlength" + ' ' + units_['meanlength'], row=2, col=2)
    fig.update_yaxes(title_text="flow" + ' ' + units_['flow'], row=2, col=2)

    fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
    fig.update_layout(title=road)
    fig.update_layout(showlegend=False)
    # fig.show()



def plot_simulation(df_sim):
    """ This function do this-

    :param df:
    :param road:
    :param columns_:
    :return:
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=4, cols=1,
                        subplot_titles=("Volume Closure Section: {} Direction:{}".format(section_to_close, direction_),
                                        "Speed Closure Section: {} Direction:{}".format(section_to_close, direction_),
                                        "Volume Previous Section: {} Direction:{}".format(prev_section, direction_),
                                        "Speed Previous Section: {} Direction:{}".format(prev_section, direction_)))

    fig.add_trace(go.Scatter(x=df_sim['volume_prev_lane_{}'.format(lane_to_close)].index,
                             y=df_sim['volume_prev_lane_{}'.format(lane_to_close)],
                             mode='markers+lines',
                             marker=dict(color='red'),
                             line=dict(width=3),
                             legendgroup="lane1",
                             name='{}'.format('lane 1')), row=3, col=1)

    fig.add_trace(go.Scatter(x=df_sim['volume_prev_lane_{}'.format(lane_free)].index,
                             y=df_sim['volume_prev_lane_{}'.format(lane_free)],
                             mode='markers+lines',
                             line=dict(width=3),
                             legendgroup="lane2",
                             marker=dict(color='darkred'),
                             name='{}'.format('lane 2')), row=3, col=1)

    fig.add_trace(go.Scatter(x=df_sim['volume_prev_lane_{}_modify'.format(lane_to_close)].index,
                             y=df_sim['volume_prev_lane_{}_modify'.format(lane_to_close)],
                             mode='markers+lines',
                             line=dict(width=2, dash='dash'),
                             marker=dict(color='red'),
                             legendgroup="lane1_after",
                             name='{}'.format('lane 1 sim.')), row=3, col=1)

    fig.add_trace(go.Scatter(x=df_sim['volume_prev_lane_{}_modify'.format(lane_free)].index,
                             y=df_sim['volume_prev_lane_{}_modify'.format(lane_free)],
                             mode='markers+lines',
                             line=dict(width=2, dash='dash'),
                             legendgroup="lane2_after",
                             marker=dict(color='darkred'),
                             name='{}'.format('lane 2 sim. ')), row=3, col=1)

    fig.add_trace(go.Scatter(x=df_sim['volume_actual_lane_{}_modify'.format(lane_to_close)].index,
                             y=df_sim['volume_actual_lane_{}_modify'.format(lane_to_close)],
                             mode='markers+lines',
                             legendgroup="lane1_after",
                             line=dict(width=2, dash='dash'),
                             marker=dict(color='red'),
                             showlegend=False,
                             name='{}'.format('lane 1 - sim.')), row=1, col=1)

    fig.add_trace(go.Scatter(x=df_sim['volume_actual_lane_{}_modify'.format(lane_free)].index,
                             y=df_sim['volume_actual_lane_{}_modify'.format(lane_free)],
                             mode='markers+lines',
                             legendgroup="lane2_after",
                             line=dict(width=2, dash='dash'),
                             marker=dict(color='darkred'),
                             showlegend=False,
                             name='{}'.format('lane 2 - sim.')), row=1, col=1)

    fig.add_trace(go.Scatter(x=df_sim['volume_actual_lane_{}'.format(lane_to_close)].index,
                             y=df_sim['volume_actual_lane_{}'.format(lane_to_close)],
                             mode='markers+lines',
                             legendgroup="lane1",
                             line=dict(width=3),
                             marker=dict(color='red'),
                             showlegend=False,
                             name='{}'.format('lane 1')), row=1, col=1)

    fig.add_trace(go.Scatter(x=df_sim['volume_actual_lane_{}'.format(lane_free)].index,
                             y=df_sim['volume_actual_lane_{}'.format(lane_free)],
                             mode='markers+lines',
                             legendgroup="lane2",
                             line=dict(width=3),
                             marker=dict(color='darkred'),
                             showlegend=False,
                             name='{}'.format('lane 1')), row=1, col=1)

    fig.add_trace(go.Scatter(x=df_sim['speed_prev_lane_{}'.format(lane_to_close)].index,
                             y=df_sim['speed_prev_lane_{}'.format(lane_to_close)],
                             mode='markers+lines',
                             legendgroup="lane1",
                             line=dict(width=3),
                             marker=dict(color='red'),
                             showlegend=False,
                             name='{}'.format('Lane 1')), row=4, col=1)

    fig.add_trace(go.Scatter(x=df_sim['speed_prev_lane_{}'.format(lane_free)].index,
                             y=df_sim['speed_prev_lane_{}'.format(lane_free)],
                             mode='markers+lines',
                             legendgroup="lane2",
                             line=dict(width=3),
                             marker=dict(color='darkred'),
                             showlegend=False,
                             name='{}'.format('Lane 2')), row=4, col=1)

    fig.add_trace(go.Scatter(x=df_sim['speed_prev_lane_{}_modify'.format(lane_to_close)].index,
                             y=df_sim['speed_prev_lane_{}_modify'.format(lane_to_close)],
                             mode='markers+lines',
                             line=dict(width=2, dash='dash'),
                             legendgroup="lane1_after",
                             marker=dict(color='red'),
                             showlegend=False,
                             name='{}'.format('Lane 1')), row=4, col=1)

    fig.add_trace(go.Scatter(x=df_sim['speed_prev_lane_{}_modify'.format(lane_free)].index,
                             y=df_sim['speed_prev_lane_{}_modify'.format(lane_free)],
                             mode='markers+lines',
                             line=dict(width=2, dash='dash'),
                             legendgroup="lane2_after",
                             marker=dict(color='darkred'),
                             showlegend=False,
                             name='{}'.format('Lane 2')), row=4, col=1)

    fig.add_trace(go.Scatter(x=df_sim['speed_actual_lane_{}_modify'.format(lane_to_close)].index,
                             y=df_sim['speed_actual_lane_{}_modify'.format(lane_to_close)],
                             mode='markers+lines',
                             legendgroup="lane1_after",
                             line=dict(width=2, dash='dash'),
                             marker=dict(color='red'),
                             showlegend=False,
                             name='{}'.format('Lane 1')), row=2, col=1)

    fig.add_trace(go.Scatter(x=df_sim['speed_actual_lane_{}_modify'.format(lane_free)].index,
                             y=df_sim['speed_actual_lane_{}_modify'.format(lane_free)],
                             mode='markers+lines',
                             legendgroup="lane2_after",
                             marker=dict(color='darkred'),
                             line=dict(width=2, dash='dash'),
                             showlegend=False,
                             name='{}'.format('Lane 2')), row=2, col=1)

    fig.add_trace(go.Scatter(x=df_sim['speed_actual_lane_{}'.format(lane_to_close)].index,
                             y=df_sim['speed_actual_lane_{}'.format(lane_to_close)],
                             mode='markers+lines',
                             legendgroup="lane1",
                             line=dict(width=3),
                             marker=dict(color='red'),
                             showlegend=False,
                             name='{}'.format('Lane 1')), row=2, col=1)

    fig.add_trace(go.Scatter(x=df_sim['speed_actual_lane_{}'.format(lane_free)].index,
                             y=df_sim['speed_actual_lane_{}'.format(lane_free)],
                             mode='markers+lines',
                             legendgroup="lane2",
                             line=dict(width=3),
                             marker=dict(color='darkred'),
                             showlegend=False,
                             name='{}'.format('Lane 2')), row=2, col=1)

    fig.update_layout(font=dict(family="Courier New, monospace", size=11, color="#7f7f7f"))
    fig.update_layout(showlegend=True)
    fig.update_yaxes(title_text=' ')
    # fig.update_xaxes(title_text='Time (15 min)')
    fig.update_layout(title='{} - Road Traffic simulation - date:{} from {}am to {}am [{}] '.format(road, date_.date(), start_closure, end_closure, message))
    fig.write_html("figures\\{}\\road_traffic_simulation.html".format(road, section_to_close, lane_to_close))
    fig.show()






def CLA_comm(msg):
    """ Funtion to show mesages in terminal

    :parm msg: meassge (str)
    :return:
    """
    print('{} {}'.format('-'*20, msg))


units_ = {"FLOW": "(veh./h)",
          'flow':"(veh./h)",
          "flow_eq":"(veh./h)",
          "flow_error":"(veh./h)",
          "VOLUME": "(veh./15min)",
          'volume':'(veh./15min)',
          'volume_smooth': '(veh./15min)',
          'volume_desc': '(veh./15min)',
          'speed':'(km/h)',
          'speed_eq':'(km/h)',
          'speed_error':'(km/h)',
          "SPEED": '(km/h)',
          "meanlength": "(dm)",
          "gap": "(tenths of sec)",
          "headway": '(tenths of sec)',
          'occupancy':'(%)',
          "OCCUPANCY": "(%)"}

colors_ = {'0-6': 'darkblue',
               '7-12': 'darkgreen',
               '13-17': 'darkred',
               '18-21': 'darkviolet',
               '22-23': 'darkgrey',
               }


def speed_eq(Occ, Oc, L, l, avg_speed):
    """

    :param Occ: Critical Occupancy (20%)
    :param Oc: Actual value of occupancy
    :param L: meanlength of the vehicle (4.4-5.5)
    :param l: lop length (2m tunnel, 1 road)
    :param avg_speed: average speed of the vehicles (km/h)
    :return: speed estimation
    """
    import numpy as np
    Lp = L + l
    a0 = (-50 / (Occ ** 2)) * (1 / Lp) ** 2
    speed_formula = avg_speed * np.exp(Oc**2 * a0)

    return speed_formula, a0

def CLA_add_holidays_info(df):
    """ Add columns HOLIDAYS_ID and SPECIAL_EVENT to the dataframe if road traffic data
        for the simlation

        HOLIDAYS_ID
            0:  Non Holidays
            1:  New Years Day
            2:  Australia Day
            3:  Good Friday
            4:  The day after Good
            5:  Easter Sunday
            6:  Easter Monday
            7:  Aznac Day
            8:  Labour Day
            9:  Royal Queensland Show
            10: Queens Birthday
            11: Christmas Eve
            12: Christmas Day
            13: Boxing Day

        SPECIAL_EVENT
            NH: Non Holidays
            HD: Holidays departure
            HV: Holidays Valley
            HR: Holidays Return

        :param df: Dataframe with the road traffic data and TIMESTAMP column
        :return df_out: dataframe with the columns HOLIDAYS_ID and SPECIAL_EVENT

    """

    import numpy as np
    import pandas as pd
    from datetime import timedelta
    from dl_ia_cla_settings import CLA_comm

    df_holidays = pd.read_csv('data\\holidays.csv')
    df_holidays['date'] = pd.to_datetime(df_holidays['date']).map(lambda x: x.strftime('%Y-%m-%d'))
    df_holidays['TIMESTAMP'] = pd.to_datetime(df_holidays['date'])
    df = pd.merge(df, df_holidays[['holidays_id', 'TIMESTAMP']], on=['TIMESTAMP'], how='left')
    del df_holidays
    df['holidays_id'].fillna(0, inplace=True)
    df['holidays_id'] = df['holidays_id'].astype(int)
    df.rename(columns={'holidays_id': 'HOLIDAYS_ID'}, inplace=True)

    ### define as special days those that are before and after holidays
    CLA_comm('define as special days those that are before and after holidays')

    # initialise variable of special evnet
    df['SPECIAL_EVENT'] = 0
    delta = timedelta(days=1)

    # iter over all holidays events
    for holiday_id in np.sort(df['HOLIDAYS_ID'].unique()):

        # for holidays not in easter period (3,4,5,6) or not holidays (0)
        if np.isin(holiday_id, [0, 3, 4, 5, 6], invert=True):

            # get the day of the week of the holidays event
            day_of_the_week = df[df['HOLIDAYS_ID'] == holiday_id]['WEEKDAY'].unique()[0]

            if day_of_the_week == 1:  # Tuesday
                # get date of the holidays event
                date_special_day = pd.to_datetime(df[df['HOLIDAYS_ID'] == holiday_id]['TIMESTAMP'].unique()[0])

                # Set Holidays return (HR) in tuesday
                df.loc[df['TIMESTAMP'] == date_special_day, ['SPECIAL_EVENT']] = 'HR'  # Tuesday

                # Set Holidays valley (HV) in monday, Saturday and sunday
                df.loc[df['TIMESTAMP'] == date_special_day - 1 * delta, ['SPECIAL_EVENT']] = 'HV'  # Monday
                df.loc[df['TIMESTAMP'] == date_special_day - 2 * delta, ['SPECIAL_EVENT']] = 'HV'  # Sunday
                df.loc[df['TIMESTAMP'] == date_special_day - 3 * delta, ['SPECIAL_EVENT']] = 'HV'  # Saturday

                # Set Holidays Departure (HD) four days before (Friday)
                df.loc[df['TIMESTAMP'] == date_special_day - 4 * delta, ['SPECIAL_EVENT']] = 'HD'  # Friday

            if day_of_the_week == 0:  # Monday
                date_special_day = pd.to_datetime(df[df['HOLIDAYS_ID'] == holiday_id]['TIMESTAMP'].unique()[0])
                df.loc[df['TIMESTAMP'] == date_special_day, ['SPECIAL_EVENT']] = 'HR'  # Monday
                df.loc[df['TIMESTAMP'] == date_special_day - 1 * delta, ['SPECIAL_EVENT']] = 'HV'  # Sunday
                df.loc[df['TIMESTAMP'] == date_special_day - 2 * delta, ['SPECIAL_EVENT']] = 'HV'  # Saturday
                df.loc[df['TIMESTAMP'] == date_special_day - 3 * delta, ['SPECIAL_EVENT']] = 'HD'  # Friday

            if day_of_the_week == 3:  # Thursday
                date_special_day = pd.to_datetime(df[df['HOLIDAYS_ID'] == holiday_id]['TIMESTAMP'].unique()[0])
                df.loc[df['TIMESTAMP'] == date_special_day, ['SPECIAL_EVENT']] = 'HV'  # Thursday
                df.loc[df['TIMESTAMP'] == date_special_day - 1 * delta, ['SPECIAL_EVENT']] = 'HD'  # Wednesday
                df.loc[df['TIMESTAMP'] == date_special_day + 1 * delta, ['SPECIAL_EVENT']] = 'HV'  # Friday
                df.loc[df['TIMESTAMP'] == date_special_day + 2 * delta, ['SPECIAL_EVENT']] = 'HV'  # Saturday
                df.loc[df['TIMESTAMP'] == date_special_day + 3 * delta, ['SPECIAL_EVENT']] = 'HR'  # Sunday

            if day_of_the_week == 4:  # Friday
                date_special_day = pd.to_datetime(df[df['HOLIDAYS_ID'] == holiday_id]['TIMESTAMP'].unique()[0])
                df.loc[df['TIMESTAMP'] == date_special_day, ['SPECIAL_EVENT']] = 'HV'  # Friday
                df.loc[df['TIMESTAMP'] == date_special_day - 1 * delta, ['SPECIAL_EVENT']] = 'HD'  # Thursday
                df.loc[df['TIMESTAMP'] == date_special_day + 1 * delta, ['SPECIAL_EVENT']] = 'HV'  # Friday
                df.loc[df['TIMESTAMP'] == date_special_day + 2 * delta, ['SPECIAL_EVENT']] = 'HV'  # Saturday
                df.loc[df['TIMESTAMP'] == date_special_day + 3 * delta, ['SPECIAL_EVENT']] = 'HR'  # Sunday

            if day_of_the_week == 5:  # Saturday
                date_special_day = pd.to_datetime(df[df['HOLIDAYS_ID'] == holiday_id]['TIMESTAMP'].unique()[0])
                df.loc[df['TIMESTAMP'] == date_special_day, ['SPECIAL_EVENT']] = 'HV'  # saturday
                df.loc[df['TIMESTAMP'] == date_special_day - 1 * delta, ['SPECIAL_EVENT']] = 'HD'  # Friday
                df.loc[df['TIMESTAMP'] == date_special_day + 1 * delta, ['SPECIAL_EVENT']] = 'HR'  # Sunday

            if day_of_the_week == 6:  # Sunday
                date_special_day = pd.to_datetime(df[df['HOLIDAYS_ID'] == holiday_id]['TIMESTAMP'].unique()[0])
                df.loc[df['TIMESTAMP'] == date_special_day, ['SPECIAL_EVENT']] = 'HR'  # Sunday
                df.loc[df['TIMESTAMP'] == date_special_day - 1 * delta, ['SPECIAL_EVENT']] = 'HV'  # Saturday
                df.loc[df['TIMESTAMP'] == date_special_day - 2 * delta, ['SPECIAL_EVENT']] = 'HR'  # Friday

        # easter period 3: good friday, 4: The day after Good Friday, 5:Easter Sunday, 6:Easter Monday
        if np.isin(holiday_id, [3, 4, 5, 6]):
            if holiday_id == 3:  # always friday Departure
                date_special_day = pd.to_datetime(df[df['HOLIDAYS_ID'] == holiday_id]['TIMESTAMP'].unique()[0])
                df.loc[df['TIMESTAMP'] == date_special_day, ['SPECIAL_EVENT']] = 'HD'

            if holiday_id == 4:  # always Saturday valley
                date_special_day = pd.to_datetime(df[df['HOLIDAYS_ID'] == holiday_id]['TIMESTAMP'].unique()[0])
                df.loc[df['TIMESTAMP'] == date_special_day, ['SPECIAL_EVENT']] = 'HV'

            if holiday_id == 5:  # always Sunday valley
                date_special_day = pd.to_datetime(df[df['HOLIDAYS_ID'] == holiday_id]['TIMESTAMP'].unique()[0])
                df.loc[df['TIMESTAMP'] == date_special_day, ['SPECIAL_EVENT']] = 'HV'

            if holiday_id == 5:  # always Monday return
                date_special_day = pd.to_datetime(df[df['HOLIDAYS_ID'] == holiday_id]['TIMESTAMP'].unique()[0])
                df.loc[df['TIMESTAMP'] == date_special_day, ['SPECIAL_EVENT']] = 'HR'

    df.loc[df['SPECIAL_EVENT'] == 0, ['SPECIAL_EVENT']] = 'NH'

    df_out = df

    return df_out



def flow_eq(speed_est, Oc, L, l, F, NL):
    """

    :param speed_est: speed estimation (km/h) from equation
    :param Oc: occupancy
    :param L: meanlength of the vehicle (4.4-5.5)
    :param l: lop length (2m tunnel, 1 road)
    :param F: correction factor (0.1-1.5)
    :param NL: number of lanes
    :return:
    """

    flow_formula = speed_est * (F * NL * 10 * Oc) / (L + l)

    return flow_formula



import pandas as pd

columns_APL = ['flow',
               'volume',
               'speed',
               'meanlength',
               'gap',
               'headway',
               'occupancy']

all_columns_APL = ['time',
                   'volume',
                   'occupancy',
                   'headway',
                   'meanlength',
                   'speed',
                   'name']

columns_APL = ['volume',
               'occupancy',
               'headway',
               'meanlength',
               'speed']

original_columns_APL = ['Name',
                        'Last_Update',
                        'Health_State',
                        'Availability_',
                        'Count_',
                        'Average_Occupancy',
                        'Average_Headway',
                        'Average_Length',
                        'Average_Speed',
                        'Count_Short',
                        'Count_Medium',
                        'Count_Long',
                        'Average_Speed_Short',
                        'Average_Speed_Medium',
                        'Average_Speed_Long']



original_columns_C7 = ['Name',
                       'Last_Update',
                       'Health_State',
                       'Availability_',
                       'Count_',
                       'Average_Occupancy',
                       'Average_Headway',
                       'Average_Length',
                       'Average_Speed',
                       'Count_Short',
                       'Count_Medium',
                       'Count_Long',
                       'Average_Speed_Short',
                       'Average_Speed_Medium',
                       'Average_Speed_Long']

columns_C7 = ['Volume',
              'Occupancy',
              'Headway',
              'Meanlength',
              'Speed']

all_columns_C7 = ['time',
                  'Volume',
                  'Occupancy',
                  'Headway',
                  'Meanlength',
                  'Speed',
                  'Name']

# LL = Left Lane, IL = Inner Left, CL = Central Lane, IR = Inner Right, RL = Right Lane, SL = Single Lane
lanes_dict_inv_APL = {'_L': 1,
                  'IL': 2,
                  '_C': 3,
                  'IR': 4,
                  '_R': 5,
                  '_S': 6}

lanes_dict_APL = {1: 'SL',  # Left Lane (Slow Lane)
                     2: 'IL',  # Interior Left
                     3: 'CL',  # Central Lane
                     4: 'IR',  # Interior Right
                     5: 'FL',  # Right Lane (Fast Lane)
                     6: 'L'}  # Single Lane

lanes_dict = {1: 'SL',  # Left Lane (Slow Lane)
              2: 'IL',  # Interior Left
              3: 'CL',  # Central Lane
              4: 'IR',  # Interior Right
              5: 'FL',  # Right Lane (Fast Lane)
              6: 'L'}  # Single Lane


""" C7 Lanes dict
"""
# (1: Left/slow lane (SL/L),  2:Right/fast lane (FL), 3: Central lane,)
# FL = Fast Lane, SL = Slow Lane, _L = Only Lane
# subsection --> lane
lanes_dict_C7 = {1: 2,
                 2: 1,
                 3: 2,
                 4: 1,
                 5: 2,
                 6: 1,
                 7: 2,
                 8: 1,
                 9: 2,
                 10: 1,
                 11: 2,
                 12: 1,
                 13: 2,
                 14: 1,
                 15: 2,
                 16: 1,
                 17: 2,
                 18: 1,
                 19: 2,
                 20: 1,
                 21: 2,
                 22: 1,
                 23: 2,
                 24: 1,
                 25: 1,
                 26: 2,
                 27: 1,
                 28: 2,
                 29: 1,
                 30: 2,
                 31: 1,
                 32: 2,
                 33: 1,
                 34: 2,
                 35: 1,
                 36: 1,
                 37: 1,
                 38: 1}

""" road topology C7
"""


road_topology_df_C7 = pd.DataFrame(columns=['sub_section_name', 'section', 'direction', 'lanes', 'sub_section',
                                            'prev_section', 'next_section'],
                                   data=[['AVI101_FL', 1, 3, 2, 1, 0, 2],
                                         ['AVI101_SL', 1, 3, 3, 2, 0, 2],
                                         ['AVI102_FL', 2, 3, 2, 3, 1, 3],
                                         ['AVI102_SL', 2, 3, 3, 4, 1, 3],
                                         ['AVI110_FL', 3, 3, 2, 5, 2, 4],
                                         ['AVI110_SL', 3, 3, 3, 6, 2, 4],
                                         ['AVI111_FL', 4, 3, 2, 7, 3, 5],
                                         ['AVI111_SL', 4, 3, 3, 8, 3, 5],
                                         ['AVI117_FL', 5, 3, 2, 9, 4, 6],
                                         ['AVI117_SL', 5, 3, 3, 10, 4, 6],
                                         ['AVI118_FL', 6, 3, 2, 11, 5, 7],
                                         ['AVI118_SL', 6, 3, 3, 12, 5, 7],
                                         ['AVI125_FL', 7, 3, 2, 13, 6, 8],
                                         ['AVI125_SL', 7, 3, 3, 14, 6, 8],
                                         ['AVI126_FL', 8, 3, 2, 15, 7, 9],
                                         ['AVI126_SL', 8, 3, 3, 16, 7, 9],
                                         ['AVI133_FL', 9, 3, 2, 17, 8, 10],
                                         ['AVI133_SL', 9, 3, 3, 18, 8, 10],
                                         ['AVI134_FL', 10, 3, 2, 19, 9, 11],
                                         ['AVI134_SL', 10, 3, 3, 20, 9, 11],
                                         ['AVI141_FL', 11, 3, 2, 21, 10, 12],
                                         ['AVI141_SL', 11, 3, 3, 22, 10, 12],
                                         ['AVI142_FL', 12, 3, 2, 23, 11, 13],
                                         ['AVI142_SL', 12, 3, 3, 24, 11, 13],
                                         ['AVI148_FL', 13, 3, 2, 25, 12, 14],
                                         ['AVI148_SL', 13, 3, 3, 26, 12, 14],
                                         ['AVI149_FL', 14, 3, 2, 27, 13, 15],
                                         ['AVI149_SL', 14, 3, 3, 28, 13, 15],
                                         ['AVI156_FL', 15, 3, 2, 29, 14, 16],
                                         ['AVI156_SL', 15, 3, 3, 30, 14, 16],
                                         ['AVI157_FL', 16, 3, 2, 31, 15, 17],
                                         ['AVI157_SL', 16, 3, 3, 32, 15, 17],
                                         ['AVI162_FL', 17, 3, 2, 33, 16, 18],
                                         ['AVI162_SL', 17, 3, 3, 34, 16, 18],
                                         ['AVI163_FL', 18, 3, 2, 35, 17, 19],
                                         ['AVI163_SL', 18, 3, 3, 36, 17, 19],
                                         ['AVI170_FL', 19, 3, 2, 37, 18, 20],
                                         ['AVI170_SL', 19, 3, 3, 38, 18, 20],
                                         ['AVI171_FL', 20, 3, 2, 39, 19, 21],
                                         ['AVI171_SL', 20, 3, 3, 40, 19, 21],
                                         ['AVI177_FL', 21, 3, 2, 41, 20, 22],
                                         ['AVI177_SL', 21, 3, 3, 42, 20, 22],
                                         ['AVI178_FL', 22, 3, 2, 43, 21, 0],
                                         ['AVI178_SL', 22, 3, 3, 44, 21, 0],
                                         ['AVI585_FL', 23, 4, 2, 45, 0, 24],
                                         ['AVI585_SL', 23, 4, 3, 46, 0, 24],
                                         ['AVI584_FL', 24, 4, 2, 47, 23, 25],
                                         ['AVI584_SL', 24, 4, 3, 48, 23, 25],
                                         ['AVI576_FL', 25, 4, 2, 49, 24, 26],
                                         ['AVI576_SL', 25, 4, 3, 50, 24, 26],
                                         ['AVI575_FL', 26, 4, 2, 51, 25, 27],
                                         ['AVI575_SL', 26, 4, 3, 52, 25, 27],
                                         ['AVI569_FL', 27, 4, 2, 53, 26, 28],
                                         ['AVI569_SL', 27, 4, 3, 54, 26, 28],
                                         ['AVI568_FL', 28, 4, 2, 55, 27, 29],
                                         ['AVI568_SL', 28, 4, 3, 56, 27, 29],
                                         ['AVI562_FL', 29, 4, 2, 57, 28, 30],
                                         ['AVI562_SL', 29, 4, 3, 58, 28, 30],
                                         ['AVI561_FL', 30, 4, 2, 59, 29, 31],
                                         ['AVI561_SL', 30, 4, 3, 60, 29, 31],
                                         ['AVI554_FL', 31, 4, 2, 61, 30, 32],
                                         ['AVI554_SL', 31, 4, 3, 62, 30, 32],
                                         ['AVI553_FL', 32, 4, 2, 63, 31, 33],
                                         ['AVI553_SL', 32, 4, 3, 64, 31, 33],
                                         ['AVI546_FL', 33, 4, 2, 65, 32, 34],
                                         ['AVI546_SL', 33, 4, 3, 66, 32, 34],
                                         ['AVI545_FL', 34, 4, 2, 67, 33, 35],
                                         ['AVI545_SL', 34, 4, 3, 68, 33, 35],
                                         ['AVI539_FL', 35, 4, 2, 69, 34, 36],
                                         ['AVI539_SL', 35, 4, 3, 70, 34, 36],
                                         ['AVI538_FL', 36, 4, 2, 71, 35, 37],
                                         ['AVI538_SL', 36, 4, 3, 72, 35, 37],
                                         ['AVI531_FL', 37, 4, 2, 73, 36, 38],
                                         ['AVI531_SL', 37, 4, 3, 74, 36, 38],
                                         ['AVI530_FL', 38, 4, 2, 75, 37, 39],
                                         ['AVI530_SL', 38, 4, 3, 76, 37, 39],
                                         ['AVI522_FL', 39, 4, 2, 77, 38, 40],
                                         ['AVI522_SL', 39, 4, 3, 78, 38, 40],
                                         ['AVI521_FL', 40, 4, 2, 79, 39, 41],
                                         ['AVI521_SL', 40, 4, 3, 80, 39, 41],
                                         ['AVI516_FL', 41, 4, 2, 81, 40, 42],
                                         ['AVI516_SL', 41, 4, 3, 82, 40, 42],
                                         ['AVI515_FL', 42, 4, 2, 83, 41, 43],
                                         ['AVI515_SL', 42, 4, 3, 84, 41, 43],
                                         ['AVI509_FL', 43, 4, 2, 85, 42, 44],
                                         ['AVI509_SL', 43, 4, 3, 86, 42, 44],
                                         ['AVI508_FL', 44, 4, 2, 87, 43, 45],
                                         ['AVI508_SL', 44, 4, 3, 88, 43, 45],
                                         ['AVI504_FL', 45, 4, 2, 89, 44, 46],
                                         ['AVI504_SL', 45, 4, 3, 90, 44, 46],
                                         ['AVI503_FL', 46, 4, 2, 91, 45, 47],
                                         ['AVI503_SL', 46, 4, 3, 92, 45, 47],
                                         ['AVT002_FL', 47, 3, 2, 93, 0, 1],
                                         ['AVT002_SL', 47, 3, 2, 94, 0, 1],
                                         ['AVI007_L', 48, 3, 1, 95, 0, 3],
                                         ['AVT020_L', 49, 3, 1, 96, 0, 50],
                                         ['AVI401_L', 50, 3, 1, 97, 49, 51],
                                         ['AVI402_L', 51, 3, 1, 98, 50, 52],
                                         ['AVI412_L', 52, 3, 1, 99, 51, 53],
                                         ['AVI414_L', 53, 3, 1, 100, 52, 11],
                                         ['AVT030_L', 54, 3, 1, 101, 22, 55],
                                         ['AVT034_FL', 55, 3, 2, 102, 54, 0],
                                         ['AVT034_SL', 55, 3, 3, 103, 54, 0],
                                         ['AVT031_FL', 56, 3, 2, 104, 22, 0],
                                         ['AVT031_SL', 56, 3, 3, 105, 22, 0],
                                         ['AVT037_L', 57, 4, 1, 106, 0, 58],
                                         ['AVI039_FL', 58, 4, 2, 107, 57, 23],
                                         ['AVT036_FL', 59, 4, 2, 108, 0, 60],
                                         ['AVT036_SL', 59, 4, 3, 109, 0, 60],
                                         ['AVI039_FL', 60, 4, 2, 110, 59, 23],
                                         ['AVT033_SL', 60, 4, 3, 111, 59, 23],
                                         ['AVT035_L', 61, 4, 1, 112, 0, 62],
                                         ['AVT032_FL', 62, 4, 2, 113, 61, 23],
                                         ['AVT032_SL', 62, 4, 3, 114, 61, 23],
                                         ['AVI805_L', 63, 4, 1, 115, 36, 64],
                                         ['AVI806_L', 64, 4, 1, 116, 63, 65],
                                         ['AVI802_FL', 65, 4, 2, 117, 64, 66],
                                         ['AVI802_SL', 65, 4, 3, 118, 64, 66],
                                         ['AVI801_FL', 66, 4, 2, 119, 65, 67],
                                         ['AVI801_SL', 66, 4, 3, 120, 65, 67],
                                         ['AVT021_FL', 67, 4, 2, 121, 66, 0],
                                         ['AVT021_SL', 67, 4, 3, 122, 66, 0],
                                         ['AVT004_L', 68, 4, 1, 123, 46, 0],
                                         ['AVT003_FL', 69, 4, 2, 124, 46, 0],
                                         ['AVT003_SL', 69, 4, 3, 125, 46, 0]
                                         ])

# Encode subsections C7
subsection_dict_C7 = {'AVI584_FL': 1,
                      'AVI584_SL': 2,
                      'AVI576_FL': 3,
                      'AVI576_SL': 4,
                      'AVI569_FL': 5,
                      'AVI569_SL': 6,
                      'AVI562_FL': 7,
                      'AVI562_SL': 8,
                      'AVI554_FL': 9,
                      'AVI554_SL': 10,
                      'AVI546_FL': 11,
                      'AVI546_SL': 12,
                      'AVI539_FL': 13,
                      'AVI539_SL': 14,
                      'AVI531_FL': 15,
                      'AVI531_SL': 16,
                      'AVI522_FL': 17,
                      'AVI522_SL': 18,
                      'AVI516_FL': 19,
                      'AVI516_SL': 20,
                      'AVI509_FL': 21,
                      'AVI509_SL': 22,
                      'AVI503_FL': 23,
                      'AVI503_SL': 24,
                      'AVI806_L': 25,
                      'AVI801_FL': 26,
                      'AVI801_SL': 27,
                      'AVT021_FL': 28,
                      'AVT021_SL': 29,
                      'AVI102_FL': 30,
                      'AVI102_SL': 31,
                      'AVI110_FL': 32,
                      'AVI110_SL': 33,
                      'AVI117_FL': 34,
                      'AVI117_SL': 35,
                      'AVT020_L': 36,
                      'AVI401_L': 37,
                      'AVI414_L': 38}

# Encode sections
section_dict_C7 = {'AVI584': 1,
                   'AVI576': 2,
                   'AVI569': 3,
                   'AVI562': 4,
                   'AVI554': 5,
                   'AVI546': 6,
                   'AVI539': 7,
                   'AVI531': 8,
                   'AVI522': 9,
                   'AVI516': 10,
                   'AVI509': 11,
                   'AVI503': 12,
                   'AVI806': 13,
                   'AVI801': 14,
                   'AVT021': 15,
                   'AVI102': 16,
                   'AVI110': 17,
                   'AVI117': 18,
                   'AVI125': 19,
                   'AVI133': 20,
                   'AVI141': 21,
                   'AVI148': 22,
                   'AVI156': 23,
                   'AVI162': 24,
                   'AVI170': 25,
                   'AVI177': 26,
                   'AVT020': 27,
                   'AVI401': 28,
                   'AVI414': 29}

lane_dict_inv_C7 = {1: 'L',
                    2: 'FL',
                    3: 'SL'}

lane_dict_C7 = {'_L': 1,
                'FL': 2,
                'SL': 3}  # FL = Fast Lane, SL = Slow Lane, _L = Only Lane

road_topology_df_APL = pd.DataFrame([['AVI203_R', 1, 3, 5, 1, 62, 2],
                                     ['AVI203_IR', 1, 3, 4, 2, 62, 2],
                                     ['AVI203_IL', 1, 3, 2, 3, 62, 2],
                                     ['AVI203_L', 1, 3, 1, 4, 62, 2],
                                     ['AVI204_R', 2, 3, 5, 5, 1, 3],
                                     ['AVI204_IR', 2, 3, 4, 6, 1, 3],
                                     ['AVI204_IL', 2, 3, 2, 7, 1, 3],
                                     ['AVI204_L', 2, 3, 1, 8, 1, 3],
                                     ['AVI210_R', 3, 3, 5, 9, 2, 4],
                                     ['AVI210_C', 3, 3, 3, 10, 2, 4],
                                     ['AVI210_L', 3, 3, 1, 11, 2, 4],
                                     ['AVI211_R', 4, 3, 5, 12, 3, 5],
                                     ['AVI211_C', 4, 3, 3, 13, 3, 5],
                                     ['AVI211_L', 4, 3, 1, 14, 3, 5],
                                     ['AVI219_R', 5, 3, 5, 15, 4, 6],
                                     ['AVI219_C', 5, 3, 3, 16, 4, 6],
                                     ['AVI219_L', 5, 3, 1, 17, 4, 6],
                                     ['AVI220_R', 6, 3, 5, 18, 5, 7],
                                     ['AVI220_C', 6, 3, 3, 19, 5, 7],
                                     ['AVI220_L', 6, 3, 1, 20, 5, 7],
                                     ['AVI227_R', 7, 3, 5, 21, 6, 8],
                                     ['AVI227_C', 7, 3, 3, 22, 6, 8],
                                     ['AVI227_L', 7, 3, 1, 23, 6, 8],
                                     ['AVI228_R', 8, 3, 5, 24, 7, 9],
                                     ['AVI228_C', 8, 3, 3, 25, 7, 9],
                                     ['AVI228_L', 8, 3, 1, 26, 7, 9],
                                     ['AVI235_R', 9, 3, 5, 27, 8, 10],
                                     ['AVI235_C', 9, 3, 3, 28, 8, 10],
                                     ['AVI235_L', 9, 3, 1, 29, 8, 10],
                                     ['AVI236_R', 10, 3, 5, 30, 9, 11],
                                     ['AVI236_C', 10, 3, 3, 31, 9, 11],
                                     ['AVI236_L', 10, 3, 1, 32, 9, 11],
                                     ['AVI243_R', 11, 1, 5, 33, 10, 12],
                                     ['AVI243_L', 11, 1, 1, 34, 10, 12],
                                     ['AVI244_R', 12, 1, 5, 35, 11, 13],
                                     ['AVI244_L', 12, 1, 1, 36, 11, 13],
                                     ['AVI250_R', 13, 1, 5, 37, 12, 14],
                                     ['AVI250_L', 13, 1, 1, 38, 12, 14],
                                     ['AVI251_R', 14, 1, 5, 39, 13, 15],
                                     ['AVI251_L', 14, 1, 1, 40, 13, 15],
                                     ['AVI258_R', 15, 1, 5, 41, 14, 16],
                                     ['AVI258_C', 15, 1, 3, 42, 14, 16],
                                     ['AVI258_L', 15, 1, 1, 43, 14, 16],
                                     ['AVI259_R', 16, 1, 5, 44, 15, 17],
                                     ['AVI259_C', 16, 1, 3, 45, 15, 17],
                                     ['AVI259_L', 16, 1, 1, 46, 15, 17],
                                     ['AVI265_R', 17, 1, 5, 47, 16, 18],
                                     ['AVI265_L', 17, 1, 1, 48, 16, 18],
                                     ['AVI266_R', 18, 1, 5, 49, 17, 19],
                                     ['AVI266_L', 18, 1, 1, 50, 17, 19],
                                     ['AVI272_R', 19, 1, 5, 51, 18, 20],
                                     ['AVI272_L', 19, 1, 1, 52, 18, 20],
                                     ['AVI273_R', 20, 1, 5, 53, 19, 21],
                                     ['AVI273_L', 20, 1, 1, 54, 19, 21],
                                     ['AVI279_R', 21, 1, 5, 55, 20, 22],
                                     ['AVI279_L', 21, 1, 1, 56, 20, 22],
                                     ['AVI280_R', 22, 1, 5, 57, 21, 93],
                                     ['AVI280_L', 22, 1, 1, 58, 21, 93],
                                     ['AVI608_R', 23, 4, 5, 59, 24, 59],
                                     ['AVI608_C', 23, 4, 3, 60, 24, 59],
                                     ['AVI608_L', 23, 4, 1, 61, 24, 59],
                                     ['AVI609_R', 24, 4, 5, 62, 25, 23],
                                     ['AVI609_C', 24, 4, 3, 63, 25, 23],
                                     ['AVI609_L', 24, 4, 1, 64, 25, 23],
                                     ['AVI616_R', 25, 4, 5, 65, 26, 24],
                                     ['AVI616_C', 25, 4, 3, 66, 26, 24],
                                     ['AVI616_L', 25, 4, 1, 67, 26, 24],
                                     ['AVI617_R', 26, 4, 5, 68, 27, 25],
                                     ['AVI617_C', 26, 4, 3, 69, 27, 25],
                                     ['AVI617_L', 26, 4, 1, 70, 27, 25],
                                     ['AVI623_R', 27, 4, 5, 71, 28, 26],
                                     ['AVI623_C', 27, 4, 3, 72, 28, 26],
                                     ['AVI623_L', 27, 4, 1, 73, 28, 26],
                                     ['AVI624_R', 28, 4, 5, 74, 29, 27],
                                     ['AVI624_C', 28, 4, 3, 75, 29, 27],
                                     ['AVI624_L', 28, 4, 1, 76, 29, 27],
                                     ['AVI630_R', 29, 4, 5, 77, 30, 28],
                                     ['AVI630_C', 29, 4, 3, 78, 30, 28],
                                     ['AVI630_L', 29, 4, 1, 79, 30, 28],
                                     ['AVI631_R', 30, 4, 5, 80, 31, 29],
                                     ['AVI631_C', 30, 4, 3, 81, 31, 29],
                                     ['AVI631_L', 30, 4, 1, 82, 31, 29],
                                     ['AVI637_R', 31, 4, 5, 83, 32, 30],
                                     ['AVI637_IR', 31, 4, 4, 84, 32, 30],
                                     ['AVI637_IL', 31, 4, 2, 85, 32, 30],
                                     ['AVI637_L', 31, 4, 1, 86, 32, 30],
                                     ['AVI638_R', 32, 4, 5, 87, 33, 31],
                                     ['AVI638_IR', 32, 4, 4, 88, 33, 31],
                                     ['AVI638_IL', 32, 4, 2, 89, 33, 31],
                                     ['AVI638_L', 32, 4, 1, 90, 33, 31],
                                     ['AVI644_R', 33, 4, 5, 91, 34, 32],
                                     ['AVI644_L', 33, 4, 1, 92, 34, 32],
                                     ['AVI645_R', 34, 4, 5, 93, 35, 33],
                                     ['AVI645_L', 34, 4, 1, 94, 35, 33],
                                     ['AVI651_R', 35, 4, 5, 95, 36, 34],
                                     ['AVI651_L', 35, 4, 1, 96, 36, 34],
                                     ['AVI652_R', 36, 4, 5, 97, 37, 35],
                                     ['AVI652_L', 36, 4, 1, 98, 37, 35],
                                     ['AVI659_R', 37, 4, 5, 99, 39, 36],
                                     ['AVI659_L', 37, 4, 1, 100, 39, 36],
                                     ['AVI661_R', 39, 2, 5, 101, 40, 37],
                                     ['AVI661_C', 39, 2, 3, 102, 40, 37],
                                     ['AVI661_L', 39, 2, 1, 103, 40, 37],
                                     ['AVI666_R', 40, 2, 5, 104, 41, 39],
                                     ['AVI666_L', 40, 2, 1, 105, 41, 39],
                                     ['AVI667_R', 41, 2, 5, 106, 42, 40],
                                     ['AVI667_L', 41, 2, 1, 107, 42, 40],
                                     ['AVI674_R', 42, 2, 5, 108, 43, 41],
                                     ['AVI674_L', 42, 2, 1, 109, 43, 41],
                                     ['AVI675_R', 43, 2, 5, 110, 44, 42],
                                     ['AVI675_L', 43, 2, 1, 111, 44, 42],
                                     ['AVI682_R', 44, 2, 5, 112, 45, 43],
                                     ['AVI682_L', 44, 2, 1, 113, 45, 43],
                                     ['AVI683_R', 45, 2, 5, 114, 46, 44],
                                     ['AVI683_L', 45, 2, 1, 115, 46, 44],
                                     ['AVI687_R', 46, 2, 5, 116, 47, 45],
                                     ['AVI687_C', 46, 2, 3, 117, 47, 45],
                                     ['AVI687_L', 46, 2, 1, 118, 47, 45],
                                     ['AVI072_R', 47, 2, 5, 119, 48, 46],
                                     ['AVI072_L', 47, 2, 1, 120, 48, 46],
                                     ['TVL041_R', 48, 2, 5, 121, 0, 47],
                                     ['TVL043_L', 48, 2, 1, 122, 0, 47],
                                     ['TVL045_S', 49, 2, 6, 123, 0, 48],
                                     ['TVL047_S', 50, 2, 6, 124, 0, 48],
                                     ['AVIO28_S', 51, 2, 6, 125, 39, 52],
                                     ['AVIO27_S', 52, 2, 6, 126, 51, 53],
                                     ['AVIO22_R', 53, 2, 5, 127, 52, 54],
                                     ['AVIO22_L', 53, 2, 1, 128, 52, 54],
                                     ['AVIO21_R', 54, 2, 5, 129, 53, 55],
                                     ['AVIO21_L', 54, 2, 1, 130, 53, 55],
                                     ['TVL019_S', 38, 3, 6, 131, 54, 0],
                                     ['TVL011_S', 55, 3, 6, 132, 54, 56],
                                     ['TVL013_S', 56, 3, 6, 133, 55, 0],
                                     ['TVLV03_S', 58, 4, 6, 134, 23, 0],
                                     ['TVLV01_S', 59, 4, 6, 135, 23, 0],
                                     ['TVLA15_S', 60, 4, 6, 136, 23, 0],
                                     ['AVIA22_R', 61, 3, 5, 137, 0, 97],
                                     ['AVIA22_L', 61, 3, 1, 138, 0, 97],
                                     ['TVLA03_R', 97, 3, 5, 139, 61, 1],
                                     ['TVLA01_L', 97, 3, 1, 140, 61, 1],
                                     ['TVLA09_R', 62, 3, 5, 141, 0, 1],
                                     ['TVLA07_L', 62, 3, 1, 142, 0, 1],
                                     ['AVIA01_R', 63, 3, 5, 143, 0, 64],
                                     ['AVIA01_L', 63, 3, 1, 144, 0, 64],
                                     ['TVLA05_S', 64, 3, 6, 145, 63, 1],
                                     ['AVIB40_R', 65, 3, 5, 146, 66, 10],
                                     ['AVIB40_L', 65, 3, 1, 147, 66, 10],
                                     ['AVIB41_R', 66, 3, 5, 148, 67, 65],
                                     ['AVIB41_L', 66, 3, 1, 149, 67, 65],
                                     ['AVIB46_R', 67, 3, 5, 150, 68, 66],
                                     ['AVIB46_L', 67, 3, 1, 151, 68, 66],
                                     ['AVIB47_R', 68, 3, 5, 152, 57, 67],
                                     ['AVIB47_L', 68, 3, 1, 153, 57, 67],
                                     ['TVL017_R', 57, 3, 5, 154, 68, 0],
                                     ['TVL015_L', 57, 3, 1, 155, 68, 0],
                                     ['TVLR21_R', 69, 4, 1, 156, 0, 75],
                                     ['AVI027_R', 70, 4, 5, 157, 0, 71],
                                     ['AVI027_C', 70, 4, 3, 158, 0, 71],
                                     ['AVI027_L', 70, 4, 1, 159, 0, 71],
                                     ['AVIQ21_R', 71, 4, 5, 160, 70, 72],
                                     ['AVIQ21_L', 71, 4, 1, 161, 70, 72],
                                     ['AVIQ22_R', 72, 4, 5, 162, 71, 73],
                                     ['AVIQ22_C', 72, 4, 3, 163, 71, 73],
                                     ['AVIQ22_L', 72, 4, 1, 164, 71, 73],
                                     ['TVLQ23_L', 73, 4, 1, 165, 72, 74],
                                     ['TVLR23_R', 74, 4, 5, 166, 73, 75],
                                     ['AVIR28_R', 75, 4, 5, 167, 74, 76],
                                     ['AVIR28_C', 75, 4, 3, 168, 74, 76],
                                     ['AVIR28_L', 75, 4, 1, 169, 74, 76],
                                     ['AVIR29_R', 76, 4, 5, 170, 75, 77],
                                     ['AVIR29_C', 76, 4, 3, 171, 75, 77],
                                     ['AVIR29_L', 76, 4, 1, 172, 75, 77],
                                     ['AVIR35_R', 77, 4, 5, 173, 76, 78],
                                     ['AVIR35_L', 77, 4, 1, 174, 76, 78],
                                     ['AVIR36_R', 78, 4, 5, 175, 77, 32],
                                     ['AVIR36_L', 78, 4, 1, 176, 77, 32],
                                     ['AVI053_R', 79, 4, 5, 177, 0, 80],
                                     ['AVI053_L', 79, 4, 1, 178, 0, 80],
                                     ['TVLQ21_S', 80, 1, 6, 179, 79, 81],
                                     ['AVID20_S', 81, 1, 6, 180, 80, 82],
                                     ['AVID21_S', 82, 1, 6, 181, 81, 83],
                                     ['AVIE27_R', 83, 1, 5, 182, 82, 84],
                                     ['AVIE27_L', 83, 1, 1, 183, 82, 84],
                                     ['AVIC31_R', 84, 1, 5, 184, 83, 85],
                                     ['AVIC31_L', 84, 1, 1, 185, 83, 85],
                                     ['AVIC32_S', 85, 1, 6, 186, 84, 15],
                                     ['TVLE23_S', 86, 1, 6, 187, 0, 83],
                                     ['TVL023_R', 87, 1, 5, 188, 0, 88],
                                     ['TVL021_L', 87, 1, 1, 189, 0, 88],
                                     ['AVIC19_S', 88, 1, 6, 190, 87, 89],
                                     ['AVIC20_S', 89, 1, 6, 191, 88, 90],
                                     ['AVIC25_S', 90, 1, 6, 192, 89, 91],
                                     ['AVIC26_S', 91, 1, 6, 193, 90, 83],
                                     ['TVL035_R', 92, 5, 6, 194, 22, 0],
                                     ['TVL033_C', 92, 3, 6, 195, 22, 0],
                                     ['TVL031_L', 92, 1, 6, 196, 22, 0],
                                     ['TVL039_R', 93, 1, 5, 197, 22, 0],
                                     ['TVL037_L', 93, 1, 1, 198, 22, 0],
                                     ['AVIV18_S', 94, 4, 6, 199, 25, 95],
                                     ['TVLV05_S', 95, 4, 6, 200, 94, 96],
                                     ['AVIV17_S', 96, 4, 6, 201, 95, 0]]
                                    , columns=['sub_section_name', 'section', 'direction', 'lane', 'sub_section',
                                               'prev_section', 'next_section'])
road_topology_df_APL.set_index(['sub_section_name'], inplace=True)

topology_dictionary_LW = {'LIU1001_01': 'Lane 1 Entry EB Subsection 1',
                          'LIU1001_02': 'Lane 2 Entry EB Subsection 2',
                          'LIU1001_03': 'Lane 1 Exit WB Subsection 7',
                          'LIU1001_04': 'Lane 2 Exit WB Subsection 8',
                          'LIU2001_01': 'Lane 1 Entry WB Subsection 5',
                          'LIU2001_02': 'Lane 2 Entry WB Subsection 6',
                          'LIU2001_03': 'Lane 1 Exit EB Subsection 3',
                          'LIU2001_04': 'Lane 2 Exit EB Subsection 2'}

# segmetns like lanes
segments_dict_LW = {1: 'Lane:SL Direction:EB (Entry)',
                    2: 'Lane:FL Direction:EB (Entry)',
                    3: 'Lane:SL Direction:EB (Exit)',
                    4: 'Lane:FL Direction:EB (Exit)',
                    5: 'Lane:SL Direction:WB (Entry)',
                    6: 'Lane:FL Direction:WB (Entry)',
                    7: 'Lane:SL Direction:WB (Exit)',
                    8: 'Lane:FL Direction:WB (Exit)'}

sections_dict_LW = {1: 'Entry-EB-CS1001',
                    2: 'Exit-EB-CS1002',
                    3: 'Entry-WB-CS2001',
                    4: 'Exit-WB-CS2002'}  #

df_aux = pd.DataFrame.from_dict(sections_dict_LW, orient='index')
df_aux.reset_index(inplace=True)
df_aux.rename(columns={'index': 'segment', 0: 'desc'}, inplace=True)
#df_aux.to_csv('data\\LW_sections_dict.csv', index=False)

#sections_dict = pd.read_csv('data\\LW_sections_dict.csv').set_index(['segment'], drop=True).to_dict()['desc']

#sections_dict_inv = pd.read_csv('data\\LW_sections_dict.csv').set_index(['desc'], drop=True).to_dict()['segment']

original_columns_rfa = ['log_time',
                        'detector_id',
                        'detector_id-name',
                        'occupancy',
                        'norm_occupancy',
                        'volume',
                        'norm_volume',
                        'speed_obs',
                        'speed_sum',
                        'configuration_id',
                        'configuration_id-name',
                        'available',
                        'incident',
                        'failed',
                        'length_normalised',
                        'distance_normalised',
                        'last_modified']




def check_values_variables(df):
    drop_ = False
    sample_ = False

    index_1 = df[(df['volume'] == 0) & (df['speed'] != 0)].index
    index_2 = df[(df['volume'] == 0) & (df['occupancy'] != 0)].index
    index_3 = df[(df['volume'] == 0) & (df['flow'] != 0)].index

    index_4 = df[(df['flow'] == 0) & (df['speed'] != 0)].index
    index_5 = df[(df['flow'] == 0) & (df['occupancy'] != 0)].index
    index_6 = df[(df['flow'] == 0) & (df['volume'] != 0)].index

    index_7 = df[(df['occupancy'] == 0) & (df['volume'] != 0)].index
    index_8 = df[(df['occupancy'] == 0) & (df['speed'] != 0)].index
    index_9 = df[(df['occupancy'] == 0) & (df['flow'] != 0)].index

    index_10 = df[(df['speed'] == 0) & (df['occupancy'] != 0)].index
    index_11 = df[(df['speed'] == 0) & (df['flow'] != 0)].index
    index_12 = df[(df['speed'] == 0) & (df['volume'] != 0)].index

    # index_13 = df[(df['meanlength'] == 0) & (df['flow'] != 0)].index
    # index_14 = df[(df['meanlength'] == 0) & (df['speed'] != 0)].index
    # index_15 = df[(df['meanlength'] == 0) & (df['volume'] != 0)].index

    index_16 = df[(df['volume'] < 0)].index
    index_17 = df[(df['speed'] < 0)].index
    index_18 = df[(df['flow'] < 0)].index
    # index_19 = df[(df['meanlength'] < 0)].index

    print('\n')
    print('{} Check Values'.format('-' * 10))
    print('\n')
    print('#1 (volume=0 & speed!=0):{}{}'.format(' ' * 8, len(index_1)))
    print('#2 (volume=0 & occupancy!=0):{}{}'.format(' ' * 4, len(index_2)))
    print('#3 (volume=0 & flow!=0):{}{}'.format(' ' * 9, len(index_3)))
    print('#4 (flow=0 & speed!=0):{}{}'.format(' ' * 10, len(index_4)))
    print('#5 (flow=0 & occupancy!=0):{}{}'.format(' ' * 6, len(index_5)))
    print('#6 (flow=0 & volume!=0):{}{}'.format(' ' * 9, len(index_6)))
    print('#7 (occupancy=0 & volume!=0):{}{}'.format(' ' * 4, len(index_7)))
    print('#8 (occupancy=0 & speed!=0):{}{}'.format(' ' * 5, len(index_8)))
    print('#9 (occupancy=0 & flow!=0):{}{}'.format(' ' * 6, len(index_9)))
    print('#10 (speed=0 & occupancy!=0):{}{}'.format(' ' * 4, len(index_10)))
    print('#11 (speed=0 & flow!=0):{}{}'.format(' ' * 9, len(index_11)))
    print('#12 (speed=0 & volume!=0):{}{}'.format(' ' * 7, len(index_12)))
    # print('#13 (meanlength=0 & flow!=0):{}{}'.format(' ' * 4, len(index_13)))
    # print('#14 (meanlength=0 & speed!=0):{}{}'.format(' ' * 3, len(index_14)))
    # print('#15 (meanlength=0 & volume!=0):{}{}'.format(' ' * 2, len(index_15)))
    print('#16 (volume<0):{}{}'.format(' ' * 18, len(index_16)))
    print('#17 (speed<0):{}{}'.format(' ' * 19, len(index_17)))
    print('#18 (flow<0):{}{}'.format(' ' * 20, len(index_18)))
    # print('#19 (meanlength<0):{}{}'.format(' ' * 20, len(index_19)))
    print('\n')


dict_lanes = {1: 'Left/Slow Lane',
              2: 'Right/fast Lane',
              3: 'Central Lane'}


dict_days = {'Monday': 0,
             'Tuesday': 1,
             'Wednesday': 2,
             'Thurday': 3,
             'Friday': 4,
             'Saturday': 5,
             'Sunday': 6}

days_dict = {0: 'Monday',
             1: 'Tuesday',
             2: 'Wednesday',
             3: 'Thurday',
             4: 'Friday',
             5: 'Saturday',
             6: 'Sunday'}

direction_dict = {'Eastbound': 1,
                  'Westbound': 2,
                  'Northbound': 3,
                  'Southbound': 4}

dict_direction = {1: 'EB',
                  2: 'WB',
                  3: 'NB',
                  4: 'SB'}




dict_season = {1: 'summer',
               2: 'Autumn',
               3: 'Winter',
               4: 'Spring'}



algorithms_ = ['tbats',
               'ets',
               'sarima',
               'fbprophet',
               'desc_ref',
               'extra_trees_2',
               'random_forest',
               'mlp_neural_network',
               'gradient_boosting']

algorithms = {2: 'TBATS (Time series)',  # time series
              3: 'ETS (Time series)',  # time series
              4: 'SARIMA (Time series)',  # time series
              5: 'fbprophet (Time series)',  # time series
              6: 'Descriptive',  # Descriptive
              8: 'Extra trees (Ensemble)',  # Extra_trees_2
              10: 'Random Forest (Ensemble)',  # Ensemble
              11: 'Neural Network',  # Neural Network
              12: 'Gradient Boosting (Ensemble)'}  # Ensemble

horizon_dict = {'PREDICCION_INT 15': 'Horizon 15 min',
                'PREDICCION_INT 60': 'Horizon 60 min',
                'PREDICCION_INT 120': 'Horizon 120 min'}





lanes_dictionary = {'SL': 1,
                    'FL': 2,
                    'CL': 3,
                    'IL': 4,
                    'IR': 5}

lanes_dictionary_inv = {1: 'SL',
                        2: 'FL',
                        3: 'CL',
                        4: 'IL',
                        5: 'IR'}

road1 = """
    |       |      
    |       |       
    |       | 
    |       |       
    |       |       
"""
road2 = """
    |       |       |
    |       |       |
    |       | (70)  |
    |       |       |
    |       |       |
"""
road3 = """
    |       |       |       |
    |       |       |       |   
    |       |       | (70)  |
    |       |       |       |
    |       |       |       |
"""
road4 = """
    |       |       |       |       |                
    |       |       |       |       |
    |       |       |       | (70)  |
    |       |       |       |       |
    |       |       |       |       |
"""
road5 = """
    |       |       |       |       |       |                
    |       |       |       |       |       |
    |       |       |       |       | (70)  |
    |       |       |       |       |       |
    |       |       |       |       |       | 
"""

legend_lanes = """
{} lanes legend 
{} (SL)-Slow Lane/Left Lane
{} (FL)-Fast Lane/Right Lane
{} (CL)-Central Lane
{} (IL)-Interior Left
{} (IR)-Interior Right
""".format('-' * 20, ' ' * 20, ' ' * 20, ' ' * 20, ' ' * 20, ' ' * 20)





# Path with raw data
read_path = 'D:/CLARENCE/APL_TraffZoneDef/TraffZoneDef/'

# Path with resampled data
write_path = 'D:/CLARENCE/APL_TraffZoneDef_agrupados_ponderados/'
data_path = write_path

images_path = 'D:/CLARENCE/figures/'

data_path_analytics = 'D:\\dataanalytics\\dataanalytics.predictive\\data\\'

models_path = 'D:/CLARENCE/models/APL'

extension = 'csv'

road = 'Airport Link'

original_columns = ['Name', 'Last_Update', 'Health_State', 'Availability_',
                    'Count_', 'Average_Occupancy', 'Average_Headway',
                    'Average_Length', 'Average_Speed', 'Count_Short',
                    'Count_Medium', 'Count_Long', 'Average_Speed_Short',
                    'Average_Speed_Medium', 'Average_Speed_Long']

columns_ = ['volume', 'occupancy',
            'headway', 'meanlength',
            'speed']

all_columns_ = ['time', 'volume', 'occupancy',
                'headway', 'meanlength',
                'speed',
                'name']



# LL = Left Lane, IL = Inner Left, CL = Central Lane, IR = Inner Right, RL = Right Lane, SL = Single Lane
lane_dict = {'_L': 1, 'IL': 2, '_C': 3, 'IR': 4, '_R': 5, '_S': 6}
direction_dict = {'Eastbound': 1, 'Westbound': 2, 'Northbound': 3, 'Southbound': 4}

road_topology_df = pd.DataFrame([['AVI203_R', 1, 3, 5, 1, 62, 2],
                                 ['AVI203_IR', 1, 3, 4, 2, 62, 2],
                                 ['AVI203_IL', 1, 3, 2, 3, 62, 2],
                                 ['AVI203_L', 1, 3, 1, 4, 62, 2],
                                 ['AVI204_R', 2, 3, 5, 5, 1, 3],
                                 ['AVI204_IR', 2, 3, 4, 6, 1, 3],
                                 ['AVI204_IL', 2, 3, 2, 7, 1, 3],
                                 ['AVI204_L', 2, 3, 1, 8, 1, 3],
                                 ['AVI210_R', 3, 3, 5, 9, 2, 4],
                                 ['AVI210_C', 3, 3, 3, 10, 2, 4],
                                 ['AVI210_L', 3, 3, 1, 11, 2, 4],
                                 ['AVI211_R', 4, 3, 5, 12, 3, 5],
                                 ['AVI211_C', 4, 3, 3, 13, 3, 5],
                                 ['AVI211_L', 4, 3, 1, 14, 3, 5],
                                 ['AVI219_R', 5, 3, 5, 15, 4, 6],
                                 ['AVI219_C', 5, 3, 3, 16, 4, 6],
                                 ['AVI219_L', 5, 3, 1, 17, 4, 6],
                                 ['AVI220_R', 6, 3, 5, 18, 5, 7],
                                 ['AVI220_C', 6, 3, 3, 19, 5, 7],
                                 ['AVI220_L', 6, 3, 1, 20, 5, 7],
                                 ['AVI227_R', 7, 3, 5, 21, 6, 8],
                                 ['AVI227_C', 7, 3, 3, 22, 6, 8],
                                 ['AVI227_L', 7, 3, 1, 23, 6, 8],
                                 ['AVI228_R', 8, 3, 5, 24, 7, 9],
                                 ['AVI228_C', 8, 3, 3, 25, 7, 9],
                                 ['AVI228_L', 8, 3, 1, 26, 7, 9],
                                 ['AVI235_R', 9, 3, 5, 27, 8, 10],
                                 ['AVI235_C', 9, 3, 3, 28, 8, 10],
                                 ['AVI235_L', 9, 3, 1, 29, 8, 10],
                                 ['AVI236_R', 10, 3, 5, 30, 9, 11],
                                 ['AVI236_C', 10, 3, 3, 31, 9, 11],
                                 ['AVI236_L', 10, 3, 1, 32, 9, 11],
                                 ['AVI243_R', 11, 1, 5, 33, 10, 12],
                                 ['AVI243_L', 11, 1, 1, 34, 10, 12],
                                 ['AVI244_R', 12, 1, 5, 35, 11, 13],
                                 ['AVI244_L', 12, 1, 1, 36, 11, 13],
                                 ['AVI250_R', 13, 1, 5, 37, 12, 14],
                                 ['AVI250_L', 13, 1, 1, 38, 12, 14],
                                 ['AVI251_R', 14, 1, 5, 39, 13, 15],
                                 ['AVI251_L', 14, 1, 1, 40, 13, 15],
                                 ['AVI258_R', 15, 1, 5, 41, 14, 16],
                                 ['AVI258_C', 15, 1, 3, 42, 14, 16],
                                 ['AVI258_L', 15, 1, 1, 43, 14, 16],
                                 ['AVI259_R', 16, 1, 5, 44, 15, 17],
                                 ['AVI259_C', 16, 1, 3, 45, 15, 17],
                                 ['AVI259_L', 16, 1, 1, 46, 15, 17],
                                 ['AVI265_R', 17, 1, 5, 47, 16, 18],
                                 ['AVI265_L', 17, 1, 1, 48, 16, 18],
                                 ['AVI266_R', 18, 1, 5, 49, 17, 19],
                                 ['AVI266_L', 18, 1, 1, 50, 17, 19],
                                 ['AVI272_R', 19, 1, 5, 51, 18, 20],
                                 ['AVI272_L', 19, 1, 1, 52, 18, 20],
                                 ['AVI273_R', 20, 1, 5, 53, 19, 21],
                                 ['AVI273_L', 20, 1, 1, 54, 19, 21],
                                 ['AVI279_R', 21, 1, 5, 55, 20, 22],
                                 ['AVI279_L', 21, 1, 1, 56, 20, 22],
                                 ['AVI280_R', 22, 1, 5, 57, 21, 93],
                                 ['AVI280_L', 22, 1, 1, 58, 21, 93],
                                 ['AVI608_R', 23, 4, 5, 59, 24, 59],
                                 ['AVI608_C', 23, 4, 3, 60, 24, 59],
                                 ['AVI608_L', 23, 4, 1, 61, 24, 59],
                                 ['AVI609_R', 24, 4, 5, 62, 25, 23],
                                 ['AVI609_C', 24, 4, 3, 63, 25, 23],
                                 ['AVI609_L', 24, 4, 1, 64, 25, 23],
                                 ['AVI616_R', 25, 4, 5, 65, 26, 24],
                                 ['AVI616_C', 25, 4, 3, 66, 26, 24],
                                 ['AVI616_L', 25, 4, 1, 67, 26, 24],
                                 ['AVI617_R', 26, 4, 5, 68, 27, 25],
                                 ['AVI617_C', 26, 4, 3, 69, 27, 25],
                                 ['AVI617_L', 26, 4, 1, 70, 27, 25],
                                 ['AVI623_R', 27, 4, 5, 71, 28, 26],
                                 ['AVI623_C', 27, 4, 3, 72, 28, 26],
                                 ['AVI623_L', 27, 4, 1, 73, 28, 26],
                                 ['AVI624_R', 28, 4, 5, 74, 29, 27],
                                 ['AVI624_C', 28, 4, 3, 75, 29, 27],
                                 ['AVI624_L', 28, 4, 1, 76, 29, 27],
                                 ['AVI630_R', 29, 4, 5, 77, 30, 28],
                                 ['AVI630_C', 29, 4, 3, 78, 30, 28],
                                 ['AVI630_L', 29, 4, 1, 79, 30, 28],
                                 ['AVI631_R', 30, 4, 5, 80, 31, 29],
                                 ['AVI631_C', 30, 4, 3, 81, 31, 29],
                                 ['AVI631_L', 30, 4, 1, 82, 31, 29],
                                 ['AVI637_R', 31, 4, 5, 83, 32, 30],
                                 ['AVI637_IR', 31, 4, 4, 84, 32, 30],
                                 ['AVI637_IL', 31, 4, 2, 85, 32, 30],
                                 ['AVI637_L', 31, 4, 1, 86, 32, 30],
                                 ['AVI638_R', 32, 4, 5, 87, 33, 31],
                                 ['AVI638_IR', 32, 4, 4, 88, 33, 31],
                                 ['AVI638_IL', 32, 4, 2, 89, 33, 31],
                                 ['AVI638_L', 32, 4, 1, 90, 33, 31],
                                 ['AVI644_R', 33, 4, 5, 91, 34, 32],
                                 ['AVI644_L', 33, 4, 1, 92, 34, 32],
                                 ['AVI645_R', 34, 4, 5, 93, 35, 33],
                                 ['AVI645_L', 34, 4, 1, 94, 35, 33],
                                 ['AVI651_R', 35, 4, 5, 95, 36, 34],
                                 ['AVI651_L', 35, 4, 1, 96, 36, 34],
                                 ['AVI652_R', 36, 4, 5, 97, 37, 35],
                                 ['AVI652_L', 36, 4, 1, 98, 37, 35],
                                 ['AVI659_R', 37, 4, 5, 99, 39, 36],
                                 ['AVI659_L', 37, 4, 1, 100, 39, 36],
                                 ['AVI661_R', 39, 2, 5, 101, 40, 37],
                                 ['AVI661_C', 39, 2, 3, 102, 40, 37],
                                 ['AVI661_L', 39, 2, 1, 103, 40, 37],
                                 ['AVI666_R', 40, 2, 5, 104, 41, 39],
                                 ['AVI666_L', 40, 2, 1, 105, 41, 39],
                                 ['AVI667_R', 41, 2, 5, 106, 42, 40],
                                 ['AVI667_L', 41, 2, 1, 107, 42, 40],
                                 ['AVI674_R', 42, 2, 5, 108, 43, 41],
                                 ['AVI674_L', 42, 2, 1, 109, 43, 41],
                                 ['AVI675_R', 43, 2, 5, 110, 44, 42],
                                 ['AVI675_L', 43, 2, 1, 111, 44, 42],
                                 ['AVI682_R', 44, 2, 5, 112, 45, 43],
                                 ['AVI682_L', 44, 2, 1, 113, 45, 43],
                                 ['AVI683_R', 45, 2, 5, 114, 46, 44],
                                 ['AVI683_L', 45, 2, 1, 115, 46, 44],
                                 ['AVI687_R', 46, 2, 5, 116, 47, 45],
                                 ['AVI687_C', 46, 2, 3, 117, 47, 45],
                                 ['AVI687_L', 46, 2, 1, 118, 47, 45],
                                 ['AVI072_R', 47, 2, 5, 119, 48, 46],
                                 ['AVI072_L', 47, 2, 1, 120, 48, 46],
                                 ['TVL041_R', 48, 2, 5, 121, 0, 47],
                                 ['TVL043_L', 48, 2, 1, 122, 0, 47],
                                 ['TVL045_S', 49, 2, 6, 123, 0, 48],
                                 ['TVL047_S', 50, 2, 6, 124, 0, 48],
                                 ['AVIO28_S', 51, 2, 6, 125, 39, 52],
                                 ['AVIO27_S', 52, 2, 6, 126, 51, 53],
                                 ['AVIO22_R', 53, 2, 5, 127, 52, 54],
                                 ['AVIO22_L', 53, 2, 1, 128, 52, 54],
                                 ['AVIO21_R', 54, 2, 5, 129, 53, 55],
                                 ['AVIO21_L', 54, 2, 1, 130, 53, 55],
                                 ['TVL019_S', 38, 3, 6, 131, 54, 0],
                                 ['TVL011_S', 55, 3, 6, 132, 54, 56],
                                 ['TVL013_S', 56, 3, 6, 133, 55, 0],
                                 ['TVLV03_S', 58, 4, 6, 134, 23, 0],
                                 ['TVLV01_S', 59, 4, 6, 135, 23, 0],
                                 ['TVLA15_S', 60, 4, 6, 136, 23, 0],
                                 ['AVIA22_R', 61, 3, 5, 137, 0, 97],
                                 ['AVIA22_L', 61, 3, 1, 138, 0, 97],
                                 ['TVLA03_R', 97, 3, 5, 139, 61, 1],
                                 ['TVLA01_L', 97, 3, 1, 140, 61, 1],
                                 ['TVLA09_R', 62, 3, 5, 141, 0, 1],
                                 ['TVLA07_L', 62, 3, 1, 142, 0, 1],
                                 ['AVIA01_R', 63, 3, 5, 143, 0, 64],
                                 ['AVIA01_L', 63, 3, 1, 144, 0, 64],
                                 ['TVLA05_S', 64, 3, 6, 145, 63, 1],
                                 ['AVIB40_R', 65, 3, 5, 146, 66, 10],
                                 ['AVIB40_L', 65, 3, 1, 147, 66, 10],
                                 ['AVIB41_R', 66, 3, 5, 148, 67, 65],
                                 ['AVIB41_L', 66, 3, 1, 149, 67, 65],
                                 ['AVIB46_R', 67, 3, 5, 150, 68, 66],
                                 ['AVIB46_L', 67, 3, 1, 151, 68, 66],
                                 ['AVIB47_R', 68, 3, 5, 152, 57, 67],
                                 ['AVIB47_L', 68, 3, 1, 153, 57, 67],
                                 ['TVL017_R', 57, 3, 5, 154, 68, 0],
                                 ['TVL015_L', 57, 3, 1, 155, 68, 0],
                                 ['TVLR21_R', 69, 4, 1, 156, 0, 75],
                                 ['AVI027_R', 70, 4, 5, 157, 0, 71],
                                 ['AVI027_C', 70, 4, 3, 158, 0, 71],
                                 ['AVI027_L', 70, 4, 1, 159, 0, 71],
                                 ['AVIQ21_R', 71, 4, 5, 160, 70, 72],
                                 ['AVIQ21_L', 71, 4, 1, 161, 70, 72],
                                 ['AVIQ22_R', 72, 4, 5, 162, 71, 73],
                                 ['AVIQ22_C', 72, 4, 3, 163, 71, 73],
                                 ['AVIQ22_L', 72, 4, 1, 164, 71, 73],
                                 ['TVLQ23_L', 73, 4, 1, 165, 72, 74],
                                 ['TVLR23_R', 74, 4, 5, 166, 73, 75],
                                 ['AVIR28_R', 75, 4, 5, 167, 74, 76],
                                 ['AVIR28_C', 75, 4, 3, 168, 74, 76],
                                 ['AVIR28_L', 75, 4, 1, 169, 74, 76],
                                 ['AVIR29_R', 76, 4, 5, 170, 75, 77],
                                 ['AVIR29_C', 76, 4, 3, 171, 75, 77],
                                 ['AVIR29_L', 76, 4, 1, 172, 75, 77],
                                 ['AVIR35_R', 77, 4, 5, 173, 76, 78],
                                 ['AVIR35_L', 77, 4, 1, 174, 76, 78],
                                 ['AVIR36_R', 78, 4, 5, 175, 77, 32],
                                 ['AVIR36_L', 78, 4, 1, 176, 77, 32],
                                 ['AVI053_R', 79, 4, 5, 177, 0, 80],
                                 ['AVI053_L', 79, 4, 1, 178, 0, 80],
                                 ['TVLQ21_S', 80, 1, 6, 179, 79, 81],
                                 ['AVID20_S', 81, 1, 6, 180, 80, 82],
                                 ['AVID21_S', 82, 1, 6, 181, 81, 83],
                                 ['AVIE27_R', 83, 1, 5, 182, 82, 84],
                                 ['AVIE27_L', 83, 1, 1, 183, 82, 84],
                                 ['AVIC31_R', 84, 1, 5, 184, 83, 85],
                                 ['AVIC31_L', 84, 1, 1, 185, 83, 85],
                                 ['AVIC32_S', 85, 1, 6, 186, 84, 15],
                                 ['TVLE23_S', 86, 1, 6, 187, 0, 83],
                                 ['TVL023_R', 87, 1, 5, 188, 0, 88],
                                 ['TVL021_L', 87, 1, 1, 189, 0, 88],
                                 ['AVIC19_S', 88, 1, 6, 190, 87, 89],
                                 ['AVIC20_S', 89, 1, 6, 191, 88, 90],
                                 ['AVIC25_S', 90, 1, 6, 192, 89, 91],
                                 ['AVIC26_S', 91, 1, 6, 193, 90, 83],
                                 ['TVL035_R', 92, 5, 6, 194, 22, 0],
                                 ['TVL033_C', 92, 3, 6, 195, 22, 0],
                                 ['TVL031_L', 92, 1, 6, 196, 22, 0],
                                 ['TVL039_R', 93, 1, 5, 197, 22, 0],
                                 ['TVL037_L', 93, 1, 1, 198, 22, 0],
                                 ['AVIV18_S', 94, 4, 6, 199, 25, 95],
                                 ['TVLV05_S', 95, 4, 6, 200, 94, 96],
                                 ['AVIV17_S', 96, 4, 6, 201, 95, 0]]
                                , columns=['sub_section_name', 'section', 'direction', 'lanes', 'sub_section',
                                           'prev_section', 'next_section'])
road_topology_df.set_index(['sub_section_name'], inplace=True)

holyday_list = [['New Years Day', '2017-01-01', 1],
                ['Australia Day', '2017-01-26', 2],
                ['Good Friday', '2017-04-14', 3],
                ['The day after Good Friday', '2017-04-15', 4],
                ['Easter Sunday', '2017-04-16', 5],
                ['Easter Monday', '2017-04-17', 6],
                ['Aznac Day', '2017-04-25', 7],
                ['Labour Day', '2017-05-01', 8],
                ['Royal Queensland Show', '2017-08-16', 9],
                ['Queens Birthday', '2017-10-2', 10],
                ['Christmas Eve', '2017-12-24', 11],
                ['Christmas Day', '2017-12-25', 12],
                ['Boxing Day', '2017-12-26', 13],
                ['New Years Day', '2018-01-01', 1],
                ['Australia Day', '2018-01-26', 2],
                ['Good Friday', '2018-03-30', 3],
                ['The day after Good Friday', '2018-03-31', 4],
                ['Easter Sunday', '2018-04-01', 5],
                ['Easter Monday', '2018-04-02', 6],
                ['Aznac Day', '2018-04-25', 7],
                ['Labour Day', '2018-05-07', 8],
                ['Royal Queensland Show', '2018-08-15', 9],
                ['Queens Birthday', '2018-10-1', 10],
                ['Christmas Eve', '2018-12-24', 11],
                ['Christmas Day', '2018-12-25', 12],
                ['Boxing Day', '2018-12-26', 13],
                ['New Years Day', '2019-01-01', 1],
                ['Australia Day', '2019-01-28', 2],
                ['Good Friday', '2019-04-19', 3],
                ['The day after Good Friday', '2019-04-20', 4],
                ['Easter Sunday', '2019-04-21', 5],
                ['Easter Monday', '2019-04-22', 6],
                ['Aznac Day', '2019-04-25', 7],
                ['Labour Day', '2019-05-06', 8],
                ['Royal Queensland Show', '2019-08-14', 9],
                ['Queens Birthday', '2019-10-7', 10],
                ['Christmas Eve', '2019-12-24', 11],
                ['Christmas Day', '2019-12-25', 12],
                ['Boxing Day', '2019-12-26', 13]]
holyday_df = pd.DataFrame(holyday_list, columns=['holyday_name', 'holyday_date', 'holyday_id'])
holyday_df['holyday_date'] = pd.to_datetime(holyday_df['holyday_date']).dt.date
holyday_df.set_index(['holyday_date'], inplace=True)

days_dict = {0: 'Monday',
             1: 'Tuesday',
             2: 'Wednesday',
             3: 'Thurday',
             4: 'Friday',
             5: 'Saturday',
             6: 'Sunday'}


def config_plotly():
    import plotly.io as pio
    pio.renderers.default = "browser"


def config_pandas():
    """
    Allows to show all the columns of a dataframe in the console
    Limit pandas warnings
    """
    import pandas as pd
    import numpy as np
    pd.options.mode.chained_assignment = None  # default='warn'
    desired_width = 350
    np.set_printoptions(linewidth=desired_width)  # show dataframes in console
    pd.set_option('display.max_columns', 10)






extension = 'csv'

road = 'Clem 7'

columns_ = ['Volume', 'Occupancy',
            'Headway', 'Meanlength',
            'Speed']

all_columns_ = ['time', 'Volume', 'Occupancy',
                'Headway', 'Meanlength',
                'Speed',
                'Name']



original_columns = ['Name', 'Last_Update', 'Health_State', 'Availability_',
                    'Count_', 'Average_Occupancy', 'Average_Headway',
                    'Average_Length', 'Average_Speed', 'Count_Short',
                    'Count_Medium', 'Count_Long', 'Average_Speed_Short',
                    'Average_Speed_Medium', 'Average_Speed_Long']

lane_dict = {'_L': 1, 'FL': 2, 'SL': 3}  # FL = Fast Lane, SL = Slow Lane, _L = Only Lane
direction_dict = {'Eastbound': 1, 'Westbound': 2, 'Northbound': 3, 'Southbound': 4}

# Encode sections
road_topology_df = pd.DataFrame([['AVI101_FL', 1, 3, 2, 1, 0, 2],
                                 ['AVI101_SL', 1, 3, 3, 2, 0, 2],
                                 ['AVI102_FL', 2, 3, 2, 3, 1, 3],
                                 ['AVI102_SL', 2, 3, 3, 4, 1, 3],
                                 ['AVI110_FL', 3, 3, 2, 5, 2, 4],
                                 ['AVI110_SL', 3, 3, 3, 6, 2, 4],
                                 ['AVI111_FL', 4, 3, 2, 7, 3, 5],
                                 ['AVI111_SL', 4, 3, 3, 8, 3, 5],
                                 ['AVI117_FL', 5, 3, 2, 9, 4, 6],
                                 ['AVI117_SL', 5, 3, 3, 10, 4, 6],
                                 ['AVI118_FL', 6, 3, 2, 11, 5, 7],
                                 ['AVI118_SL', 6, 3, 3, 12, 5, 7],
                                 ['AVI125_FL', 7, 3, 2, 13, 6, 8],
                                 ['AVI125_SL', 7, 3, 3, 14, 6, 8],
                                 ['AVI126_FL', 8, 3, 2, 15, 7, 9],
                                 ['AVI126_SL', 8, 3, 3, 16, 7, 9],
                                 ['AVI133_FL', 9, 3, 2, 17, 8, 10],
                                 ['AVI133_SL', 9, 3, 3, 18, 8, 10],
                                 ['AVI134_FL', 10, 3, 2, 19, 9, 11],
                                 ['AVI134_SL', 10, 3, 3, 20, 9, 11],
                                 ['AVI141_FL', 11, 3, 2, 21, 10, 12],
                                 ['AVI141_SL', 11, 3, 3, 22, 10, 12],
                                 ['AVI142_FL', 12, 3, 2, 23, 11, 13],
                                 ['AVI142_SL', 12, 3, 3, 24, 11, 13],
                                 ['AVI148_FL', 13, 3, 2, 25, 12, 14],
                                 ['AVI148_SL', 13, 3, 3, 26, 12, 14],
                                 ['AVI149_FL', 14, 3, 2, 27, 13, 15],
                                 ['AVI149_SL', 14, 3, 3, 28, 13, 15],
                                 ['AVI156_FL', 15, 3, 2, 29, 14, 16],
                                 ['AVI156_SL', 15, 3, 3, 30, 14, 16],
                                 ['AVI157_FL', 16, 3, 2, 31, 15, 17],
                                 ['AVI157_SL', 16, 3, 3, 32, 15, 17],
                                 ['AVI162_FL', 17, 3, 2, 33, 16, 18],
                                 ['AVI162_SL', 17, 3, 3, 34, 16, 18],
                                 ['AVI163_FL', 18, 3, 2, 35, 17, 19],
                                 ['AVI163_SL', 18, 3, 3, 36, 17, 19],
                                 ['AVI170_FL', 19, 3, 2, 37, 18, 20],
                                 ['AVI170_SL', 19, 3, 3, 38, 18, 20],
                                 ['AVI171_FL', 20, 3, 2, 39, 19, 21],
                                 ['AVI171_SL', 20, 3, 3, 40, 19, 21],
                                 ['AVI177_FL', 21, 3, 2, 41, 20, 22],
                                 ['AVI177_SL', 21, 3, 3, 42, 20, 22],
                                 ['AVI178_FL', 22, 3, 2, 43, 21, 0],
                                 ['AVI178_SL', 22, 3, 3, 44, 21, 0],
                                 ['AVI585_FL', 23, 4, 2, 45, 0, 24],
                                 ['AVI585_SL', 23, 4, 3, 46, 0, 24],
                                 ['AVI584_FL', 24, 4, 2, 47, 23, 25],
                                 ['AVI584_SL', 24, 4, 3, 48, 23, 25],
                                 ['AVI576_FL', 25, 4, 2, 49, 24, 26],
                                 ['AVI576_SL', 25, 4, 3, 50, 24, 26],
                                 ['AVI575_FL', 26, 4, 2, 51, 25, 27],
                                 ['AVI575_SL', 26, 4, 3, 52, 25, 27],
                                 ['AVI569_FL', 27, 4, 2, 53, 26, 28],
                                 ['AVI569_SL', 27, 4, 3, 54, 26, 28],
                                 ['AVI568_FL', 28, 4, 2, 55, 27, 29],
                                 ['AVI568_SL', 28, 4, 3, 56, 27, 29],
                                 ['AVI562_FL', 29, 4, 2, 57, 28, 30],
                                 ['AVI562_SL', 29, 4, 3, 58, 28, 30],
                                 ['AVI561_FL', 30, 4, 2, 59, 29, 31],
                                 ['AVI561_SL', 30, 4, 3, 60, 29, 31],
                                 ['AVI554_FL', 31, 4, 2, 61, 30, 32],
                                 ['AVI554_SL', 31, 4, 3, 62, 30, 32],
                                 ['AVI553_FL', 32, 4, 2, 63, 31, 33],
                                 ['AVI553_SL', 32, 4, 3, 64, 31, 33],
                                 ['AVI546_FL', 33, 4, 2, 65, 32, 34],
                                 ['AVI546_SL', 33, 4, 3, 66, 32, 34],
                                 ['AVI545_FL', 34, 4, 2, 67, 33, 35],
                                 ['AVI545_SL', 34, 4, 3, 68, 33, 35],
                                 ['AVI539_FL', 35, 4, 2, 69, 34, 36],
                                 ['AVI539_SL', 35, 4, 3, 70, 34, 36],
                                 ['AVI538_FL', 36, 4, 2, 71, 35, 37],
                                 ['AVI538_SL', 36, 4, 3, 72, 35, 37],
                                 ['AVI531_FL', 37, 4, 2, 73, 36, 38],
                                 ['AVI531_SL', 37, 4, 3, 74, 36, 38],
                                 ['AVI530_FL', 38, 4, 2, 75, 37, 39],
                                 ['AVI530_SL', 38, 4, 3, 76, 37, 39],
                                 ['AVI522_FL', 39, 4, 2, 77, 38, 40],
                                 ['AVI522_SL', 39, 4, 3, 78, 38, 40],
                                 ['AVI521_FL', 40, 4, 2, 79, 39, 41],
                                 ['AVI521_SL', 40, 4, 3, 80, 39, 41],
                                 ['AVI516_FL', 41, 4, 2, 81, 40, 42],
                                 ['AVI516_SL', 41, 4, 3, 82, 40, 42],
                                 ['AVI515_FL', 42, 4, 2, 83, 41, 43],
                                 ['AVI515_SL', 42, 4, 3, 84, 41, 43],
                                 ['AVI509_FL', 43, 4, 2, 85, 42, 44],
                                 ['AVI509_SL', 43, 4, 3, 86, 42, 44],
                                 ['AVI508_FL', 44, 4, 2, 87, 43, 45],
                                 ['AVI508_SL', 44, 4, 3, 88, 43, 45],
                                 ['AVI504_FL', 45, 4, 2, 89, 44, 46],
                                 ['AVI504_SL', 45, 4, 3, 90, 44, 46],
                                 ['AVI503_FL', 46, 4, 2, 91, 45, 47],
                                 ['AVI503_SL', 46, 4, 3, 92, 45, 47],
                                 ['AVT002_FL', 47, 3, 2, 93, 0, 1],
                                 ['AVT002_SL', 47, 3, 2, 94, 0, 1],
                                 ['AVI007_L', 48, 3, 1, 95, 0, 3],
                                 ['AVT020_L', 49, 3, 1, 96, 0, 50],
                                 ['AVI401_L', 50, 3, 1, 97, 49, 51],
                                 ['AVI402_L', 51, 3, 1, 98, 50, 52],
                                 ['AVI412_L', 52, 3, 1, 99, 51, 53],
                                 ['AVI414_L', 53, 3, 1, 100, 52, 11],
                                 ['AVT030_L', 54, 3, 1, 101, 22, 55],
                                 ['AVT034_FL', 55, 3, 2, 102, 54, 0],
                                 ['AVT034_SL', 55, 3, 3, 103, 54, 0],
                                 ['AVT031_FL', 56, 3, 2, 104, 22, 0],
                                 ['AVT031_SL', 56, 3, 3, 105, 22, 0],
                                 ['AVT037_L', 57, 4, 1, 106, 0, 58],
                                 ['AVI039_FL', 58, 4, 2, 107, 57, 23],
                                 ['AVT036_FL', 59, 4, 2, 108, 0, 60],
                                 ['AVT036_SL', 59, 4, 3, 109, 0, 60],
                                 ['AVI039_FL', 60, 4, 2, 110, 59, 23],
                                 ['AVT033_SL', 60, 4, 3, 111, 59, 23],
                                 ['AVT035_L', 61, 4, 1, 112, 0, 62],
                                 ['AVT032_FL', 62, 4, 2, 113, 61, 23],
                                 ['AVT032_SL', 62, 4, 3, 114, 61, 23],
                                 ['AVI805_L', 63, 4, 1, 115, 36, 64],
                                 ['AVI806_L', 64, 4, 1, 116, 63, 65],
                                 ['AVI802_FL', 65, 4, 2, 117, 64, 66],
                                 ['AVI802_SL', 65, 4, 3, 118, 64, 66],
                                 ['AVI801_FL', 66, 4, 2, 119, 65, 67],
                                 ['AVI801_SL', 66, 4, 3, 120, 65, 67],
                                 ['AVT021_FL', 67, 4, 2, 121, 66, 0],
                                 ['AVT021_SL', 67, 4, 3, 122, 66, 0],
                                 ['AVT004_L', 68, 4, 1, 123, 46, 0],
                                 ['AVT003_FL', 69, 4, 2, 124, 46, 0],
                                 ['AVT003_SL', 69, 4, 3, 125, 46, 0]
                                 ]
                                , columns=['sub_section_name', 'section', 'direction', 'lanes', 'sub_section',
                                           'prev_section', 'next_section'])
road_topology_df.set_index(['sub_section_name'], inplace=True)

holyday_list = [['New Years Day', '2017-01-01', 1],
                ['Australia Day', '2017-01-26', 2],
                ['Good Friday', '2017-04-14', 3],
                ['The day after Good Friday', '2017-04-15', 4],
                ['Easter Sunday', '2017-04-16', 5],
                ['Easter Monday', '2017-04-17', 6],
                ['Aznac Day', '2017-04-25', 7],
                ['Labour Day', '2017-05-01', 8],
                ['Royal Queensland Show', '2017-08-16', 9],
                ['Queens Birthday', '2017-10-2', 10],
                ['Christmas Eve', '2017-12-24', 11],
                ['Christmas Day', '2017-12-25', 12],
                ['Boxing Day', '2017-12-26', 13],
                ['New Years Day', '2018-01-01', 1],
                ['Australia Day', '2018-01-26', 2],
                ['Good Friday', '2018-03-30', 3],
                ['The day after Good Friday', '2018-03-31', 4],
                ['Easter Sunday', '2018-04-01', 5],
                ['Easter Monday', '2018-04-02', 6],
                ['Aznac Day', '2018-04-25', 7],
                ['Labour Day', '2018-05-07', 8],
                ['Royal Queensland Show', '2018-08-15', 9],
                ['Queens Birthday', '2018-10-1', 10],
                ['Christmas Eve', '2018-12-24', 11],
                ['Christmas Day', '2018-12-25', 12],
                ['Boxing Day', '2018-12-26', 13],
                ['New Years Day', '2019-01-01', 1],
                ['Australia Day', '2019-01-28', 2],
                ['Good Friday', '2019-04-19', 3],
                ['The day after Good Friday', '2019-04-20', 4],
                ['Easter Sunday', '2019-04-21', 5],
                ['Easter Monday', '2019-04-22', 6],
                ['Aznac Day', '2019-04-25', 7],
                ['Labour Day', '2019-05-06', 8],
                ['Royal Queensland Show', '2019-08-14', 9],
                ['Queens Birthday', '2019-10-7', 10],
                ['Christmas Eve', '2019-12-24', 11],
                ['Christmas Day', '2019-12-25', 12],
                ['Boxing Day', '2019-12-26', 13]]
holyday_df = pd.DataFrame(holyday_list, columns=['holyday_name', 'holyday_date', 'holyday_id'])
holyday_df['holyday_date'] = pd.to_datetime(holyday_df['holyday_date']).dt.date
holyday_df.set_index(['holyday_date'], inplace=True)

days_dict = {0: 'Monday',
             1: 'Tuesday',
             2: 'Wednesday',
             3: 'Thurday',
             4: 'Friday',
             5: 'Saturday',
             6: 'Sunday'}

import pandas as pd

route = 'G'  # route of the project
data_path_clarence = '{}:\\clarence-data\\'.format(route)


def dl_ia_conn_plot_line(df, var_x, var_y, var_color, x_label, y_label, title_):
    """ Plot line plot from dataframe
    :param df:
    :param var_x:
    :param var_y:
    :param x_label:
    :param y_label:
    :param title_:
    """
    if var_color == None:
        fig = px.line(df,
                      x=var_x,
                      y=var_y)
    else:
        fig = px.line(df,
                      x=var_x,
                      y=var_y,
                      color=var_color)
    fig.update_layout(title=title_,
                      xaxis_title=x_label,
                      yaxis_title=y_label)
    fig.update_layout(
        font=dict(
            family="Courier New, monospace",
            color="RebeccaPurple",
            size=14))
    fig.show()


def dl_ia_conn_plot_scatter(df, var_x, var_y, var_color, x_label, y_label, title_):
    """ Plot scatter plot from dataframe
    :param df:
    :param var_x:
    :param var_y:
    :param x_label:
    :param y_label:
    :param title_:
    """
    if var_color == None:
        fig = px.scatter(df,
                         x=var_x,
                         y=var_y)
    else:
        fig = px.scatter(df,
                         x=var_x,
                         y=var_y,
                         color=var_color)
    fig.update_layout(title=title_,
                      xaxis_title=x_label,
                      yaxis_title=y_label)
    fig.update_layout(
        font=dict(
            family="Courier New, monospace",
            color="RebeccaPurple",
            size=14))
    fig.show()


def dl_ia_conn_plot_contour(df, title, x_label, y_label):
    """

    :return:
    """

    import plotly.graph_objects as go

    fig = go.Figure(data=
    go.Contour(
        z=df.values,
        x=list(range(df.shape[1])),  # horizontal axis
        y=list(range(df.shape[0])),  # vertical axis
        line_smoothing=0.85,
        contours=dict(
            showlabels=True,  # show labels on contours
            start=0,
            end=18,
            size=1)
    ))
    fig.update_layout(title=title,
                      xaxis_title=x_label,
                      yaxis_title=y_label)
    fig.update_layout(
        yaxis=dict(
            tickvals=np.arange(0, len(df.index)),
            ticktext=df.index
        ),
        xaxis=dict(
            tickvals=np.arange(0, len(df.index)),
            ticktext=df.columns,
            tickangle=90,
            tickfont=dict(size=9)
        )
    )

    fig.update_layout(
        font=dict(
            family="Courier New, monospace",
            color="RebeccaPurple",
            size=10))

    fig.show()


def dl_ia_conn_plot_heatmap(df, title, x_label, y_label, x_ticks, y_ticks):
    """ Plot heatmap
    :df param: dataframe to plot
    :title param: string with the title
    :x_label param: string with the label of the x axis
    :y_label param: string with the label of the y axis
    :x_ticks param: list with the ticks of the x axis
    :y_ticks param: list with the ticks of the y axis
    :return:
    """
    import plotly.express as px
    import numpy as np

    fig = px.imshow(df.values)
    fig.update_layout(title=title,
                      yaxis_title=y_label,
                      xaxis_title=x_label)

    fig.update_layout(
        yaxis=dict(
            tickvals=np.arange(0, len(y_ticks)),
            ticktext=y_ticks
        ),
        xaxis=dict(
            tickvals=np.arange(0, len(x_ticks)),
            ticktext=x_ticks
        )
    )
    fig.update_layout(
        font=dict(
            family="Courier New, monospace",
            color="RebeccaPurple",
            size=11))

    fig.show()


def CONN_create_timestamp(row):
    """ Create a time stamp column from others columns with year, month, day and hour
        Use with apply
        :param row: lambda variable regarding columns of the dataframe
        :return datetime:
    """
    import pandas as pd
    try:
        return pd.Timestamp(int(row['YEAR']), int(row['MONTH']), int(row['DAY']), int(row['HOUR']), int(row['MINUTE']))

    except Exception as exception_msg:
        print('(!) Error in CONN_create_timestamp: ' + str(exception_msg))


def CONN_create_date(row):
    """ convert values of HOUR and MINUTE to datetime
    :param row: lambda variable regarding columns of the dataframe
    :return datetime:
    """
    import pandas as pd
    try:
        return pd.Timestamp(int(row['YEAR']), int(row['MONTH']), int(row['DAY']))
    except Exception as exception_msg:
        print('(!) Error in CONN_create_date: ' + str(exception_msg))


def CONN_create_time(row):
    """ convert values of HOUR and MINUTE to datetime
    :param row: lambda variable regarding columns of the dataframe
    :return datetime:
    """
    import datetime
    try:
        return datetime.time(int(row['HOUR']), int(row['MINUTE']), int(row['SECOND']))
    except Exception as exception_msg:
        print('(!) Error in CONN_create_time: ' + str(exception_msg))


def dl_ia_conn_utils_create_datetime(row):
    """ create datetime with hour and minute
    :param row: lambda variable regarding columns of the dataframe
    :return datetime:
    """
    import datetime
    try:
        return datetime.time(int(row['HOUR']), int(row['MINUTE']))
    except Exception as exception_msg:
        print('(!) Error in dl_ia_conn_utils_create_datetime: ' + str(exception_msg))


def dl_ia_conn_utils_create_date(row):
    """ create date with year, month and day
       :param row: lambda variable regarding columns of the dataframe
       :return datetime:
       """
    import pandas as pd
    try:
        return pd.Timestamp(int(row['YEAR']), int(row['MONTH']), int(row['DAY']))
    except Exception as exception_msg:
        print('(!) Error in dl_ia_conn_utils_create_date: ' + str(exception_msg))


def dl_ia_conn_utils_create_timestamp(row):
    """ create date with year, month and day
       :param row: lambda variable regarding columns of the dataframe
       :return datetime:
       """
    import pandas as pd
    try:
        return pd.Timestamp(int(row['YEAR']), int(row['MONTH']), int(row['DAY']), int(row['HOUR']), int(row['MINUTE']))
    except Exception as exception_msg:
        print('(!) Error in dl_ia_conn_utils_create_timestamp: ' + str(exception_msg))


def dl_ia_conn_query_get_data(query, ddbb_settings):
    """
    this function perform the connection to the HORUS  SQL serverdata base and  executes the query provided
    :param query: string with the query
    :param ddbb_settings: List with DB connection settings (driver, server, database, schema,
    user and pass)
    :return error:
    """
    import pyodbc
    import pandas as pd
    error = 0

    try:
        # print('define parameters')

        # for a in db_settings_PRO.keys():
        #    print(a,':', db_settings_PRO[a])

        ### define conection to DDBB
        driver = ddbb_settings['driver']
        server = ddbb_settings['server']
        database = ddbb_settings['database']
        schema = ddbb_settings['schema']
        user = ddbb_settings['user']
        password = ddbb_settings['pass']
        port = ddbb_settings['port']


    except Exception as exception_msg:
        print('(!) Error in dl_ia_conn_query_get_data: ' + str(exception_msg))
        error = 1
        df_input = []
        return error, df_input

    if error == 0:
        try:
            print('Loading data from server:[{}] database:[{}] schema:[{}] port : [{}] '.format(server, database, schema, port))
            ### connect do DDBB and get last 6 hours od data
            # sql_conn = pyodbc.connect('DRIVER={ODBC Driver 13 for SQL Server};SERVER=' + server + ';DATABASE=' + database + ';UID=' + user + ';PWD=' + password + 'Trusted_Connection=yes')
            # sql_conn = pyodbc.connect('DRIVER={SQL Server Native Client RDA 11.0};SERVER=' + server + ';DATABASE=' + database + ';UID=' + user + ';PWD=' + password + 'Trusted_Connection=yes')
            sql_conn_str = 'DRIVER={};SERVER={},{};DATABASE={};UID={};PWD={}'.format(driver, server, port, database, user, password)
            # print(sql_conn_str)
            # sql_conn = pyodbc.connect('DRIVER=' + driver +';SERVER=' + server + ',' + port + ';DATABASE=' + database + ';UID=' + user + ';PWD=' + password)
            sql_conn = pyodbc.connect(sql_conn_str)
            df_input = pd.read_sql(query, sql_conn)
            sql_conn.close()
            return error, df_input

        except Exception as exception_msg:
            print('(!) Error in dl_ia_conn_query_get_data: ' + str(exception_msg))
            error = 2
            df_input = []
            return error, df_input


def dl_ia_conn_initialize_engine(ddbb_settings):
    """ Initialize an SQL ALchemy engine
    :param ddbb_settings: DIctionary with driver user, pass, server, database, schema
    :param engine:
    """
    from sqlalchemy.engine import create_engine

    try:
        engine = create_engine("{}://{}:{}@{}:{}/{}".format(ddbb_settings['driver'],
                                                            ddbb_settings['user'],
                                                            ddbb_settings['pass'],
                                                            ddbb_settings['server'],
                                                            ddbb_settings['port'],
                                                            ddbb_settings['database']))
        print('Engine successfully initialized')
        return engine
    except Exception as exception_msg:
        print('(!) Error in dl_ia_conn_initialize_engine: {}'.format(exception_msg))
        engine = []
        return engine


def check_folder(path_folder):
    """ check that exists a folder, and if not, create it
    :param path_folder: string with the path
    :return error: error code (0:good, 1:bad)
    """
    import os
    error = 0
    try:
        if not os.path.isdir(path_folder):
            CLA_comm('Creating folder: {} '.format(path_folder))
            os.mkdir(path_folder)
    except Exception as exception_msg:
        print('(!) Error in check_folder: ' + str(exception_msg))
        error = 1
        return error


def CONN_config_plotly():
    """ this function configures the plotly visualization

    :return:
    """
    import plotly.io as pio
    import plotly.graph_objects as go
    import plotly.express as px

    pio.renderers.default = "browser"


def CONN_config_matplotlib():
    """ this function configures the matplotlib style

    :return:
    """
    from matplotlib import rc
    rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
    rc('font', **{'family': 'serif', 'serif': ['Times']})
    rc('text', usetex=True)


def CONN_config_pandas():
    """ this function configures the pandas library

    :return:
    """
    import pandas as pd
    pd.set_option('mode.chained_assignment', None)


def read_csv_per_chunks(path):
    """ This function read a large csv file into a dataframe per chunks

    :param path:
    :return df: dataframe
    """
    import pandas as pd
    chunksize_ = 1000
    error = 0

    try:
        TextFileReader = pd.read_csv(path, sep=";", chunksize=chunksize_)
        dfList = []
        for df in TextFileReader:
            dfList.append(df)
        df = pd.concat(dfList, sort=False)

        return error, df

    except Exception as exception_msg:
        print("Error in read_csv_per_chunks {}".format(exception_msg))
        # raise
        error = 1
        df = []
        return error, df


def CONN_comm(msg):
    """ Funtion to show mesages in terminal

    :parm msg: meassge (str)
    :return:
    """
    print('{} {}'.format('-' * 20, msg))


def plot_timeseries_with_slider(df, var_x, var_y, title):
    """

    :param df:
    :param var_x:
    :param var_y:
    :return:
    """

    import plotly.graph_objects as go
    from CONN_CONN_utils import config_plotly
    config_plotly()

    show = True
    print_ = True

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[var_x],
                             y=df[var_y],
                             mode='markers+lines',
                             marker=dict(color='red'),
                             name=var_y))
    fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
    fig.update_layout(title=title,
                      xaxis_title=var_x,
                      yaxis_title=var_y,
                      showlegend=True
                      )

    # Add range slider
    fig.update_layout(
        xaxis=go.layout.XAxis(
            rangeselector=dict(
                buttons=list([
                    dict(count=1,
                         label="1d",
                         step="day",
                         stepmode="backward"),
                    dict(count=3,
                         label="3d",
                         step="day",
                         stepmode="backward"),
                    dict(count=7,
                         label="1w",
                         step="day",
                         stepmode="backward"),
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            type="date"
        )
    )
    if show:
        fig.show()
    if print_:
        fig.write_html("figures\\timeseries_with_slider_{}.html".format(var_y))


def CONN_plot_timeseries(df, var_x, var_y):
    """

    :param df:
    :param var_x:
    :param var_y:
    :return:
    """

    import plotly.graph_objects as go
    from CONN_CONN_utils import CONN_config_plotly
    CONN_config_plotly()

    show = True
    print_ = True

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[var_x],
                             y=df[var_y],
                             marker=dict(color='red'),
                             mode='markers+lines',
                             name=var_y))
    fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
    fig.update_layout(title='Time series',
                      xaxis_title=var_x,
                      yaxis_title=var_y,
                      showlegend=True
                      )

    if show:
        fig.show()
    if print_:
        fig.write_html("figures\\timeseries_{}.html".format(var_y))


def plot_two_timeseries(df, var_x, var_y1, var_y2):
    """

    :param df:
    :param var_x:
    :param var_y1:
    :param var_y2:
    :return:
    """
    import plotly.graph_objects as go
    from CONN_utils import config_plotly, comm
    config_plotly()

    x_label = 'TIME (15 min)'
    y_label = 'FLOW (veh./h)'
    title_ = 'Time series'
    show = True
    print_ = True

    try:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df[var_x],
                                 y=df[var_y1],
                                 marker=dict(color='red'),
                                 mode='markers',
                                 name=var_y1))
        fig.add_trace(go.Scatter(x=df[var_x],
                                 marker=dict(color='blue'),
                                 y=df[var_y2],
                                 mode='markers+lines',
                                 name=var_y2))
        fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
        fig.update_layout(title=title_,
                          xaxis_title=x_label,
                          yaxis_title=y_label,
                          showlegend=True
                          )

        # Add range slider
        fig.update_layout(
            xaxis=go.layout.XAxis(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1,
                             label="1d",
                             step="day",
                             stepmode="backward"),
                        dict(count=3,
                             label="3d",
                             step="day",
                             stepmode="backward"),
                        dict(count=7,
                             label="1w",
                             step="day",
                             stepmode="backward"),
                    ])
                ),
                rangeslider=dict(
                    visible=True
                ),
                type="date"
            )
        )
        if show:
            fig.show()
        if print_:
            fig.write_html("figures\\two_timeseries_{}_{}.html".format(var_y1, var_y2))

    except Exception as exception_msg:
        comm('(!) Error in plot_two_timeseries : {} '.format(exception_msg))


def plot_line(df, var_x, var_y, var_group):
    """

    :param df:
    :param var_x:
    :param var_y:
    :param var_group:
    :return:
    """
    import plotly.express as px
    from CONN_utils import config_plotly
    config_plotly()

    show = True
    print_ = True

    fig = px.line(df,
                  x=var_x,
                  y=var_y,
                  color=var_group,
                  )
    fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
    if show:
        fig.show()
    if print_:
        fig.write_html("figures\\line_plot_simple_{}_{}.html".format(var_x, var_y))


def dl_ia_quarter_classify(x):
    """ classify a variabel x into four cuadrants

    :param x: value with values in (0,60)
    :return y: values with values in (1,2,3,4)

    """
    if x <= 15:
        y = 0
    if 30 >= x > 15:
        y = 15
    if 45 >= x > 30:
        y = 30
    if x > 45:
        y = 45
    return y


def model_evaluation(model, X, y, type):
    """ Evaluate regression model

    :param model:
    :param X:
    :param y:
    :param type:
    :return:
    """

    import numpy as np
    from CONN_utils import comm, variability_captured, mape
    from sklearn.metrics import r2_score, mean_absolute_error, explained_variance_score, mean_squared_error

    # make prediction
    y_hat = model.predict(X)

    R2 = round(r2_score(y_hat, y), 3)
    MAE = round(mean_absolute_error(y_hat, y), 3)
    MSE = round(mean_squared_error(y_hat, y), 3)
    EV = round(explained_variance_score(y_hat, y), 3)
    VC = round(variability_captured(y_hat, y), 3)

    errors = abs(y_hat - y)
    # MAPE = 100 * np.mean(errors / y)
    MAPE = mape(y_hat, y)
    accuracy = 100 - MAPE

    comm('{} - Metrics'.format(type))
    comm('R2 = {:0.2f}'.format(R2))
    comm('EV = {:0.2f}'.format(VC))
    comm('Variability = {:0.2f}'.format(EV))
    comm('MSE = {:0.2f}'.format(MSE))
    comm('MAE = {:0.2f}'.format(MAE))
    comm('MAPE: {:0.4f} %'.format(MAPE))
    comm('Accuracy = {:0.2f} %.'.format(accuracy))

    return R2, MAE


def comm(msg):
    """ function to display in console msg

    :param msg:
    :return:
    """

    print('{} {}'.format('-' * 20, msg))


def variability_captured(y, y_hat):
    """ function to calculate the varibility captured or explained variance

    :param y:
    :param y_hat:
    :return:
    """
    import numpy as np
    return round(1 - np.var(y_hat - y) / np.var(y_hat), 3)


def mape(y_true, y_pred):
    """ function to calculate the Mean Absolute Percentage Error
        Zero values are treated by vertical translation

    :param y:
    :param y_hat:
    :return:
    """
    import numpy as np
    from CONN_utils import vertical_translation

    y_true = vertical_translation(y_true)  # vertical translation +1
    y_pred = vertical_translation(y_pred)  # vertical translation +1

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return round(np.mean(np.abs((y_true - y_pred) / y_true)) * 100, 3)


def subs_zeros_values(y):
    """ subs zero values from from an array by values close to zeros 1e-10
        e.g.: y = np.array([1,4,2,3,7,8,0,0,8,7,0,0,9,8])
    :param y:
    :return:
    """
    import pandas as pd

    df = pd.DataFrame({'y': y})
    df.loc[df['y'] == 0, ['y']] = 1e-9
    return df['y'].values


def vertical_translation(y):
    """ detects in exist a zero value and translate the time series with the minimum value
    :param y:
    :return:
    """
    import numpy as np
    if np.isin(0, y):
        # exists a zero value, find the minimum distinct from zero
        delta = np.min(y[y > 0])
        # vertical translation
        # ym = y + delta
        ym = y + 1
        return ym
    return y


def plot_marginal_dist_plot(df, var_x, var_y):
    """

    :param df:
    :param var_x:
    :param var_y:
    :return:
    """
    from CONN_utils import config_plotly
    config_plotly()
    import plotly.express as px

    show = True
    print_ = True

    fig = px.density_heatmap(df,
                             x=var_x,  # title="Click on the legend items!"
                             y=var_y,  # add color="species",
                             marginal_x="box",  # histogram, rug
                             marginal_y="violin")
    fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
    if show:
        fig.show()
    if print_:
        fig.write_html("figures\\plot_marginal_dist_plot_{}_{}.html".format(var_x, var_y))


def plot_scatter_with_facets(df, var_x, var_y, var_color, var_group):
    """

    :param df:
    :param var_x:
    :param var_y:
    :param var_color:
    :param var_group:
    :return:
    """
    import plotly.express as px
    from CONN_utils import config_plotly
    config_plotly()

    show = True
    print_ = True

    fig = px.scatter(df,
                     x=var_x,
                     y=var_y,
                     color=var_color,
                     facet_col=var_group,
                     marginal_x="box")  # violin, histogram, rug
    fig.update_layout(font=dict(family="Courier New, monospace",
                                size=18,
                                color="#7f7f7f"))
    if show:
        fig.show()
    if print_:
        fig.write_html("figures\\plot_scatter_with_facets_{}_{}_{}_{}.html".format(var_x, var_y, var_color, var_group))


def create_lagged_variables(df, variable, number_lags):
    """ create lagged versions of the variable in a dataframe in which each row is an observation

    :param df:
    :param number_lags:
    :return:
    """
    number_lags = 23
    for lag in range(1, number_lags + 1):
        df[variable + '_lag_' + str(lag)] = df[variable].shift(lag)

    # if you want numpy arrays with no null values:
    df.dropna(inplace=True)

    return df


def plot_multi_timeseries():
    start = '2018-05-01 00:00:00'
    end = '2018-05-15 00:00:00'
    df_aux = df[(df['DATE'] > start) & (df['DATE'] < end)]

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df_aux['DATE'],
                             y=df_aux['TOTAL_VEHICULOS'] * 4,
                             name='Real'))

    fig.add_trace(go.Scatter(x=df_aux['DATE'],
                             y=df_aux['TOTAL_VEHICULOS_DESCRIPTIVE'] * 4,
                             name='Descriptive Model (R2:{:2.2f} / MAE:{:2.2f}%)'.format(r2_desc, MAE_desc)))

    fig.add_trace(go.Scatter(x=df_aux['DATE'],
                             y=df_aux['TOTAL_VEHICLES_RF_PREDICTION'] * 4,
                             name='Random Forest Model (R2:{:2.2f} / MAE:{:2.2f}%)'.format(r2_rf,
                                                                                           MAE_rf)))
    fig.add_trace(go.Scatter(x=df_aux['DATE'],
                             y=df_aux['TOTAL_VEHICLES_NN_PREDICTION'] * 4,
                             mode='lines',
                             marker_color='rgba(152, 0, 0, .8)',
                             name='Neural Network Model (R2:{:2.2f} / MAE:{:2.2f}%)'.format(r2_nn,
                                                                                            MAE_nn)))

    fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
    fig.update_layout(title='Predictive results - Segment: {} -  Dates: {} to {}'.format(segments[seg],
                                                                                         start[0:10],
                                                                                         end[0:10]),
                      xaxis_title='Time (15 min. resolution)',
                      yaxis_title='Flow (veh./h)',
                      showlegend=True
                      )

    fig.update_layout(legend=dict(x=0, y=-0.5, bgcolor="white"))

    fig.show()


def plot_multi_timeseries_with_slider():
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df['DATE'],
                             y=df['TOTAL_VEHICULOS'],
                             name='Real Data'))

    fig.add_trace(go.Scatter(x=df['DATE'],
                             y=df['TOTAL_VEHICULOS_DESCRIPTIVE'],
                             name='Descriptive Model (R2:{:2.2f} / MAE:{:2.2f}%)'.format(r2_desc, MAE_desc)))

    fig.add_trace(go.Scatter(x=df['DATE'],
                             y=df['TOTAL_VEHICLES_RF_PREDICTION'],
                             name='Random Forest Predictive Model (R2:{:2.2f} / MAE:{:2.2f}%)'.format(r2_rf,
                                                                                                      MAE_rf)))
    # Set x-axis title
    fig.update_xaxes(title_text="Time")

    # Set y-axes titles
    fig.update_yaxes(title_text="Total Vehicles")

    fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
    fig.update_layout(title='AUSOL - Road Traffic Flow - Segment: {}'.format(segments[seg]),
                      showlegend=True
                      )

    fig.update_layout(legend=dict(x=0, y=-1.0, bgcolor="white"))

    # Add range slider
    fig.update_layout(
        xaxis=go.layout.XAxis(
            rangeslider=dict(
                visible=True
            ),
            type="date"
        )
    )
    fig.show()


def check_null_values(df):
    """

    :param df:
    :return df:
    """
    from CONN_utils import comm
    # check nans
    if df.isna().sum().sum() > 0:
        comm('(!) NAN Values detected')
        print(df.isna().sum())
        df.dropna(inplace=True)
        return df
    elif df.isnull().sum().sum() > 0:
        comm('(!) NULLs Values detected')
        print(df.isnull().sum())
        df.dropna(inplace=True)
        return df
    else:
        comm('Everything ok')
        return df


def CONN_plot_histogram(df, variable, n_bins_, label):
    """ plot a histogram using plotly from a vairable in a dataframe

    :param df: Data frame with the variable
    :param variable: name of the variable (column name)
    :param n_bins_: number of bins of the histogram
    :param label: string with a name for the title
    :return error: error code 0: everything ok, 1: something happened
    """

    import plotly.express as px
    from CONN_utils import CONN_config_plotly, CONN_comm
    import numpy as np
    CONN_config_plotly()

    print_ = True
    show = True
    fontsize_ = 18
    error = 0

    try:
        max_value = int(df[variable].max())
        x_axis = np.arange(0, max_value, int(max_value / 20))

        fig = px.histogram(df, x=variable, nbins=n_bins_, marginal="box")
        fig.update_xaxes(title_text=variable)
        fig.update_layout(font=dict(family="Courier New, monospace", size=fontsize_, color="#7f7f7f"))
        fig.update_layout(title='Histogram - {} - {}'.format(label, variable))
        fig.update_layout(showlegend=True)
        fig.update_traces(opacity=0.9)
        fig.update_layout(bargap=0.2)  # gap between bars of adjacent location coordinates
        fig.update_xaxes(ticktext=x_axis, tickvals=x_axis)

        if print_:
            fig.write_html("figures\\plot_histogram_{}.html".format(variable))
        if show:
            fig.show()
        return error

    except Exception as exception_msg:
        CONN_comm('(!) Error in plot_histogram: ' + str(exception_msg))
        error = 1
        return error


def CONN_time_series_plot(time_index, y1, label1, title):
    """ Plot one single time series

    :param time_index:
    :param y1:
    :param y2:
    :param label1:
    :param label2:
    :return:
    """

    import plotly.graph_objects as go
    from CONN_utils import CONN_config_plotly
    CONN_config_plotly()

    print_ = True
    show = True
    # title = 'time_series_comparison'

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_index,
                             y=y1,
                             mode='markers+lines',
                             marker=dict(color='red'),
                             name='{}'.format(label1)))

    fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
    fig.update_layout(showlegend=True)
    fig.update_yaxes(title_text=label1)
    fig.update_xaxes(title_text='Time')
    fig.update_layout(title=title)

    # Add range slider
    fig.update_layout(
        xaxis=go.layout.XAxis(
            rangeselector=dict(
                buttons=list([
                    dict(count=1,
                         label="1 day",
                         step="day",
                         stepmode="backward"),
                    dict(count=3,
                         label="3 day",
                         step="day",
                         stepmode="backward"),
                    dict(count=7,
                         label="1 week",
                         step="day",
                         stepmode="backward"),
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            type="date"
        )
    )
    if print_:
        fig.write_html("figures\\time_series_{}.html".format(label1))
    if show:
        fig.show()


def CONN_time_series_comparison(time_index, y1, y2, label1, label2, title):
    """ Plot two time series with the same time index

    :param time_index:
    :param y1:
    :param y2:
    :param label1:
    :param label2:
    :return:
    """

    import plotly.graph_objects as go
    from CONN_utils import CONN_config_plotly
    CONN_config_plotly()

    print_ = True
    show = True
    # title = 'time_series_comparison'

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_index,
                             y=y1,
                             mode='markers+lines',
                             marker=dict(color='red'),
                             name='{}'.format(label1)))

    fig.add_trace(go.Scatter(x=time_index,
                             y=y2,
                             mode='markers+lines',
                             marker=dict(color='blue'),
                             name='{}'.format(label2)))

    fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
    fig.update_layout(showlegend=True)
    fig.update_yaxes(title_text=label1)
    fig.update_xaxes(title_text='Time')
    fig.update_layout(title=title)

    # Add range slider
    fig.update_layout(
        xaxis=go.layout.XAxis(
            rangeselector=dict(
                buttons=list([
                    dict(count=1,
                         label="1 day",
                         step="day",
                         stepmode="backward"),
                    dict(count=3,
                         label="3 day",
                         step="day",
                         stepmode="backward"),
                    dict(count=7,
                         label="1 week",
                         step="day",
                         stepmode="backward"),
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            type="date"
        )
    )
    if print_:
        fig.write_html("figures\\time_series_comparison_{}_{}.html".format(label1, label2))
    if show:
        fig.show()


def CONN_plot_scatterplot_simple(df, var_x, var_y, label):
    """ Produce a simple scatter plot with plotly

    :param df: dataframe that contains the variables
    :param variable_x: variable to plot in x axis
    :param variable_y: variable to plot in y axis
    :param variable_to_color: variable to use as color
    :param variable_to_color: variable to use as size
    :return:
    """
    import plotly.express as px
    from CONN_utils import CONN_config_plotly, CONN_comm
    CONN_config_plotly()

    print_ = True
    show = True
    error = 0

    try:
        fig = px.scatter(df,
                         x=var_x,
                         y=var_y,  # marker = dict(color='blue')
                         )

        fig.update_xaxes(title_text=var_x)
        fig.update_yaxes(title_text=var_y)
        fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
        fig.update_layout(title='Scatterplot - {}'.format(label))
        fig.update_layout(showlegend=True)

        if print_:
            fig.write_html("figures\\scatterplot_{}_{}_{}.html".format(label, var_x, var_y))
        if show:
            fig.show()

        return error

    except Exception as exception_msg:

        CONN_comm('(!) Error in CONN_plot_scatterplot_simple: ' + str(exception_msg))
        error = 1
        return error


def CONN_plot_scatterplot(df, var_x, var_y, var_color, var_size, label):
    """ Produce a simple scatter plot with plotly

    :param df: dataframe that contains the variables
    :param variable_x: variable to plot in x axis
    :param variable_y: variable to plot in y axis
    :param variable_to_color: variable to use as color
    :param variable_to_color: variable to use as size
    :return:
    """
    import plotly.express as px
    from CONN_utils import CONN_config_plotly, CONN_comm
    CONN_config_plotly()

    print_ = True
    show = True
    error = 0

    try:
        fig = px.scatter(df,
                         x=var_x,
                         y=var_y,  # marker = dict(color='blue')
                         size=var_size,
                         color=var_color)

        fig.update_xaxes(title_text=var_x)
        fig.update_yaxes(title_text=var_y)
        fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
        fig.update_layout(title='Scatterplot - {}'.format(label))
        fig.update_layout(showlegend=True)

        if print_:
            fig.write_html("figures\\scatterplot_{}_{}_{}.html".format(label, var_x, var_y))
        if show:
            fig.show()

        return error


    except Exception as exception_msg:

        CONN_comm('(!) Error in plot_scatterplot: ' + str(exception_msg))
        error = 1
        return error


def CONN_create_DDBB_connection():
    """ This function build a engine with the connection to the DDBB of the INDRA architecture
    :return engine: ENgine to run queries against DDBB
    """

    from sqlalchemy.engine import create_engine
    from CONN_utils import CONN_comm

    driver = 'postgresql'
    user_ = 'gpadmin'
    pass_ = 'pivotal'
    host = '10.0.2.6'
    port = '5432'
    ddbb = 'gpadmin'

    CONN_comm('Build engine with connection to DDBB:')
    CONN_comm('Driver:{}'.format(driver))
    CONN_comm('User:{}'.format(user_))
    CONN_comm('Pass:{}'.format(pass_))
    CONN_comm('Host:{}'.format(host))
    CONN_comm('Port:{}'.format(port))
    CONN_comm('Database:{}'.format(ddbb))

    engine = create_engine("{}://{}:{}@{}:{}/{}".format(driver, user_, pass_, host, port, ddbb))
    return engine


def CONN_systems_info():
    """ Function that shows the system properties

    """
    import sys
    from platform import python_version
    from CONN_utils import CONN_comm

    # CONN_comm('Python version:{}'.format(python_version()))
    CONN_comm('Python version:{}'.format(sys.version))
    CONN_comm('Path:{}'.format(sys.executable))
    # CONN_comm('Python version info:{}'.format(sys.version_info))


def CONN_anomaly_detection_univariate(df, variable):
    """ Produce anomaly detection with forest isolation with univariate data
    :param df:
    :param variable:

    """

    from sklearn.ensemble import IsolationForest
    import numpy as np
    import pandas as pd
    from CONN_utils import CONN_comm

    error = 0

    try:
        # instantiate model
        isolation_forest = IsolationForest(n_estimators=200)

        # fit model
        isolation_forest.fit(df[variable].values.reshape(-1, 1))
        xx = np.linspace(df[variable].min(), df[variable].max(), len(df)).reshape(-1, 1)
        anomaly_score = isolation_forest.decision_function(xx)

        # make prediction
        outlier = isolation_forest.predict(xx)

        df_out = pd.DataFrame({'X': xx.T.ravel(), 'outlier': outlier.ravel(), 'anomaly_score': anomaly_score.ravel()})

        lower_limit_outlier = df_out[df_out['outlier'] == -1]['X'].min()
        upper_limit_outlier = df_out[df_out['outlier'] == -1]['X'].max()

        CONN_comm('lower limit outlier:{}'.format(lower_limit_outlier))
        CONN_comm('upper limit outlier:{}'.format(upper_limit_outlier))

        return error, df_out

        # plot
        if 0 == 1:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 4))
            plt.plot(xx, anomaly_score, label='anomaly score')
            plt.fill_between(xx.T[0],
                             np.min(anomaly_score),
                             np.max(anomaly_score),
                             where=outlier == -1,
                             color='r',
                             alpha=.4,
                             label='outlier region')
            plt.legend()
            plt.ylabel('anomaly score')
            plt.xlabel('Sales')
            plt.show();

    except Exception as exception_msg:
        CONN_comm('(!) Error in CONN_anomaly_detection_univariate: ' + str(exception_msg))
        error = 1
        return error


def sql_server_query_get_data(query):
    """ this function perform the connection to the HORUS  SQL serverdata base and  executes the query provided
    :param query: string with the query
    :return error:
    """
    import pyodbc
    import pandas as pd
    from dl_ia_cla_settings import CLA_comm
    error = 0

    ### define conection to DDBB
    driver = 'ODBC Driver 13 for SQL Server'
    server = '10.72.1.11'
    database = 'HIST_HORUS'
    schema = 'dbo'
    user = 'sa'
    password = 'sa$2019'

    try:
        CLA_comm('Loading data from server:[{}] database:[{}] schema:[{}] '.format(server, database, schema))
        ### connect do DDBB and get last 6 hours od data
        # sql_conn = pyodbc.connect('DRIVER={ODBC Driver 13 for SQL Server};SERVER=' + server + ';DATABASE=' + database + ';UID=' + user + ';PWD=' + password + 'Trusted_Connection=yes')
        # sql_conn = pyodbc.connect('DRIVER={SQL Server Native Client RDA 11.0};SERVER=' + server + ';DATABASE=' + database + ';UID=' + user + ';PWD=' + password + 'Trusted_Connection=yes')
        sql_conn = pyodbc.connect('DRIVER={SQL Server};SERVER=' + server + ';DATABASE=' + database + ';UID=' + user + ';PWD=' + password)
        df_input = pd.read_sql(query, sql_conn)
        sql_conn.close()
        return df_input

    except Exception as exception_msg:
        CLA_comm('(!) Error in sql_server_query_get_data: ' + str(exception_msg))
        error = 1
        return error


def get_unique_values(df_in):
    """ this function calculate the unique values of the column of a data frame

    :param df_in: dataframe with the columns of interest
    :return dict_out: dictionary with unique values of the columns
    """

    import numpy as np

    dict_out = dict()
    for column in df_in.columns:
        dict_out[column] = np.sort(df_in[column].unique())

    return dict_out


def quarter_classify(x):
    """ classify a variabel x into four cuadrants

    :param x: value with values in (0,60)
    :return y: values with values in (1,2,3,4)

    """
    if x <= 15:
        y = 0
    if 30 >= x > 15:
        y = 15
    if 45 >= x > 30:
        y = 30
    if x > 45:
        y = 45
    return y


def quarter_groups(x):
    """ classify a variabel x into four cuadrants

    :param x: value with values in (0,60)
    :return y: values with values in (1,2,3,4)

    """
    if x <= 15:
        y = 1
    if 30 >= x > 15:
        y = 2
    if 45 >= x > 30:
        y = 3
    if x > 45:
        y = 4
    return y


def create_date(row):
    """
     df['date'] = df.apply(lambda row: create_date(row), axis=1)
    """
    return row['TIMESTAMP'].strftime('%Y-%m-%d')


def model_evaluation(model, X, y, type):
    """ Evaluate regression model

    :param model:
    :param X:
    :param y:
    :param type:
    :return:
    """

    from dl_ia_cla_settings import comm, variability_captured, mape
    from sklearn.metrics import r2_score, mean_absolute_error, explained_variance_score, mean_squared_error

    # make prediction
    y_hat = model.predict(X)

    R2 = round(r2_score(y_hat, y), 3)
    MAE = round(mean_absolute_error(y_hat, y), 3)
    MSE = round(mean_squared_error(y_hat, y), 3)
    EV = round(explained_variance_score(y_hat, y), 3)
    VC = round(variability_captured(y_hat, y), 3)

    errors = abs(y_hat - y)
    # MAPE = 100 * np.mean(errors / y)
    MAPE = mape(y_hat, y)
    accuracy = 100 - MAPE

    comm('{} - Metrics'.format(type))
    comm('R2 = {:0.2f}'.format(R2))
    comm('EV = {:0.2f}'.format(VC))
    comm('Variability = {:0.2f}'.format(EV))
    comm('MSE = {:0.2f}'.format(MSE))
    comm('MAE = {:0.2f}'.format(MAE))
    comm('MAPE: {:0.4f} %'.format(MAPE))
    comm('Accuracy = {:0.2f} %.'.format(accuracy))

    return R2, MAE


def variability_captured(y, y_hat):
    """ function to calculate the varibility captured or explained variance

    :param y:
    :param y_hat:
    :return:
    """
    import numpy as np
    return round(1 - np.var(y_hat - y) / np.var(y_hat), 3)


def regression_evaluation(y_hat, y):
    """ Evaluate regression metrics

    :param model:
    :param X:
    :param y:
    :param type:
    :return:
    """

    from dl_ia_cla_settings import CLA_comm, variability_captured, mape
    from sklearn.metrics import r2_score, mean_absolute_error, explained_variance_score, mean_squared_error

    R2 = round(r2_score(y_hat, y), 3)
    MAE = round(mean_absolute_error(y_hat, y), 3)
    MSE = round(mean_squared_error(y_hat, y), 3)
    EV = round(explained_variance_score(y_hat, y), 3)
    VC = round(variability_captured(y_hat, y), 3)

    errors = abs(y_hat - y)
    # MAPE = 100 * np.mean(errors / y)
    MAPE = mape(y_hat, y)
    accuracy = 100 - MAPE

    CLA_comm('Regression Metrics'.format(type))
    CLA_comm('R2 = {:0.2f}'.format(R2))
    CLA_comm('EV = {:0.2f}'.format(VC))
    CLA_comm('Variability = {:0.2f}'.format(EV))
    CLA_comm('MSE = {:0.2f}'.format(MSE))
    CLA_comm('MAE = {:0.2f}'.format(MAE))
    CLA_comm('MAPE: {:0.4f} %'.format(MAPE))
    CLA_comm('Accuracy = {:0.2f} %.'.format(accuracy))

    return R2, MAE


def mape(y_true, y_pred):
    """ function to calculate the Mean Absolute Percentage Error
        Zero values are treated by vertical translation

    :param y:
    :param y_hat:
    :return:
    """
    import numpy as np
    from dl_ia_cla_settings import vertical_translation

    y_true = vertical_translation(y_true)  # vertical translation +1
    y_pred = vertical_translation(y_pred)  # vertical translation +1

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return round(np.mean(np.abs((y_true - y_pred) / y_true)) * 100, 3)


def subs_zeros_values(y):
    """ subs zero values from from an array by values close to zeros 1e-10
        e.g.: y = np.array([1,4,2,3,7,8,0,0,8,7,0,0,9,8])
    :param y:
    :return:
    """
    import pandas as pd

    df = pd.DataFrame({'y': y})
    df.loc[df['y'] == 0, ['y']] = 1e-9
    return df['y'].values


def vertical_translation(y):
    """ detects in exist a zero value and translate the time series with the minimum value
    :param y:
    :return:
    """
    import numpy as np
    if np.isin(0, y):
        # exists a zero value, find the minimum distinct from zero
        delta = np.min(y[y > 0])
        # vertical translation
        # ym = y + delta
        ym = y + 1
        return ym
    return y


def convert_datetime(row):
    """ convert values of HOUR and MINUTE to datetime

    :param row: lambda variable regarding columns of the dataframe
    :return datetime:
    """
    import datetime
    return datetime.time(int(row['HOUR']), int(row['MINUTE']))


def convert_date(row):
    """ convert values of HOUR and MINUTE to datetime

    :param row: lambda variable regarding columns of the dataframe
    :return datetime:
    """
    import datetime
    return datetime.time(int(row['YEAR']), int(row['MONTH']), int(row['DAY']))


def create_time_array(start, end, freq):
    """ function that creates an array of times

    :param start: string with the initial time (e.g.: 00:00:00)
    :param end: string with the end time (e.g.: 23:59:59)
    :parm freq: string indicating the frequency (e.g.: 15min)
    :return: array of time

    """
    t = pd.DataFrame({'t': pd.date_range(start=start, end=end, freq=freq)}).t.dt.date

    return t


def CLA_comm(msg):
    """ Funtion to show mesages in terminal

    :parm msg: meassge (str)
    :return:
    """
    print('{} {}'.format('-' * 20, msg))


units_ = {"FLOW": "(veh./h)",
          'flow': "(veh./h)",
          "flow_eq": "(veh./h)",
          "flow_error": "(veh./h)",
          "VOLUME": "(veh./15min)",
          'volume': '(veh./15min)',
          'volume_smooth': '(veh./15min)',
          'volume_desc': '(veh./15min)',
          'speed': '(km/h)',
          'speed_eq': '(km/h)',
          'speed_error': '(km/h)',
          "SPEED": '(km/h)',
          "meanlength": "(dm)",
          "gap": "(tenths of sec)",
          "headway": '(tenths of sec)',
          'occupancy': '(%)',
          "OCCUPANCY": "(%)"}

colors_ = {'0-6': 'darkblue',
           '7-12': 'darkgreen',
           '13-17': 'darkred',
           '18-21': 'darkviolet',
           '22-23': 'darkgrey',
           }


def speed_eq(Occ, Oc, L, l, avg_speed):
    """

    :param Occ: Critical Occupancy (20%)
    :param Oc: Actual value of occupancy
    :param L: meanlength of the vehicle (4.4-5.5)
    :param l: lop length (2m tunnel, 1 road)
    :param avg_speed: average speed of the vehicles (km/h)
    :return: speed estimation
    """
    import numpy as np
    Lp = L + l
    a0 = (-50 / (Occ ** 2)) * (1 / Lp) ** 2
    speed_formula = avg_speed * np.exp(Oc ** 2 * a0)

    return speed_formula, a0


def CLA_add_holidays_info(df):
    """ Add columns HOLIDAYS_ID and SPECIAL_EVENT to the dataframe if road traffic data
        for the simlation

        HOLIDAYS_ID
            0:  Non Holidays
            1:  New Years Day
            2:  Australia Day
            3:  Good Friday
            4:  The day after Good
            5:  Easter Sunday
            6:  Easter Monday
            7:  Aznac Day
            8:  Labour Day
            9:  Royal Queensland Show
            10: Queens Birthday
            11: Christmas Eve
            12: Christmas Day
            13: Boxing Day

        SPECIAL_EVENT
            NH: Non Holidays
            HD: Holidays departure
            HV: Holidays Valley
            HR: Holidays Return

        :param df: Dataframe with the road traffic data and TIMESTAMP column
        :return df_out: dataframe with the columns HOLIDAYS_ID and SPECIAL_EVENT

    """

    import numpy as np
    import pandas as pd
    from datetime import timedelta
    from dl_ia_cla_settings import CLA_comm

    df_holidays = pd.read_csv('data\\holidays.csv')
    df_holidays['date'] = pd.to_datetime(df_holidays['date']).map(lambda x: x.strftime('%Y-%m-%d'))
    df_holidays['TIMESTAMP'] = pd.to_datetime(df_holidays['date'])
    df = pd.merge(df, df_holidays[['holidays_id', 'TIMESTAMP']], on=['TIMESTAMP'], how='left')
    del df_holidays
    df['holidays_id'].fillna(0, inplace=True)
    df['holidays_id'] = df['holidays_id'].astype(int)
    df.rename(columns={'holidays_id': 'HOLIDAYS_ID'}, inplace=True)

    ### define as special days those that are before and after holidays
    CLA_comm('define as special days those that are before and after holidays')

    # initialise variable of special evnet
    df['SPECIAL_EVENT'] = 0
    delta = timedelta(days=1)

    # iter over all holidays events
    for holiday_id in np.sort(df['HOLIDAYS_ID'].unique()):

        # for holidays not in easter period (3,4,5,6) or not holidays (0)
        if np.isin(holiday_id, [0, 3, 4, 5, 6], invert=True):

            # get the day of the week of the holidays event
            day_of_the_week = df[df['HOLIDAYS_ID'] == holiday_id]['WEEKDAY'].unique()[0]

            if day_of_the_week == 1:  # Tuesday
                # get date of the holidays event
                date_special_day = pd.to_datetime(df[df['HOLIDAYS_ID'] == holiday_id]['TIMESTAMP'].unique()[0])

                # Set Holidays return (HR) in tuesday
                df.loc[df['TIMESTAMP'] == date_special_day, ['SPECIAL_EVENT']] = 'HR'  # Tuesday

                # Set Holidays valley (HV) in monday, Saturday and sunday
                df.loc[df['TIMESTAMP'] == date_special_day - 1 * delta, ['SPECIAL_EVENT']] = 'HV'  # Monday
                df.loc[df['TIMESTAMP'] == date_special_day - 2 * delta, ['SPECIAL_EVENT']] = 'HV'  # Sunday
                df.loc[df['TIMESTAMP'] == date_special_day - 3 * delta, ['SPECIAL_EVENT']] = 'HV'  # Saturday

                # Set Holidays Departure (HD) four days before (Friday)
                df.loc[df['TIMESTAMP'] == date_special_day - 4 * delta, ['SPECIAL_EVENT']] = 'HD'  # Friday

            if day_of_the_week == 0:  # Monday
                date_special_day = pd.to_datetime(df[df['HOLIDAYS_ID'] == holiday_id]['TIMESTAMP'].unique()[0])
                df.loc[df['TIMESTAMP'] == date_special_day, ['SPECIAL_EVENT']] = 'HR'  # Monday
                df.loc[df['TIMESTAMP'] == date_special_day - 1 * delta, ['SPECIAL_EVENT']] = 'HV'  # Sunday
                df.loc[df['TIMESTAMP'] == date_special_day - 2 * delta, ['SPECIAL_EVENT']] = 'HV'  # Saturday
                df.loc[df['TIMESTAMP'] == date_special_day - 3 * delta, ['SPECIAL_EVENT']] = 'HD'  # Friday

            if day_of_the_week == 3:  # Thursday
                date_special_day = pd.to_datetime(df[df['HOLIDAYS_ID'] == holiday_id]['TIMESTAMP'].unique()[0])
                df.loc[df['TIMESTAMP'] == date_special_day, ['SPECIAL_EVENT']] = 'HV'  # Thursday
                df.loc[df['TIMESTAMP'] == date_special_day - 1 * delta, ['SPECIAL_EVENT']] = 'HD'  # Wednesday
                df.loc[df['TIMESTAMP'] == date_special_day + 1 * delta, ['SPECIAL_EVENT']] = 'HV'  # Friday
                df.loc[df['TIMESTAMP'] == date_special_day + 2 * delta, ['SPECIAL_EVENT']] = 'HV'  # Saturday
                df.loc[df['TIMESTAMP'] == date_special_day + 3 * delta, ['SPECIAL_EVENT']] = 'HR'  # Sunday

            if day_of_the_week == 4:  # Friday
                date_special_day = pd.to_datetime(df[df['HOLIDAYS_ID'] == holiday_id]['TIMESTAMP'].unique()[0])
                df.loc[df['TIMESTAMP'] == date_special_day, ['SPECIAL_EVENT']] = 'HV'  # Friday
                df.loc[df['TIMESTAMP'] == date_special_day - 1 * delta, ['SPECIAL_EVENT']] = 'HD'  # Thursday
                df.loc[df['TIMESTAMP'] == date_special_day + 1 * delta, ['SPECIAL_EVENT']] = 'HV'  # Friday
                df.loc[df['TIMESTAMP'] == date_special_day + 2 * delta, ['SPECIAL_EVENT']] = 'HV'  # Saturday
                df.loc[df['TIMESTAMP'] == date_special_day + 3 * delta, ['SPECIAL_EVENT']] = 'HR'  # Sunday

            if day_of_the_week == 5:  # Saturday
                date_special_day = pd.to_datetime(df[df['HOLIDAYS_ID'] == holiday_id]['TIMESTAMP'].unique()[0])
                df.loc[df['TIMESTAMP'] == date_special_day, ['SPECIAL_EVENT']] = 'HV'  # saturday
                df.loc[df['TIMESTAMP'] == date_special_day - 1 * delta, ['SPECIAL_EVENT']] = 'HD'  # Friday
                df.loc[df['TIMESTAMP'] == date_special_day + 1 * delta, ['SPECIAL_EVENT']] = 'HR'  # Sunday

            if day_of_the_week == 6:  # Sunday
                date_special_day = pd.to_datetime(df[df['HOLIDAYS_ID'] == holiday_id]['TIMESTAMP'].unique()[0])
                df.loc[df['TIMESTAMP'] == date_special_day, ['SPECIAL_EVENT']] = 'HR'  # Sunday
                df.loc[df['TIMESTAMP'] == date_special_day - 1 * delta, ['SPECIAL_EVENT']] = 'HV'  # Saturday
                df.loc[df['TIMESTAMP'] == date_special_day - 2 * delta, ['SPECIAL_EVENT']] = 'HR'  # Friday

        # easter period 3: good friday, 4: The day after Good Friday, 5:Easter Sunday, 6:Easter Monday
        if np.isin(holiday_id, [3, 4, 5, 6]):
            if holiday_id == 3:  # always friday Departure
                date_special_day = pd.to_datetime(df[df['HOLIDAYS_ID'] == holiday_id]['TIMESTAMP'].unique()[0])
                df.loc[df['TIMESTAMP'] == date_special_day, ['SPECIAL_EVENT']] = 'HD'

            if holiday_id == 4:  # always Saturday valley
                date_special_day = pd.to_datetime(df[df['HOLIDAYS_ID'] == holiday_id]['TIMESTAMP'].unique()[0])
                df.loc[df['TIMESTAMP'] == date_special_day, ['SPECIAL_EVENT']] = 'HV'

            if holiday_id == 5:  # always Sunday valley
                date_special_day = pd.to_datetime(df[df['HOLIDAYS_ID'] == holiday_id]['TIMESTAMP'].unique()[0])
                df.loc[df['TIMESTAMP'] == date_special_day, ['SPECIAL_EVENT']] = 'HV'

            if holiday_id == 5:  # always Monday return
                date_special_day = pd.to_datetime(df[df['HOLIDAYS_ID'] == holiday_id]['TIMESTAMP'].unique()[0])
                df.loc[df['TIMESTAMP'] == date_special_day, ['SPECIAL_EVENT']] = 'HR'

    df.loc[df['SPECIAL_EVENT'] == 0, ['SPECIAL_EVENT']] = 'NH'

    df_out = df

    return df_out


def flow_eq(speed_est, Oc, L, l, F, NL):
    """

    :param speed_est: speed estimation (km/h) from equation
    :param Oc: occupancy
    :param L: meanlength of the vehicle (4.4-5.5)
    :param l: lop length (2m tunnel, 1 road)
    :param F: correction factor (0.1-1.5)
    :param NL: number of lanes
    :return:
    """

    flow_formula = speed_est * (F * NL * 10 * Oc) / (L + l)

    return flow_formula


def config_plotly():
    """" TBD
    """

    import plotly.io as pio
    pio.renderers.default = "browser"


def config_pandas():
    """
    Allows to show all the columns of a dataframe in the console
    Limit pandas warnings
    """
    import pandas as pd
    import numpy as np
    pd.options.mode.chained_assignment = None  # default='warn'
    desired_width = 350
    np.set_printoptions(linewidth=desired_width)  # show dataframes in console
    pd.set_option('display.max_columns', 10)


def create_timestamp(row):
    """ Create a time stamp column from others columns with year, month, day and hour
        Use with apply
    """
    from dl_ia_cla_settings import CLA_comm
    import pandas as pd

    try:
        return pd.Timestamp(int(row['YEAR']), int(row['MONTH']), int(row['DAY']), int(row['HOUR']), int(row['MINUTE']))

    except Exception as exception_msg:
        CLA_comm('(!) Error in create_timestamp: ' + str(exception_msg))


def filter_by_std(df, variable, option):
    if option == 2:
        df_aux = df[(df[variable] < (df[variable].mean() + 2 * df[variable].std()))
                    & (df[variable] > (df[variable].mean() - 2 * df[variable].std()))]
    elif option == 1:
        df_aux = df[(df[variable] < (df[variable].mean() + df[variable].std()))
                    & (df[variable] > (df[variable].mean() - df[variable].std()))]

    print('Rows dropped:{} %'.format(round(100 * (1 - (len(df_aux) / len(df))), 3)))
    return df_aux


def dl_ia_cla_settings_check_folder(path_folder):
    """ check that exists a folder, and if not, create it
    :param path_folder: string with the path
    :return error: error code (0:good, 1:bad)
    """
    import os
    from dl_ia_cla_settings import CLA_comm
    error = 0
    try:
        if not os.path.isdir(path_folder):
            CLA_comm('Creating folder: {} '.format(path_folder))
            os.mkdir(path_folder)
    except Exception as exception_msg:
        CLA_comm('(!) Error in check_folder: ' + str(exception_msg))
        error = 1
        return error


def check_figures_folder(road):
    import os
    if not os.path.isdir('figures\\{}\\'.format(road)):
        print('{} Creating figures folder for {}'.format('-' * 20, road))
        os.mkdir('figures\\{}\\'.format(road))


def check_models_folder(road):
    import os
    if not os.path.isdir('models\\{}\\'.format(road)):
        print('{} Creating models folder for {}'.format('-' * 20, road))
        os.mkdir('models\\{}\\'.format(road))


def check_metrics_folder(road):
    import os

    if not os.path.isdir('metrics\\{}\\'.format(road)):
        print('{} Creating metrics folder for {}'.format('-' * 20, road))
        os.mkdir('metrics\\{}\\'.format(road))


def check_descriptive_statistics(df):
    """ calculate descriptive statiscs of a dataframe columns

    :param df: dataframe with columns of interest
    :return error: error code (0:ok, 1: something wrong)

    """
    error = 0
    from dl_ia_cla_settings import CLA_comm

    try:
        for variable in df.columns:
            print('variable:{}{}'.format(' ' * 2, variable))
            print('---------------')
            print('Mean Value:{}{}'.format(' ' * 2, round(df[variable].mean(), 2)))
            print('std Value:{}{}'.format(' ' * 3, round(df[variable].std(), 2)))
            print('Q3.:{}{}'.format(' ' * 9, round(df[variable].quantile(0.75), 2)))
            print('Max.:{}{}'.format(' ' * 8, round(df[variable].max(), 2)))
            print('Q2 :{}{}'.format(' ' * 2, round(df[variable].median(), 2)))
            print('Min.:{}{}'.format(' ' * 8, round(df[variable].min(), 2)))
            print('Q1.:{}{}'.format(' ' * 9, round(df[variable].quantile(0.25), 2)))
            print('IQR.:{}{}'.format(' ' * 8, round(df[variable].quantile(0.75) - df0[variable].quantile(0.25), 2)))
        return error

    except Exception as exception_msg:
        CLA_comm('(!) Error in check_descriptive_statistics: ' + str(exception_msg))
        error = 1
        return error


def memory_usage(df):
    """ Calculate and print the memory usage by the dataframe

    :param df:
    :return:
    """
    from dl_ia_cla_settings import CLA_comm

    CLA_comm('Data Frame Memory usage: {:2.2f} GB'.format(df.memory_usage(deep=True).sum() / 1000000000))
    CLA_comm('Data Frame Size: {} '.format(df.shape))


columns_APL = ['flow',
               'volume',
               'speed',
               'meanlength',
               'gap',
               'headway',
               'occupancy']

all_columns_APL = ['time',
                   'volume',
                   'occupancy',
                   'headway',
                   'meanlength',
                   'speed',
                   'name']

columns_APL = ['volume',
               'occupancy',
               'headway',
               'meanlength',
               'speed']

original_columns_APL = ['Name',
                        'Last_Update',
                        'Health_State',
                        'Availability_',
                        'Count_',
                        'Average_Occupancy',
                        'Average_Headway',
                        'Average_Length',
                        'Average_Speed',
                        'Count_Short',
                        'Count_Medium',
                        'Count_Long',
                        'Average_Speed_Short',
                        'Average_Speed_Medium',
                        'Average_Speed_Long']

original_columns_C7 = ['Name',
                       'Last_Update',
                       'Health_State',
                       'Availability_',
                       'Count_',
                       'Average_Occupancy',
                       'Average_Headway',
                       'Average_Length',
                       'Average_Speed',
                       'Count_Short',
                       'Count_Medium',
                       'Count_Long',
                       'Average_Speed_Short',
                       'Average_Speed_Medium',
                       'Average_Speed_Long']

columns_C7 = ['Volume',
              'Occupancy',
              'Headway',
              'Meanlength',
              'Speed']

all_columns_C7 = ['time',
                  'Volume',
                  'Occupancy',
                  'Headway',
                  'Meanlength',
                  'Speed',
                  'Name']

# LL = Left Lane, IL = Inner Left, CL = Central Lane, IR = Inner Right, RL = Right Lane, SL = Single Lane
lanes_dict_inv_APL = {'_L': 1,
                      'IL': 2,
                      '_C': 3,
                      'IR': 4,
                      '_R': 5,
                      '_S': 6}

lanes_dict_APL = {1: 'SL',  # Left Lane (Slow Lane)
                  2: 'IL',  # Interior Left
                  3: 'CL',  # Central Lane
                  4: 'IR',  # Interior Right
                  5: 'FL',  # Right Lane (Fast Lane)
                  6: 'L'}  # Single Lane

lanes_dict = {1: 'SL',  # Left Lane (Slow Lane)
              2: 'IL',  # Interior Left
              3: 'CL',  # Central Lane
              4: 'IR',  # Interior Right
              5: 'FL',  # Right Lane (Fast Lane)
              6: 'L'}  # Single Lane

""" C7 Lanes dict
"""
# (1: Left/slow lane (SL/L),  2:Right/fast lane (FL), 3: Central lane,)
# FL = Fast Lane, SL = Slow Lane, _L = Only Lane
# subsection --> lane
lanes_dict_C7 = {1: 2,
                 2: 1,
                 3: 2,
                 4: 1,
                 5: 2,
                 6: 1,
                 7: 2,
                 8: 1,
                 9: 2,
                 10: 1,
                 11: 2,
                 12: 1,
                 13: 2,
                 14: 1,
                 15: 2,
                 16: 1,
                 17: 2,
                 18: 1,
                 19: 2,
                 20: 1,
                 21: 2,
                 22: 1,
                 23: 2,
                 24: 1,
                 25: 1,
                 26: 2,
                 27: 1,
                 28: 2,
                 29: 1,
                 30: 2,
                 31: 1,
                 32: 2,
                 33: 1,
                 34: 2,
                 35: 1,
                 36: 1,
                 37: 1,
                 38: 1}

""" road topology C7
"""

road_topology_df_C7 = pd.DataFrame(columns=['sub_section_name', 'section', 'direction', 'lanes', 'sub_section',
                                            'prev_section', 'next_section'],
                                   data=[['AVI101_FL', 1, 3, 2, 1, 0, 2],
                                         ['AVI101_SL', 1, 3, 3, 2, 0, 2],
                                         ['AVI102_FL', 2, 3, 2, 3, 1, 3],
                                         ['AVI102_SL', 2, 3, 3, 4, 1, 3],
                                         ['AVI110_FL', 3, 3, 2, 5, 2, 4],
                                         ['AVI110_SL', 3, 3, 3, 6, 2, 4],
                                         ['AVI111_FL', 4, 3, 2, 7, 3, 5],
                                         ['AVI111_SL', 4, 3, 3, 8, 3, 5],
                                         ['AVI117_FL', 5, 3, 2, 9, 4, 6],
                                         ['AVI117_SL', 5, 3, 3, 10, 4, 6],
                                         ['AVI118_FL', 6, 3, 2, 11, 5, 7],
                                         ['AVI118_SL', 6, 3, 3, 12, 5, 7],
                                         ['AVI125_FL', 7, 3, 2, 13, 6, 8],
                                         ['AVI125_SL', 7, 3, 3, 14, 6, 8],
                                         ['AVI126_FL', 8, 3, 2, 15, 7, 9],
                                         ['AVI126_SL', 8, 3, 3, 16, 7, 9],
                                         ['AVI133_FL', 9, 3, 2, 17, 8, 10],
                                         ['AVI133_SL', 9, 3, 3, 18, 8, 10],
                                         ['AVI134_FL', 10, 3, 2, 19, 9, 11],
                                         ['AVI134_SL', 10, 3, 3, 20, 9, 11],
                                         ['AVI141_FL', 11, 3, 2, 21, 10, 12],
                                         ['AVI141_SL', 11, 3, 3, 22, 10, 12],
                                         ['AVI142_FL', 12, 3, 2, 23, 11, 13],
                                         ['AVI142_SL', 12, 3, 3, 24, 11, 13],
                                         ['AVI148_FL', 13, 3, 2, 25, 12, 14],
                                         ['AVI148_SL', 13, 3, 3, 26, 12, 14],
                                         ['AVI149_FL', 14, 3, 2, 27, 13, 15],
                                         ['AVI149_SL', 14, 3, 3, 28, 13, 15],
                                         ['AVI156_FL', 15, 3, 2, 29, 14, 16],
                                         ['AVI156_SL', 15, 3, 3, 30, 14, 16],
                                         ['AVI157_FL', 16, 3, 2, 31, 15, 17],
                                         ['AVI157_SL', 16, 3, 3, 32, 15, 17],
                                         ['AVI162_FL', 17, 3, 2, 33, 16, 18],
                                         ['AVI162_SL', 17, 3, 3, 34, 16, 18],
                                         ['AVI163_FL', 18, 3, 2, 35, 17, 19],
                                         ['AVI163_SL', 18, 3, 3, 36, 17, 19],
                                         ['AVI170_FL', 19, 3, 2, 37, 18, 20],
                                         ['AVI170_SL', 19, 3, 3, 38, 18, 20],
                                         ['AVI171_FL', 20, 3, 2, 39, 19, 21],
                                         ['AVI171_SL', 20, 3, 3, 40, 19, 21],
                                         ['AVI177_FL', 21, 3, 2, 41, 20, 22],
                                         ['AVI177_SL', 21, 3, 3, 42, 20, 22],
                                         ['AVI178_FL', 22, 3, 2, 43, 21, 0],
                                         ['AVI178_SL', 22, 3, 3, 44, 21, 0],
                                         ['AVI585_FL', 23, 4, 2, 45, 0, 24],
                                         ['AVI585_SL', 23, 4, 3, 46, 0, 24],
                                         ['AVI584_FL', 24, 4, 2, 47, 23, 25],
                                         ['AVI584_SL', 24, 4, 3, 48, 23, 25],
                                         ['AVI576_FL', 25, 4, 2, 49, 24, 26],
                                         ['AVI576_SL', 25, 4, 3, 50, 24, 26],
                                         ['AVI575_FL', 26, 4, 2, 51, 25, 27],
                                         ['AVI575_SL', 26, 4, 3, 52, 25, 27],
                                         ['AVI569_FL', 27, 4, 2, 53, 26, 28],
                                         ['AVI569_SL', 27, 4, 3, 54, 26, 28],
                                         ['AVI568_FL', 28, 4, 2, 55, 27, 29],
                                         ['AVI568_SL', 28, 4, 3, 56, 27, 29],
                                         ['AVI562_FL', 29, 4, 2, 57, 28, 30],
                                         ['AVI562_SL', 29, 4, 3, 58, 28, 30],
                                         ['AVI561_FL', 30, 4, 2, 59, 29, 31],
                                         ['AVI561_SL', 30, 4, 3, 60, 29, 31],
                                         ['AVI554_FL', 31, 4, 2, 61, 30, 32],
                                         ['AVI554_SL', 31, 4, 3, 62, 30, 32],
                                         ['AVI553_FL', 32, 4, 2, 63, 31, 33],
                                         ['AVI553_SL', 32, 4, 3, 64, 31, 33],
                                         ['AVI546_FL', 33, 4, 2, 65, 32, 34],
                                         ['AVI546_SL', 33, 4, 3, 66, 32, 34],
                                         ['AVI545_FL', 34, 4, 2, 67, 33, 35],
                                         ['AVI545_SL', 34, 4, 3, 68, 33, 35],
                                         ['AVI539_FL', 35, 4, 2, 69, 34, 36],
                                         ['AVI539_SL', 35, 4, 3, 70, 34, 36],
                                         ['AVI538_FL', 36, 4, 2, 71, 35, 37],
                                         ['AVI538_SL', 36, 4, 3, 72, 35, 37],
                                         ['AVI531_FL', 37, 4, 2, 73, 36, 38],
                                         ['AVI531_SL', 37, 4, 3, 74, 36, 38],
                                         ['AVI530_FL', 38, 4, 2, 75, 37, 39],
                                         ['AVI530_SL', 38, 4, 3, 76, 37, 39],
                                         ['AVI522_FL', 39, 4, 2, 77, 38, 40],
                                         ['AVI522_SL', 39, 4, 3, 78, 38, 40],
                                         ['AVI521_FL', 40, 4, 2, 79, 39, 41],
                                         ['AVI521_SL', 40, 4, 3, 80, 39, 41],
                                         ['AVI516_FL', 41, 4, 2, 81, 40, 42],
                                         ['AVI516_SL', 41, 4, 3, 82, 40, 42],
                                         ['AVI515_FL', 42, 4, 2, 83, 41, 43],
                                         ['AVI515_SL', 42, 4, 3, 84, 41, 43],
                                         ['AVI509_FL', 43, 4, 2, 85, 42, 44],
                                         ['AVI509_SL', 43, 4, 3, 86, 42, 44],
                                         ['AVI508_FL', 44, 4, 2, 87, 43, 45],
                                         ['AVI508_SL', 44, 4, 3, 88, 43, 45],
                                         ['AVI504_FL', 45, 4, 2, 89, 44, 46],
                                         ['AVI504_SL', 45, 4, 3, 90, 44, 46],
                                         ['AVI503_FL', 46, 4, 2, 91, 45, 47],
                                         ['AVI503_SL', 46, 4, 3, 92, 45, 47],
                                         ['AVT002_FL', 47, 3, 2, 93, 0, 1],
                                         ['AVT002_SL', 47, 3, 2, 94, 0, 1],
                                         ['AVI007_L', 48, 3, 1, 95, 0, 3],
                                         ['AVT020_L', 49, 3, 1, 96, 0, 50],
                                         ['AVI401_L', 50, 3, 1, 97, 49, 51],
                                         ['AVI402_L', 51, 3, 1, 98, 50, 52],
                                         ['AVI412_L', 52, 3, 1, 99, 51, 53],
                                         ['AVI414_L', 53, 3, 1, 100, 52, 11],
                                         ['AVT030_L', 54, 3, 1, 101, 22, 55],
                                         ['AVT034_FL', 55, 3, 2, 102, 54, 0],
                                         ['AVT034_SL', 55, 3, 3, 103, 54, 0],
                                         ['AVT031_FL', 56, 3, 2, 104, 22, 0],
                                         ['AVT031_SL', 56, 3, 3, 105, 22, 0],
                                         ['AVT037_L', 57, 4, 1, 106, 0, 58],
                                         ['AVI039_FL', 58, 4, 2, 107, 57, 23],
                                         ['AVT036_FL', 59, 4, 2, 108, 0, 60],
                                         ['AVT036_SL', 59, 4, 3, 109, 0, 60],
                                         ['AVI039_FL', 60, 4, 2, 110, 59, 23],
                                         ['AVT033_SL', 60, 4, 3, 111, 59, 23],
                                         ['AVT035_L', 61, 4, 1, 112, 0, 62],
                                         ['AVT032_FL', 62, 4, 2, 113, 61, 23],
                                         ['AVT032_SL', 62, 4, 3, 114, 61, 23],
                                         ['AVI805_L', 63, 4, 1, 115, 36, 64],
                                         ['AVI806_L', 64, 4, 1, 116, 63, 65],
                                         ['AVI802_FL', 65, 4, 2, 117, 64, 66],
                                         ['AVI802_SL', 65, 4, 3, 118, 64, 66],
                                         ['AVI801_FL', 66, 4, 2, 119, 65, 67],
                                         ['AVI801_SL', 66, 4, 3, 120, 65, 67],
                                         ['AVT021_FL', 67, 4, 2, 121, 66, 0],
                                         ['AVT021_SL', 67, 4, 3, 122, 66, 0],
                                         ['AVT004_L', 68, 4, 1, 123, 46, 0],
                                         ['AVT003_FL', 69, 4, 2, 124, 46, 0],
                                         ['AVT003_SL', 69, 4, 3, 125, 46, 0]
                                         ])

# Encode subsections C7
subsection_dict_C7 = {'AVI584_FL': 1,
                      'AVI584_SL': 2,
                      'AVI576_FL': 3,
                      'AVI576_SL': 4,
                      'AVI569_FL': 5,
                      'AVI569_SL': 6,
                      'AVI562_FL': 7,
                      'AVI562_SL': 8,
                      'AVI554_FL': 9,
                      'AVI554_SL': 10,
                      'AVI546_FL': 11,
                      'AVI546_SL': 12,
                      'AVI539_FL': 13,
                      'AVI539_SL': 14,
                      'AVI531_FL': 15,
                      'AVI531_SL': 16,
                      'AVI522_FL': 17,
                      'AVI522_SL': 18,
                      'AVI516_FL': 19,
                      'AVI516_SL': 20,
                      'AVI509_FL': 21,
                      'AVI509_SL': 22,
                      'AVI503_FL': 23,
                      'AVI503_SL': 24,
                      'AVI806_L': 25,
                      'AVI801_FL': 26,
                      'AVI801_SL': 27,
                      'AVT021_FL': 28,
                      'AVT021_SL': 29,
                      'AVI102_FL': 30,
                      'AVI102_SL': 31,
                      'AVI110_FL': 32,
                      'AVI110_SL': 33,
                      'AVI117_FL': 34,
                      'AVI117_SL': 35,
                      'AVT020_L': 36,
                      'AVI401_L': 37,
                      'AVI414_L': 38}

# Encode sections
section_dict_C7 = {'AVI584': 1,
                   'AVI576': 2,
                   'AVI569': 3,
                   'AVI562': 4,
                   'AVI554': 5,
                   'AVI546': 6,
                   'AVI539': 7,
                   'AVI531': 8,
                   'AVI522': 9,
                   'AVI516': 10,
                   'AVI509': 11,
                   'AVI503': 12,
                   'AVI806': 13,
                   'AVI801': 14,
                   'AVT021': 15,
                   'AVI102': 16,
                   'AVI110': 17,
                   'AVI117': 18,
                   'AVI125': 19,
                   'AVI133': 20,
                   'AVI141': 21,
                   'AVI148': 22,
                   'AVI156': 23,
                   'AVI162': 24,
                   'AVI170': 25,
                   'AVI177': 26,
                   'AVT020': 27,
                   'AVI401': 28,
                   'AVI414': 29}

lane_dict_inv_C7 = {1: 'L',
                    2: 'FL',
                    3: 'SL'}

lane_dict_C7 = {'_L': 1,
                'FL': 2,
                'SL': 3}  # FL = Fast Lane, SL = Slow Lane, _L = Only Lane

road_topology_df_APL = pd.DataFrame([['AVI203_R', 1, 3, 5, 1, 62, 2],
                                     ['AVI203_IR', 1, 3, 4, 2, 62, 2],
                                     ['AVI203_IL', 1, 3, 2, 3, 62, 2],
                                     ['AVI203_L', 1, 3, 1, 4, 62, 2],
                                     ['AVI204_R', 2, 3, 5, 5, 1, 3],
                                     ['AVI204_IR', 2, 3, 4, 6, 1, 3],
                                     ['AVI204_IL', 2, 3, 2, 7, 1, 3],
                                     ['AVI204_L', 2, 3, 1, 8, 1, 3],
                                     ['AVI210_R', 3, 3, 5, 9, 2, 4],
                                     ['AVI210_C', 3, 3, 3, 10, 2, 4],
                                     ['AVI210_L', 3, 3, 1, 11, 2, 4],
                                     ['AVI211_R', 4, 3, 5, 12, 3, 5],
                                     ['AVI211_C', 4, 3, 3, 13, 3, 5],
                                     ['AVI211_L', 4, 3, 1, 14, 3, 5],
                                     ['AVI219_R', 5, 3, 5, 15, 4, 6],
                                     ['AVI219_C', 5, 3, 3, 16, 4, 6],
                                     ['AVI219_L', 5, 3, 1, 17, 4, 6],
                                     ['AVI220_R', 6, 3, 5, 18, 5, 7],
                                     ['AVI220_C', 6, 3, 3, 19, 5, 7],
                                     ['AVI220_L', 6, 3, 1, 20, 5, 7],
                                     ['AVI227_R', 7, 3, 5, 21, 6, 8],
                                     ['AVI227_C', 7, 3, 3, 22, 6, 8],
                                     ['AVI227_L', 7, 3, 1, 23, 6, 8],
                                     ['AVI228_R', 8, 3, 5, 24, 7, 9],
                                     ['AVI228_C', 8, 3, 3, 25, 7, 9],
                                     ['AVI228_L', 8, 3, 1, 26, 7, 9],
                                     ['AVI235_R', 9, 3, 5, 27, 8, 10],
                                     ['AVI235_C', 9, 3, 3, 28, 8, 10],
                                     ['AVI235_L', 9, 3, 1, 29, 8, 10],
                                     ['AVI236_R', 10, 3, 5, 30, 9, 11],
                                     ['AVI236_C', 10, 3, 3, 31, 9, 11],
                                     ['AVI236_L', 10, 3, 1, 32, 9, 11],
                                     ['AVI243_R', 11, 1, 5, 33, 10, 12],
                                     ['AVI243_L', 11, 1, 1, 34, 10, 12],
                                     ['AVI244_R', 12, 1, 5, 35, 11, 13],
                                     ['AVI244_L', 12, 1, 1, 36, 11, 13],
                                     ['AVI250_R', 13, 1, 5, 37, 12, 14],
                                     ['AVI250_L', 13, 1, 1, 38, 12, 14],
                                     ['AVI251_R', 14, 1, 5, 39, 13, 15],
                                     ['AVI251_L', 14, 1, 1, 40, 13, 15],
                                     ['AVI258_R', 15, 1, 5, 41, 14, 16],
                                     ['AVI258_C', 15, 1, 3, 42, 14, 16],
                                     ['AVI258_L', 15, 1, 1, 43, 14, 16],
                                     ['AVI259_R', 16, 1, 5, 44, 15, 17],
                                     ['AVI259_C', 16, 1, 3, 45, 15, 17],
                                     ['AVI259_L', 16, 1, 1, 46, 15, 17],
                                     ['AVI265_R', 17, 1, 5, 47, 16, 18],
                                     ['AVI265_L', 17, 1, 1, 48, 16, 18],
                                     ['AVI266_R', 18, 1, 5, 49, 17, 19],
                                     ['AVI266_L', 18, 1, 1, 50, 17, 19],
                                     ['AVI272_R', 19, 1, 5, 51, 18, 20],
                                     ['AVI272_L', 19, 1, 1, 52, 18, 20],
                                     ['AVI273_R', 20, 1, 5, 53, 19, 21],
                                     ['AVI273_L', 20, 1, 1, 54, 19, 21],
                                     ['AVI279_R', 21, 1, 5, 55, 20, 22],
                                     ['AVI279_L', 21, 1, 1, 56, 20, 22],
                                     ['AVI280_R', 22, 1, 5, 57, 21, 93],
                                     ['AVI280_L', 22, 1, 1, 58, 21, 93],
                                     ['AVI608_R', 23, 4, 5, 59, 24, 59],
                                     ['AVI608_C', 23, 4, 3, 60, 24, 59],
                                     ['AVI608_L', 23, 4, 1, 61, 24, 59],
                                     ['AVI609_R', 24, 4, 5, 62, 25, 23],
                                     ['AVI609_C', 24, 4, 3, 63, 25, 23],
                                     ['AVI609_L', 24, 4, 1, 64, 25, 23],
                                     ['AVI616_R', 25, 4, 5, 65, 26, 24],
                                     ['AVI616_C', 25, 4, 3, 66, 26, 24],
                                     ['AVI616_L', 25, 4, 1, 67, 26, 24],
                                     ['AVI617_R', 26, 4, 5, 68, 27, 25],
                                     ['AVI617_C', 26, 4, 3, 69, 27, 25],
                                     ['AVI617_L', 26, 4, 1, 70, 27, 25],
                                     ['AVI623_R', 27, 4, 5, 71, 28, 26],
                                     ['AVI623_C', 27, 4, 3, 72, 28, 26],
                                     ['AVI623_L', 27, 4, 1, 73, 28, 26],
                                     ['AVI624_R', 28, 4, 5, 74, 29, 27],
                                     ['AVI624_C', 28, 4, 3, 75, 29, 27],
                                     ['AVI624_L', 28, 4, 1, 76, 29, 27],
                                     ['AVI630_R', 29, 4, 5, 77, 30, 28],
                                     ['AVI630_C', 29, 4, 3, 78, 30, 28],
                                     ['AVI630_L', 29, 4, 1, 79, 30, 28],
                                     ['AVI631_R', 30, 4, 5, 80, 31, 29],
                                     ['AVI631_C', 30, 4, 3, 81, 31, 29],
                                     ['AVI631_L', 30, 4, 1, 82, 31, 29],
                                     ['AVI637_R', 31, 4, 5, 83, 32, 30],
                                     ['AVI637_IR', 31, 4, 4, 84, 32, 30],
                                     ['AVI637_IL', 31, 4, 2, 85, 32, 30],
                                     ['AVI637_L', 31, 4, 1, 86, 32, 30],
                                     ['AVI638_R', 32, 4, 5, 87, 33, 31],
                                     ['AVI638_IR', 32, 4, 4, 88, 33, 31],
                                     ['AVI638_IL', 32, 4, 2, 89, 33, 31],
                                     ['AVI638_L', 32, 4, 1, 90, 33, 31],
                                     ['AVI644_R', 33, 4, 5, 91, 34, 32],
                                     ['AVI644_L', 33, 4, 1, 92, 34, 32],
                                     ['AVI645_R', 34, 4, 5, 93, 35, 33],
                                     ['AVI645_L', 34, 4, 1, 94, 35, 33],
                                     ['AVI651_R', 35, 4, 5, 95, 36, 34],
                                     ['AVI651_L', 35, 4, 1, 96, 36, 34],
                                     ['AVI652_R', 36, 4, 5, 97, 37, 35],
                                     ['AVI652_L', 36, 4, 1, 98, 37, 35],
                                     ['AVI659_R', 37, 4, 5, 99, 39, 36],
                                     ['AVI659_L', 37, 4, 1, 100, 39, 36],
                                     ['AVI661_R', 39, 2, 5, 101, 40, 37],
                                     ['AVI661_C', 39, 2, 3, 102, 40, 37],
                                     ['AVI661_L', 39, 2, 1, 103, 40, 37],
                                     ['AVI666_R', 40, 2, 5, 104, 41, 39],
                                     ['AVI666_L', 40, 2, 1, 105, 41, 39],
                                     ['AVI667_R', 41, 2, 5, 106, 42, 40],
                                     ['AVI667_L', 41, 2, 1, 107, 42, 40],
                                     ['AVI674_R', 42, 2, 5, 108, 43, 41],
                                     ['AVI674_L', 42, 2, 1, 109, 43, 41],
                                     ['AVI675_R', 43, 2, 5, 110, 44, 42],
                                     ['AVI675_L', 43, 2, 1, 111, 44, 42],
                                     ['AVI682_R', 44, 2, 5, 112, 45, 43],
                                     ['AVI682_L', 44, 2, 1, 113, 45, 43],
                                     ['AVI683_R', 45, 2, 5, 114, 46, 44],
                                     ['AVI683_L', 45, 2, 1, 115, 46, 44],
                                     ['AVI687_R', 46, 2, 5, 116, 47, 45],
                                     ['AVI687_C', 46, 2, 3, 117, 47, 45],
                                     ['AVI687_L', 46, 2, 1, 118, 47, 45],
                                     ['AVI072_R', 47, 2, 5, 119, 48, 46],
                                     ['AVI072_L', 47, 2, 1, 120, 48, 46],
                                     ['TVL041_R', 48, 2, 5, 121, 0, 47],
                                     ['TVL043_L', 48, 2, 1, 122, 0, 47],
                                     ['TVL045_S', 49, 2, 6, 123, 0, 48],
                                     ['TVL047_S', 50, 2, 6, 124, 0, 48],
                                     ['AVIO28_S', 51, 2, 6, 125, 39, 52],
                                     ['AVIO27_S', 52, 2, 6, 126, 51, 53],
                                     ['AVIO22_R', 53, 2, 5, 127, 52, 54],
                                     ['AVIO22_L', 53, 2, 1, 128, 52, 54],
                                     ['AVIO21_R', 54, 2, 5, 129, 53, 55],
                                     ['AVIO21_L', 54, 2, 1, 130, 53, 55],
                                     ['TVL019_S', 38, 3, 6, 131, 54, 0],
                                     ['TVL011_S', 55, 3, 6, 132, 54, 56],
                                     ['TVL013_S', 56, 3, 6, 133, 55, 0],
                                     ['TVLV03_S', 58, 4, 6, 134, 23, 0],
                                     ['TVLV01_S', 59, 4, 6, 135, 23, 0],
                                     ['TVLA15_S', 60, 4, 6, 136, 23, 0],
                                     ['AVIA22_R', 61, 3, 5, 137, 0, 97],
                                     ['AVIA22_L', 61, 3, 1, 138, 0, 97],
                                     ['TVLA03_R', 97, 3, 5, 139, 61, 1],
                                     ['TVLA01_L', 97, 3, 1, 140, 61, 1],
                                     ['TVLA09_R', 62, 3, 5, 141, 0, 1],
                                     ['TVLA07_L', 62, 3, 1, 142, 0, 1],
                                     ['AVIA01_R', 63, 3, 5, 143, 0, 64],
                                     ['AVIA01_L', 63, 3, 1, 144, 0, 64],
                                     ['TVLA05_S', 64, 3, 6, 145, 63, 1],
                                     ['AVIB40_R', 65, 3, 5, 146, 66, 10],
                                     ['AVIB40_L', 65, 3, 1, 147, 66, 10],
                                     ['AVIB41_R', 66, 3, 5, 148, 67, 65],
                                     ['AVIB41_L', 66, 3, 1, 149, 67, 65],
                                     ['AVIB46_R', 67, 3, 5, 150, 68, 66],
                                     ['AVIB46_L', 67, 3, 1, 151, 68, 66],
                                     ['AVIB47_R', 68, 3, 5, 152, 57, 67],
                                     ['AVIB47_L', 68, 3, 1, 153, 57, 67],
                                     ['TVL017_R', 57, 3, 5, 154, 68, 0],
                                     ['TVL015_L', 57, 3, 1, 155, 68, 0],
                                     ['TVLR21_R', 69, 4, 1, 156, 0, 75],
                                     ['AVI027_R', 70, 4, 5, 157, 0, 71],
                                     ['AVI027_C', 70, 4, 3, 158, 0, 71],
                                     ['AVI027_L', 70, 4, 1, 159, 0, 71],
                                     ['AVIQ21_R', 71, 4, 5, 160, 70, 72],
                                     ['AVIQ21_L', 71, 4, 1, 161, 70, 72],
                                     ['AVIQ22_R', 72, 4, 5, 162, 71, 73],
                                     ['AVIQ22_C', 72, 4, 3, 163, 71, 73],
                                     ['AVIQ22_L', 72, 4, 1, 164, 71, 73],
                                     ['TVLQ23_L', 73, 4, 1, 165, 72, 74],
                                     ['TVLR23_R', 74, 4, 5, 166, 73, 75],
                                     ['AVIR28_R', 75, 4, 5, 167, 74, 76],
                                     ['AVIR28_C', 75, 4, 3, 168, 74, 76],
                                     ['AVIR28_L', 75, 4, 1, 169, 74, 76],
                                     ['AVIR29_R', 76, 4, 5, 170, 75, 77],
                                     ['AVIR29_C', 76, 4, 3, 171, 75, 77],
                                     ['AVIR29_L', 76, 4, 1, 172, 75, 77],
                                     ['AVIR35_R', 77, 4, 5, 173, 76, 78],
                                     ['AVIR35_L', 77, 4, 1, 174, 76, 78],
                                     ['AVIR36_R', 78, 4, 5, 175, 77, 32],
                                     ['AVIR36_L', 78, 4, 1, 176, 77, 32],
                                     ['AVI053_R', 79, 4, 5, 177, 0, 80],
                                     ['AVI053_L', 79, 4, 1, 178, 0, 80],
                                     ['TVLQ21_S', 80, 1, 6, 179, 79, 81],
                                     ['AVID20_S', 81, 1, 6, 180, 80, 82],
                                     ['AVID21_S', 82, 1, 6, 181, 81, 83],
                                     ['AVIE27_R', 83, 1, 5, 182, 82, 84],
                                     ['AVIE27_L', 83, 1, 1, 183, 82, 84],
                                     ['AVIC31_R', 84, 1, 5, 184, 83, 85],
                                     ['AVIC31_L', 84, 1, 1, 185, 83, 85],
                                     ['AVIC32_S', 85, 1, 6, 186, 84, 15],
                                     ['TVLE23_S', 86, 1, 6, 187, 0, 83],
                                     ['TVL023_R', 87, 1, 5, 188, 0, 88],
                                     ['TVL021_L', 87, 1, 1, 189, 0, 88],
                                     ['AVIC19_S', 88, 1, 6, 190, 87, 89],
                                     ['AVIC20_S', 89, 1, 6, 191, 88, 90],
                                     ['AVIC25_S', 90, 1, 6, 192, 89, 91],
                                     ['AVIC26_S', 91, 1, 6, 193, 90, 83],
                                     ['TVL035_R', 92, 5, 6, 194, 22, 0],
                                     ['TVL033_C', 92, 3, 6, 195, 22, 0],
                                     ['TVL031_L', 92, 1, 6, 196, 22, 0],
                                     ['TVL039_R', 93, 1, 5, 197, 22, 0],
                                     ['TVL037_L', 93, 1, 1, 198, 22, 0],
                                     ['AVIV18_S', 94, 4, 6, 199, 25, 95],
                                     ['TVLV05_S', 95, 4, 6, 200, 94, 96],
                                     ['AVIV17_S', 96, 4, 6, 201, 95, 0]]
                                    , columns=['sub_section_name', 'section', 'direction', 'lane', 'sub_section',
                                               'prev_section', 'next_section'])
road_topology_df_APL.set_index(['sub_section_name'], inplace=True)

topology_dictionary_LW = {'LIU1001_01': 'Lane 1 Entry EB Subsection 1',
                          'LIU1001_02': 'Lane 2 Entry EB Subsection 2',
                          'LIU1001_03': 'Lane 1 Exit WB Subsection 7',
                          'LIU1001_04': 'Lane 2 Exit WB Subsection 8',
                          'LIU2001_01': 'Lane 1 Entry WB Subsection 5',
                          'LIU2001_02': 'Lane 2 Entry WB Subsection 6',
                          'LIU2001_03': 'Lane 1 Exit EB Subsection 3',
                          'LIU2001_04': 'Lane 2 Exit EB Subsection 2'}

# segmetns like lanes
segments_dict_LW = {1: 'Lane:SL Direction:EB (Entry)',
                    2: 'Lane:FL Direction:EB (Entry)',
                    3: 'Lane:SL Direction:EB (Exit)',
                    4: 'Lane:FL Direction:EB (Exit)',
                    5: 'Lane:SL Direction:WB (Entry)',
                    6: 'Lane:FL Direction:WB (Entry)',
                    7: 'Lane:SL Direction:WB (Exit)',
                    8: 'Lane:FL Direction:WB (Exit)'}

sections_dict_LW = {1: 'Entry-EB-CS1001',
                    2: 'Exit-EB-CS1002',
                    3: 'Entry-WB-CS2001',
                    4: 'Exit-WB-CS2002'}  #

df_aux = pd.DataFrame.from_dict(sections_dict_LW, orient='index')
df_aux.reset_index(inplace=True)
df_aux.rename(columns={'index': 'segment', 0: 'desc'}, inplace=True)
# df_aux.to_csv('data\\LW_sections_dict.csv', index=False)

# sections_dict = pd.read_csv('data\\LW_sections_dict.csv').set_index(['segment'], drop=True).to_dict()['desc']

# sections_dict_inv = pd.read_csv('data\\LW_sections_dict.csv').set_index(['desc'], drop=True).to_dict()['segment']

original_columns_rfa = ['log_time',
                        'detector_id',
                        'detector_id-name',
                        'occupancy',
                        'norm_occupancy',
                        'volume',
                        'norm_volume',
                        'speed_obs',
                        'speed_sum',
                        'configuration_id',
                        'configuration_id-name',
                        'available',
                        'incident',
                        'failed',
                        'length_normalised',
                        'distance_normalised',
                        'last_modified']


def check_values_variables(df):
    drop_ = False
    sample_ = False

    index_1 = df[(df['volume'] == 0) & (df['speed'] != 0)].index
    index_2 = df[(df['volume'] == 0) & (df['occupancy'] != 0)].index
    index_3 = df[(df['volume'] == 0) & (df['flow'] != 0)].index

    index_4 = df[(df['flow'] == 0) & (df['speed'] != 0)].index
    index_5 = df[(df['flow'] == 0) & (df['occupancy'] != 0)].index
    index_6 = df[(df['flow'] == 0) & (df['volume'] != 0)].index

    index_7 = df[(df['occupancy'] == 0) & (df['volume'] != 0)].index
    index_8 = df[(df['occupancy'] == 0) & (df['speed'] != 0)].index
    index_9 = df[(df['occupancy'] == 0) & (df['flow'] != 0)].index

    index_10 = df[(df['speed'] == 0) & (df['occupancy'] != 0)].index
    index_11 = df[(df['speed'] == 0) & (df['flow'] != 0)].index
    index_12 = df[(df['speed'] == 0) & (df['volume'] != 0)].index

    # index_13 = df[(df['meanlength'] == 0) & (df['flow'] != 0)].index
    # index_14 = df[(df['meanlength'] == 0) & (df['speed'] != 0)].index
    # index_15 = df[(df['meanlength'] == 0) & (df['volume'] != 0)].index

    index_16 = df[(df['volume'] < 0)].index
    index_17 = df[(df['speed'] < 0)].index
    index_18 = df[(df['flow'] < 0)].index
    # index_19 = df[(df['meanlength'] < 0)].index

    print('\n')
    print('{} Check Values'.format('-' * 10))
    print('\n')
    print('#1 (volume=0 & speed!=0):{}{}'.format(' ' * 8, len(index_1)))
    print('#2 (volume=0 & occupancy!=0):{}{}'.format(' ' * 4, len(index_2)))
    print('#3 (volume=0 & flow!=0):{}{}'.format(' ' * 9, len(index_3)))
    print('#4 (flow=0 & speed!=0):{}{}'.format(' ' * 10, len(index_4)))
    print('#5 (flow=0 & occupancy!=0):{}{}'.format(' ' * 6, len(index_5)))
    print('#6 (flow=0 & volume!=0):{}{}'.format(' ' * 9, len(index_6)))
    print('#7 (occupancy=0 & volume!=0):{}{}'.format(' ' * 4, len(index_7)))
    print('#8 (occupancy=0 & speed!=0):{}{}'.format(' ' * 5, len(index_8)))
    print('#9 (occupancy=0 & flow!=0):{}{}'.format(' ' * 6, len(index_9)))
    print('#10 (speed=0 & occupancy!=0):{}{}'.format(' ' * 4, len(index_10)))
    print('#11 (speed=0 & flow!=0):{}{}'.format(' ' * 9, len(index_11)))
    print('#12 (speed=0 & volume!=0):{}{}'.format(' ' * 7, len(index_12)))
    # print('#13 (meanlength=0 & flow!=0):{}{}'.format(' ' * 4, len(index_13)))
    # print('#14 (meanlength=0 & speed!=0):{}{}'.format(' ' * 3, len(index_14)))
    # print('#15 (meanlength=0 & volume!=0):{}{}'.format(' ' * 2, len(index_15)))
    print('#16 (volume<0):{}{}'.format(' ' * 18, len(index_16)))
    print('#17 (speed<0):{}{}'.format(' ' * 19, len(index_17)))
    print('#18 (flow<0):{}{}'.format(' ' * 20, len(index_18)))
    # print('#19 (meanlength<0):{}{}'.format(' ' * 20, len(index_19)))
    print('\n')


dict_lanes = {1: 'Left/Slow Lane',
              2: 'Right/fast Lane',
              3: 'Central Lane'}

dict_days = {'Monday': 0,
             'Tuesday': 1,
             'Wednesday': 2,
             'Thurday': 3,
             'Friday': 4,
             'Saturday': 5,
             'Sunday': 6}

days_dict = {0: 'Monday',
             1: 'Tuesday',
             2: 'Wednesday',
             3: 'Thurday',
             4: 'Friday',
             5: 'Saturday',
             6: 'Sunday'}

direction_dict = {'Eastbound': 1,
                  'Westbound': 2,
                  'Northbound': 3,
                  'Southbound': 4}

dict_direction = {1: 'EB',
                  2: 'WB',
                  3: 'NB',
                  4: 'SB'}

dict_season = {1: 'summer',
               2: 'Autumn',
               3: 'Winter',
               4: 'Spring'}

algorithms_ = ['tbats',
               'ets',
               'sarima',
               'fbprophet',
               'desc_ref',
               'extra_trees_2',
               'random_forest',
               'mlp_neural_network',
               'gradient_boosting']

algorithms = {2: 'TBATS (Time series)',  # time series
              3: 'ETS (Time series)',  # time series
              4: 'SARIMA (Time series)',  # time series
              5: 'fbprophet (Time series)',  # time series
              6: 'Descriptive',  # Descriptive
              8: 'Extra trees (Ensemble)',  # Extra_trees_2
              10: 'Random Forest (Ensemble)',  # Ensemble
              11: 'Neural Network',  # Neural Network
              12: 'Gradient Boosting (Ensemble)'}  # Ensemble

horizon_dict = {'PREDICCION_INT 15': 'Horizon 15 min',
                'PREDICCION_INT 60': 'Horizon 60 min',
                'PREDICCION_INT 120': 'Horizon 120 min'}

lanes_dictionary = {'SL': 1,
                    'FL': 2,
                    'CL': 3,
                    'IL': 4,
                    'IR': 5}

lanes_dictionary_inv = {1: 'SL',
                        2: 'FL',
                        3: 'CL',
                        4: 'IL',
                        5: 'IR'}

road1 = """
    |       |      
    |       |       
    |       | 
    |       |       
    |       |       
"""
road2 = """
    |       |       |
    |       |       |
    |       | (70)  |
    |       |       |
    |       |       |
"""
road3 = """
    |       |       |       |
    |       |       |       |   
    |       |       | (70)  |
    |       |       |       |
    |       |       |       |
"""
road4 = """
    |       |       |       |       |                
    |       |       |       |       |
    |       |       |       | (70)  |
    |       |       |       |       |
    |       |       |       |       |
"""
road5 = """
    |       |       |       |       |       |                
    |       |       |       |       |       |
    |       |       |       |       | (70)  |
    |       |       |       |       |       |
    |       |       |       |       |       | 
"""

legend_lanes = """
{} lanes legend 
{} (SL)-Slow Lane/Left Lane
{} (FL)-Fast Lane/Right Lane
{} (CL)-Central Lane
{} (IL)-Interior Left
{} (IR)-Interior Right
""".format('-' * 20, ' ' * 20, ' ' * 20, ' ' * 20, ' ' * 20, ' ' * 20)

if 1 == 0:
    """ ============= DESCRIPTION ===============
        This script defines global variables, paths and settings
    """
    import pandas as pd

    # Path with raw data
    read_path = 'D:/CLARENCE/APL_TraffZoneDef/TraffZoneDef/'

    # Path with resampled data
    write_path = 'D:/CLARENCE/APL_TraffZoneDef_agrupados_ponderados/'
    data_path = write_path

    images_path = 'D:/CLARENCE/figures/'

    data_path_analytics = 'D:\\dataanalytics\\dataanalytics.predictive\\data\\'

    models_path = 'D:/CLARENCE/models/APL'

    extension = 'csv'

    road = 'Airport Link'

    original_columns = ['Name', 'Last_Update', 'Health_State', 'Availability_',
                        'Count_', 'Average_Occupancy', 'Average_Headway',
                        'Average_Length', 'Average_Speed', 'Count_Short',
                        'Count_Medium', 'Count_Long', 'Average_Speed_Short',
                        'Average_Speed_Medium', 'Average_Speed_Long']

    columns_ = ['volume', 'occupancy',
                'headway', 'meanlength',
                'speed']

    all_columns_ = ['time', 'volume', 'occupancy',
                    'headway', 'meanlength',
                    'speed',
                    'name']

    # LL = Left Lane, IL = Inner Left, CL = Central Lane, IR = Inner Right, RL = Right Lane, SL = Single Lane
    lane_dict = {'_L': 1, 'IL': 2, '_C': 3, 'IR': 4, '_R': 5, '_S': 6}
    direction_dict = {'Eastbound': 1, 'Westbound': 2, 'Northbound': 3, 'Southbound': 4}

    road_topology_df = pd.DataFrame([['AVI203_R', 1, 3, 5, 1, 62, 2],
                                     ['AVI203_IR', 1, 3, 4, 2, 62, 2],
                                     ['AVI203_IL', 1, 3, 2, 3, 62, 2],
                                     ['AVI203_L', 1, 3, 1, 4, 62, 2],
                                     ['AVI204_R', 2, 3, 5, 5, 1, 3],
                                     ['AVI204_IR', 2, 3, 4, 6, 1, 3],
                                     ['AVI204_IL', 2, 3, 2, 7, 1, 3],
                                     ['AVI204_L', 2, 3, 1, 8, 1, 3],
                                     ['AVI210_R', 3, 3, 5, 9, 2, 4],
                                     ['AVI210_C', 3, 3, 3, 10, 2, 4],
                                     ['AVI210_L', 3, 3, 1, 11, 2, 4],
                                     ['AVI211_R', 4, 3, 5, 12, 3, 5],
                                     ['AVI211_C', 4, 3, 3, 13, 3, 5],
                                     ['AVI211_L', 4, 3, 1, 14, 3, 5],
                                     ['AVI219_R', 5, 3, 5, 15, 4, 6],
                                     ['AVI219_C', 5, 3, 3, 16, 4, 6],
                                     ['AVI219_L', 5, 3, 1, 17, 4, 6],
                                     ['AVI220_R', 6, 3, 5, 18, 5, 7],
                                     ['AVI220_C', 6, 3, 3, 19, 5, 7],
                                     ['AVI220_L', 6, 3, 1, 20, 5, 7],
                                     ['AVI227_R', 7, 3, 5, 21, 6, 8],
                                     ['AVI227_C', 7, 3, 3, 22, 6, 8],
                                     ['AVI227_L', 7, 3, 1, 23, 6, 8],
                                     ['AVI228_R', 8, 3, 5, 24, 7, 9],
                                     ['AVI228_C', 8, 3, 3, 25, 7, 9],
                                     ['AVI228_L', 8, 3, 1, 26, 7, 9],
                                     ['AVI235_R', 9, 3, 5, 27, 8, 10],
                                     ['AVI235_C', 9, 3, 3, 28, 8, 10],
                                     ['AVI235_L', 9, 3, 1, 29, 8, 10],
                                     ['AVI236_R', 10, 3, 5, 30, 9, 11],
                                     ['AVI236_C', 10, 3, 3, 31, 9, 11],
                                     ['AVI236_L', 10, 3, 1, 32, 9, 11],
                                     ['AVI243_R', 11, 1, 5, 33, 10, 12],
                                     ['AVI243_L', 11, 1, 1, 34, 10, 12],
                                     ['AVI244_R', 12, 1, 5, 35, 11, 13],
                                     ['AVI244_L', 12, 1, 1, 36, 11, 13],
                                     ['AVI250_R', 13, 1, 5, 37, 12, 14],
                                     ['AVI250_L', 13, 1, 1, 38, 12, 14],
                                     ['AVI251_R', 14, 1, 5, 39, 13, 15],
                                     ['AVI251_L', 14, 1, 1, 40, 13, 15],
                                     ['AVI258_R', 15, 1, 5, 41, 14, 16],
                                     ['AVI258_C', 15, 1, 3, 42, 14, 16],
                                     ['AVI258_L', 15, 1, 1, 43, 14, 16],
                                     ['AVI259_R', 16, 1, 5, 44, 15, 17],
                                     ['AVI259_C', 16, 1, 3, 45, 15, 17],
                                     ['AVI259_L', 16, 1, 1, 46, 15, 17],
                                     ['AVI265_R', 17, 1, 5, 47, 16, 18],
                                     ['AVI265_L', 17, 1, 1, 48, 16, 18],
                                     ['AVI266_R', 18, 1, 5, 49, 17, 19],
                                     ['AVI266_L', 18, 1, 1, 50, 17, 19],
                                     ['AVI272_R', 19, 1, 5, 51, 18, 20],
                                     ['AVI272_L', 19, 1, 1, 52, 18, 20],
                                     ['AVI273_R', 20, 1, 5, 53, 19, 21],
                                     ['AVI273_L', 20, 1, 1, 54, 19, 21],
                                     ['AVI279_R', 21, 1, 5, 55, 20, 22],
                                     ['AVI279_L', 21, 1, 1, 56, 20, 22],
                                     ['AVI280_R', 22, 1, 5, 57, 21, 93],
                                     ['AVI280_L', 22, 1, 1, 58, 21, 93],
                                     ['AVI608_R', 23, 4, 5, 59, 24, 59],
                                     ['AVI608_C', 23, 4, 3, 60, 24, 59],
                                     ['AVI608_L', 23, 4, 1, 61, 24, 59],
                                     ['AVI609_R', 24, 4, 5, 62, 25, 23],
                                     ['AVI609_C', 24, 4, 3, 63, 25, 23],
                                     ['AVI609_L', 24, 4, 1, 64, 25, 23],
                                     ['AVI616_R', 25, 4, 5, 65, 26, 24],
                                     ['AVI616_C', 25, 4, 3, 66, 26, 24],
                                     ['AVI616_L', 25, 4, 1, 67, 26, 24],
                                     ['AVI617_R', 26, 4, 5, 68, 27, 25],
                                     ['AVI617_C', 26, 4, 3, 69, 27, 25],
                                     ['AVI617_L', 26, 4, 1, 70, 27, 25],
                                     ['AVI623_R', 27, 4, 5, 71, 28, 26],
                                     ['AVI623_C', 27, 4, 3, 72, 28, 26],
                                     ['AVI623_L', 27, 4, 1, 73, 28, 26],
                                     ['AVI624_R', 28, 4, 5, 74, 29, 27],
                                     ['AVI624_C', 28, 4, 3, 75, 29, 27],
                                     ['AVI624_L', 28, 4, 1, 76, 29, 27],
                                     ['AVI630_R', 29, 4, 5, 77, 30, 28],
                                     ['AVI630_C', 29, 4, 3, 78, 30, 28],
                                     ['AVI630_L', 29, 4, 1, 79, 30, 28],
                                     ['AVI631_R', 30, 4, 5, 80, 31, 29],
                                     ['AVI631_C', 30, 4, 3, 81, 31, 29],
                                     ['AVI631_L', 30, 4, 1, 82, 31, 29],
                                     ['AVI637_R', 31, 4, 5, 83, 32, 30],
                                     ['AVI637_IR', 31, 4, 4, 84, 32, 30],
                                     ['AVI637_IL', 31, 4, 2, 85, 32, 30],
                                     ['AVI637_L', 31, 4, 1, 86, 32, 30],
                                     ['AVI638_R', 32, 4, 5, 87, 33, 31],
                                     ['AVI638_IR', 32, 4, 4, 88, 33, 31],
                                     ['AVI638_IL', 32, 4, 2, 89, 33, 31],
                                     ['AVI638_L', 32, 4, 1, 90, 33, 31],
                                     ['AVI644_R', 33, 4, 5, 91, 34, 32],
                                     ['AVI644_L', 33, 4, 1, 92, 34, 32],
                                     ['AVI645_R', 34, 4, 5, 93, 35, 33],
                                     ['AVI645_L', 34, 4, 1, 94, 35, 33],
                                     ['AVI651_R', 35, 4, 5, 95, 36, 34],
                                     ['AVI651_L', 35, 4, 1, 96, 36, 34],
                                     ['AVI652_R', 36, 4, 5, 97, 37, 35],
                                     ['AVI652_L', 36, 4, 1, 98, 37, 35],
                                     ['AVI659_R', 37, 4, 5, 99, 39, 36],
                                     ['AVI659_L', 37, 4, 1, 100, 39, 36],
                                     ['AVI661_R', 39, 2, 5, 101, 40, 37],
                                     ['AVI661_C', 39, 2, 3, 102, 40, 37],
                                     ['AVI661_L', 39, 2, 1, 103, 40, 37],
                                     ['AVI666_R', 40, 2, 5, 104, 41, 39],
                                     ['AVI666_L', 40, 2, 1, 105, 41, 39],
                                     ['AVI667_R', 41, 2, 5, 106, 42, 40],
                                     ['AVI667_L', 41, 2, 1, 107, 42, 40],
                                     ['AVI674_R', 42, 2, 5, 108, 43, 41],
                                     ['AVI674_L', 42, 2, 1, 109, 43, 41],
                                     ['AVI675_R', 43, 2, 5, 110, 44, 42],
                                     ['AVI675_L', 43, 2, 1, 111, 44, 42],
                                     ['AVI682_R', 44, 2, 5, 112, 45, 43],
                                     ['AVI682_L', 44, 2, 1, 113, 45, 43],
                                     ['AVI683_R', 45, 2, 5, 114, 46, 44],
                                     ['AVI683_L', 45, 2, 1, 115, 46, 44],
                                     ['AVI687_R', 46, 2, 5, 116, 47, 45],
                                     ['AVI687_C', 46, 2, 3, 117, 47, 45],
                                     ['AVI687_L', 46, 2, 1, 118, 47, 45],
                                     ['AVI072_R', 47, 2, 5, 119, 48, 46],
                                     ['AVI072_L', 47, 2, 1, 120, 48, 46],
                                     ['TVL041_R', 48, 2, 5, 121, 0, 47],
                                     ['TVL043_L', 48, 2, 1, 122, 0, 47],
                                     ['TVL045_S', 49, 2, 6, 123, 0, 48],
                                     ['TVL047_S', 50, 2, 6, 124, 0, 48],
                                     ['AVIO28_S', 51, 2, 6, 125, 39, 52],
                                     ['AVIO27_S', 52, 2, 6, 126, 51, 53],
                                     ['AVIO22_R', 53, 2, 5, 127, 52, 54],
                                     ['AVIO22_L', 53, 2, 1, 128, 52, 54],
                                     ['AVIO21_R', 54, 2, 5, 129, 53, 55],
                                     ['AVIO21_L', 54, 2, 1, 130, 53, 55],
                                     ['TVL019_S', 38, 3, 6, 131, 54, 0],
                                     ['TVL011_S', 55, 3, 6, 132, 54, 56],
                                     ['TVL013_S', 56, 3, 6, 133, 55, 0],
                                     ['TVLV03_S', 58, 4, 6, 134, 23, 0],
                                     ['TVLV01_S', 59, 4, 6, 135, 23, 0],
                                     ['TVLA15_S', 60, 4, 6, 136, 23, 0],
                                     ['AVIA22_R', 61, 3, 5, 137, 0, 97],
                                     ['AVIA22_L', 61, 3, 1, 138, 0, 97],
                                     ['TVLA03_R', 97, 3, 5, 139, 61, 1],
                                     ['TVLA01_L', 97, 3, 1, 140, 61, 1],
                                     ['TVLA09_R', 62, 3, 5, 141, 0, 1],
                                     ['TVLA07_L', 62, 3, 1, 142, 0, 1],
                                     ['AVIA01_R', 63, 3, 5, 143, 0, 64],
                                     ['AVIA01_L', 63, 3, 1, 144, 0, 64],
                                     ['TVLA05_S', 64, 3, 6, 145, 63, 1],
                                     ['AVIB40_R', 65, 3, 5, 146, 66, 10],
                                     ['AVIB40_L', 65, 3, 1, 147, 66, 10],
                                     ['AVIB41_R', 66, 3, 5, 148, 67, 65],
                                     ['AVIB41_L', 66, 3, 1, 149, 67, 65],
                                     ['AVIB46_R', 67, 3, 5, 150, 68, 66],
                                     ['AVIB46_L', 67, 3, 1, 151, 68, 66],
                                     ['AVIB47_R', 68, 3, 5, 152, 57, 67],
                                     ['AVIB47_L', 68, 3, 1, 153, 57, 67],
                                     ['TVL017_R', 57, 3, 5, 154, 68, 0],
                                     ['TVL015_L', 57, 3, 1, 155, 68, 0],
                                     ['TVLR21_R', 69, 4, 1, 156, 0, 75],
                                     ['AVI027_R', 70, 4, 5, 157, 0, 71],
                                     ['AVI027_C', 70, 4, 3, 158, 0, 71],
                                     ['AVI027_L', 70, 4, 1, 159, 0, 71],
                                     ['AVIQ21_R', 71, 4, 5, 160, 70, 72],
                                     ['AVIQ21_L', 71, 4, 1, 161, 70, 72],
                                     ['AVIQ22_R', 72, 4, 5, 162, 71, 73],
                                     ['AVIQ22_C', 72, 4, 3, 163, 71, 73],
                                     ['AVIQ22_L', 72, 4, 1, 164, 71, 73],
                                     ['TVLQ23_L', 73, 4, 1, 165, 72, 74],
                                     ['TVLR23_R', 74, 4, 5, 166, 73, 75],
                                     ['AVIR28_R', 75, 4, 5, 167, 74, 76],
                                     ['AVIR28_C', 75, 4, 3, 168, 74, 76],
                                     ['AVIR28_L', 75, 4, 1, 169, 74, 76],
                                     ['AVIR29_R', 76, 4, 5, 170, 75, 77],
                                     ['AVIR29_C', 76, 4, 3, 171, 75, 77],
                                     ['AVIR29_L', 76, 4, 1, 172, 75, 77],
                                     ['AVIR35_R', 77, 4, 5, 173, 76, 78],
                                     ['AVIR35_L', 77, 4, 1, 174, 76, 78],
                                     ['AVIR36_R', 78, 4, 5, 175, 77, 32],
                                     ['AVIR36_L', 78, 4, 1, 176, 77, 32],
                                     ['AVI053_R', 79, 4, 5, 177, 0, 80],
                                     ['AVI053_L', 79, 4, 1, 178, 0, 80],
                                     ['TVLQ21_S', 80, 1, 6, 179, 79, 81],
                                     ['AVID20_S', 81, 1, 6, 180, 80, 82],
                                     ['AVID21_S', 82, 1, 6, 181, 81, 83],
                                     ['AVIE27_R', 83, 1, 5, 182, 82, 84],
                                     ['AVIE27_L', 83, 1, 1, 183, 82, 84],
                                     ['AVIC31_R', 84, 1, 5, 184, 83, 85],
                                     ['AVIC31_L', 84, 1, 1, 185, 83, 85],
                                     ['AVIC32_S', 85, 1, 6, 186, 84, 15],
                                     ['TVLE23_S', 86, 1, 6, 187, 0, 83],
                                     ['TVL023_R', 87, 1, 5, 188, 0, 88],
                                     ['TVL021_L', 87, 1, 1, 189, 0, 88],
                                     ['AVIC19_S', 88, 1, 6, 190, 87, 89],
                                     ['AVIC20_S', 89, 1, 6, 191, 88, 90],
                                     ['AVIC25_S', 90, 1, 6, 192, 89, 91],
                                     ['AVIC26_S', 91, 1, 6, 193, 90, 83],
                                     ['TVL035_R', 92, 5, 6, 194, 22, 0],
                                     ['TVL033_C', 92, 3, 6, 195, 22, 0],
                                     ['TVL031_L', 92, 1, 6, 196, 22, 0],
                                     ['TVL039_R', 93, 1, 5, 197, 22, 0],
                                     ['TVL037_L', 93, 1, 1, 198, 22, 0],
                                     ['AVIV18_S', 94, 4, 6, 199, 25, 95],
                                     ['TVLV05_S', 95, 4, 6, 200, 94, 96],
                                     ['AVIV17_S', 96, 4, 6, 201, 95, 0]]
                                    , columns=['sub_section_name', 'section', 'direction', 'lanes', 'sub_section',
                                               'prev_section', 'next_section'])
    road_topology_df.set_index(['sub_section_name'], inplace=True)

    holyday_list = [['New Years Day', '2017-01-01', 1],
                    ['Australia Day', '2017-01-26', 2],
                    ['Good Friday', '2017-04-14', 3],
                    ['The day after Good Friday', '2017-04-15', 4],
                    ['Easter Sunday', '2017-04-16', 5],
                    ['Easter Monday', '2017-04-17', 6],
                    ['Aznac Day', '2017-04-25', 7],
                    ['Labour Day', '2017-05-01', 8],
                    ['Royal Queensland Show', '2017-08-16', 9],
                    ['Queens Birthday', '2017-10-2', 10],
                    ['Christmas Eve', '2017-12-24', 11],
                    ['Christmas Day', '2017-12-25', 12],
                    ['Boxing Day', '2017-12-26', 13],
                    ['New Years Day', '2018-01-01', 1],
                    ['Australia Day', '2018-01-26', 2],
                    ['Good Friday', '2018-03-30', 3],
                    ['The day after Good Friday', '2018-03-31', 4],
                    ['Easter Sunday', '2018-04-01', 5],
                    ['Easter Monday', '2018-04-02', 6],
                    ['Aznac Day', '2018-04-25', 7],
                    ['Labour Day', '2018-05-07', 8],
                    ['Royal Queensland Show', '2018-08-15', 9],
                    ['Queens Birthday', '2018-10-1', 10],
                    ['Christmas Eve', '2018-12-24', 11],
                    ['Christmas Day', '2018-12-25', 12],
                    ['Boxing Day', '2018-12-26', 13],
                    ['New Years Day', '2019-01-01', 1],
                    ['Australia Day', '2019-01-28', 2],
                    ['Good Friday', '2019-04-19', 3],
                    ['The day after Good Friday', '2019-04-20', 4],
                    ['Easter Sunday', '2019-04-21', 5],
                    ['Easter Monday', '2019-04-22', 6],
                    ['Aznac Day', '2019-04-25', 7],
                    ['Labour Day', '2019-05-06', 8],
                    ['Royal Queensland Show', '2019-08-14', 9],
                    ['Queens Birthday', '2019-10-7', 10],
                    ['Christmas Eve', '2019-12-24', 11],
                    ['Christmas Day', '2019-12-25', 12],
                    ['Boxing Day', '2019-12-26', 13]]
    holyday_df = pd.DataFrame(holyday_list, columns=['holyday_name', 'holyday_date', 'holyday_id'])
    holyday_df['holyday_date'] = pd.to_datetime(holyday_df['holyday_date']).dt.date
    holyday_df.set_index(['holyday_date'], inplace=True)

    days_dict = {0: 'Monday',
                 1: 'Tuesday',
                 2: 'Wednesday',
                 3: 'Thurday',
                 4: 'Friday',
                 5: 'Saturday',
                 6: 'Sunday'}


    def config_plotly():
        import plotly.io as pio
        pio.renderers.default = "browser"


    def config_pandas():
        """
        Allows to show all the columns of a dataframe in the console
        Limit pandas warnings
        """
        import pandas as pd
        import numpy as np
        pd.options.mode.chained_assignment = None  # default='warn'
        desired_width = 350
        np.set_printoptions(linewidth=desired_width)  # show dataframes in console
        pd.set_option('display.max_columns', 10)

if 1 == 0:
    """ ============= DESCRIPTION ===============
        This script defines global variables, paths and settings
    """
    import pandas as pd

    read_path = 'D:/CLARENCE/C7_AVTZoneDef/AVTZoneDef'
    # read_path = 'D:/CLARENCE/prueba'

    write_path = 'D:/CLARENCE/C7_AVTZoneDef_agrupados_ponderados/'

    # Path with grouped data
    data_path = 'D:/OneDrive/Indra/DIGITAL LABS DIGINET - 15_Proyecto_TTE_Clarence/03_Documentacion_Tecnica/08_Datos/02_Grouped_Data/C7_AVTZoneDef/'
    # data_path = 'D:/CLARENCE/prueba/'

    images_path = 'D:/CLARENCE/figures'

    segments_path = 'D:/CLARENCE/SegmentsLocations_C7.csv'

    models_path = 'D:/CLARENCE/models/C7'

    data_path_analytics = 'D:\\dataanalytics\\dataanalytics.predictive\\data\\'

    extension = 'csv'

    road = 'Clem 7'

    columns_ = ['Volume', 'Occupancy',
                'Headway', 'Meanlength',
                'Speed']

    all_columns_ = ['time', 'Volume', 'Occupancy',
                    'Headway', 'Meanlength',
                    'Speed',
                    'Name']

    original_columns = ['Name', 'Last_Update', 'Health_State', 'Availability_',
                        'Count_', 'Average_Occupancy', 'Average_Headway',
                        'Average_Length', 'Average_Speed', 'Count_Short',
                        'Count_Medium', 'Count_Long', 'Average_Speed_Short',
                        'Average_Speed_Medium', 'Average_Speed_Long']

    lane_dict = {'_L': 1, 'FL': 2, 'SL': 3}  # FL = Fast Lane, SL = Slow Lane, _L = Only Lane
    direction_dict = {'Eastbound': 1, 'Westbound': 2, 'Northbound': 3, 'Southbound': 4}

    # Encode sections
    road_topology_df = pd.DataFrame([['AVI101_FL', 1, 3, 2, 1, 0, 2],
                                     ['AVI101_SL', 1, 3, 3, 2, 0, 2],
                                     ['AVI102_FL', 2, 3, 2, 3, 1, 3],
                                     ['AVI102_SL', 2, 3, 3, 4, 1, 3],
                                     ['AVI110_FL', 3, 3, 2, 5, 2, 4],
                                     ['AVI110_SL', 3, 3, 3, 6, 2, 4],
                                     ['AVI111_FL', 4, 3, 2, 7, 3, 5],
                                     ['AVI111_SL', 4, 3, 3, 8, 3, 5],
                                     ['AVI117_FL', 5, 3, 2, 9, 4, 6],
                                     ['AVI117_SL', 5, 3, 3, 10, 4, 6],
                                     ['AVI118_FL', 6, 3, 2, 11, 5, 7],
                                     ['AVI118_SL', 6, 3, 3, 12, 5, 7],
                                     ['AVI125_FL', 7, 3, 2, 13, 6, 8],
                                     ['AVI125_SL', 7, 3, 3, 14, 6, 8],
                                     ['AVI126_FL', 8, 3, 2, 15, 7, 9],
                                     ['AVI126_SL', 8, 3, 3, 16, 7, 9],
                                     ['AVI133_FL', 9, 3, 2, 17, 8, 10],
                                     ['AVI133_SL', 9, 3, 3, 18, 8, 10],
                                     ['AVI134_FL', 10, 3, 2, 19, 9, 11],
                                     ['AVI134_SL', 10, 3, 3, 20, 9, 11],
                                     ['AVI141_FL', 11, 3, 2, 21, 10, 12],
                                     ['AVI141_SL', 11, 3, 3, 22, 10, 12],
                                     ['AVI142_FL', 12, 3, 2, 23, 11, 13],
                                     ['AVI142_SL', 12, 3, 3, 24, 11, 13],
                                     ['AVI148_FL', 13, 3, 2, 25, 12, 14],
                                     ['AVI148_SL', 13, 3, 3, 26, 12, 14],
                                     ['AVI149_FL', 14, 3, 2, 27, 13, 15],
                                     ['AVI149_SL', 14, 3, 3, 28, 13, 15],
                                     ['AVI156_FL', 15, 3, 2, 29, 14, 16],
                                     ['AVI156_SL', 15, 3, 3, 30, 14, 16],
                                     ['AVI157_FL', 16, 3, 2, 31, 15, 17],
                                     ['AVI157_SL', 16, 3, 3, 32, 15, 17],
                                     ['AVI162_FL', 17, 3, 2, 33, 16, 18],
                                     ['AVI162_SL', 17, 3, 3, 34, 16, 18],
                                     ['AVI163_FL', 18, 3, 2, 35, 17, 19],
                                     ['AVI163_SL', 18, 3, 3, 36, 17, 19],
                                     ['AVI170_FL', 19, 3, 2, 37, 18, 20],
                                     ['AVI170_SL', 19, 3, 3, 38, 18, 20],
                                     ['AVI171_FL', 20, 3, 2, 39, 19, 21],
                                     ['AVI171_SL', 20, 3, 3, 40, 19, 21],
                                     ['AVI177_FL', 21, 3, 2, 41, 20, 22],
                                     ['AVI177_SL', 21, 3, 3, 42, 20, 22],
                                     ['AVI178_FL', 22, 3, 2, 43, 21, 0],
                                     ['AVI178_SL', 22, 3, 3, 44, 21, 0],
                                     ['AVI585_FL', 23, 4, 2, 45, 0, 24],
                                     ['AVI585_SL', 23, 4, 3, 46, 0, 24],
                                     ['AVI584_FL', 24, 4, 2, 47, 23, 25],
                                     ['AVI584_SL', 24, 4, 3, 48, 23, 25],
                                     ['AVI576_FL', 25, 4, 2, 49, 24, 26],
                                     ['AVI576_SL', 25, 4, 3, 50, 24, 26],
                                     ['AVI575_FL', 26, 4, 2, 51, 25, 27],
                                     ['AVI575_SL', 26, 4, 3, 52, 25, 27],
                                     ['AVI569_FL', 27, 4, 2, 53, 26, 28],
                                     ['AVI569_SL', 27, 4, 3, 54, 26, 28],
                                     ['AVI568_FL', 28, 4, 2, 55, 27, 29],
                                     ['AVI568_SL', 28, 4, 3, 56, 27, 29],
                                     ['AVI562_FL', 29, 4, 2, 57, 28, 30],
                                     ['AVI562_SL', 29, 4, 3, 58, 28, 30],
                                     ['AVI561_FL', 30, 4, 2, 59, 29, 31],
                                     ['AVI561_SL', 30, 4, 3, 60, 29, 31],
                                     ['AVI554_FL', 31, 4, 2, 61, 30, 32],
                                     ['AVI554_SL', 31, 4, 3, 62, 30, 32],
                                     ['AVI553_FL', 32, 4, 2, 63, 31, 33],
                                     ['AVI553_SL', 32, 4, 3, 64, 31, 33],
                                     ['AVI546_FL', 33, 4, 2, 65, 32, 34],
                                     ['AVI546_SL', 33, 4, 3, 66, 32, 34],
                                     ['AVI545_FL', 34, 4, 2, 67, 33, 35],
                                     ['AVI545_SL', 34, 4, 3, 68, 33, 35],
                                     ['AVI539_FL', 35, 4, 2, 69, 34, 36],
                                     ['AVI539_SL', 35, 4, 3, 70, 34, 36],
                                     ['AVI538_FL', 36, 4, 2, 71, 35, 37],
                                     ['AVI538_SL', 36, 4, 3, 72, 35, 37],
                                     ['AVI531_FL', 37, 4, 2, 73, 36, 38],
                                     ['AVI531_SL', 37, 4, 3, 74, 36, 38],
                                     ['AVI530_FL', 38, 4, 2, 75, 37, 39],
                                     ['AVI530_SL', 38, 4, 3, 76, 37, 39],
                                     ['AVI522_FL', 39, 4, 2, 77, 38, 40],
                                     ['AVI522_SL', 39, 4, 3, 78, 38, 40],
                                     ['AVI521_FL', 40, 4, 2, 79, 39, 41],
                                     ['AVI521_SL', 40, 4, 3, 80, 39, 41],
                                     ['AVI516_FL', 41, 4, 2, 81, 40, 42],
                                     ['AVI516_SL', 41, 4, 3, 82, 40, 42],
                                     ['AVI515_FL', 42, 4, 2, 83, 41, 43],
                                     ['AVI515_SL', 42, 4, 3, 84, 41, 43],
                                     ['AVI509_FL', 43, 4, 2, 85, 42, 44],
                                     ['AVI509_SL', 43, 4, 3, 86, 42, 44],
                                     ['AVI508_FL', 44, 4, 2, 87, 43, 45],
                                     ['AVI508_SL', 44, 4, 3, 88, 43, 45],
                                     ['AVI504_FL', 45, 4, 2, 89, 44, 46],
                                     ['AVI504_SL', 45, 4, 3, 90, 44, 46],
                                     ['AVI503_FL', 46, 4, 2, 91, 45, 47],
                                     ['AVI503_SL', 46, 4, 3, 92, 45, 47],
                                     ['AVT002_FL', 47, 3, 2, 93, 0, 1],
                                     ['AVT002_SL', 47, 3, 2, 94, 0, 1],
                                     ['AVI007_L', 48, 3, 1, 95, 0, 3],
                                     ['AVT020_L', 49, 3, 1, 96, 0, 50],
                                     ['AVI401_L', 50, 3, 1, 97, 49, 51],
                                     ['AVI402_L', 51, 3, 1, 98, 50, 52],
                                     ['AVI412_L', 52, 3, 1, 99, 51, 53],
                                     ['AVI414_L', 53, 3, 1, 100, 52, 11],
                                     ['AVT030_L', 54, 3, 1, 101, 22, 55],
                                     ['AVT034_FL', 55, 3, 2, 102, 54, 0],
                                     ['AVT034_SL', 55, 3, 3, 103, 54, 0],
                                     ['AVT031_FL', 56, 3, 2, 104, 22, 0],
                                     ['AVT031_SL', 56, 3, 3, 105, 22, 0],
                                     ['AVT037_L', 57, 4, 1, 106, 0, 58],
                                     ['AVI039_FL', 58, 4, 2, 107, 57, 23],
                                     ['AVT036_FL', 59, 4, 2, 108, 0, 60],
                                     ['AVT036_SL', 59, 4, 3, 109, 0, 60],
                                     ['AVI039_FL', 60, 4, 2, 110, 59, 23],
                                     ['AVT033_SL', 60, 4, 3, 111, 59, 23],
                                     ['AVT035_L', 61, 4, 1, 112, 0, 62],
                                     ['AVT032_FL', 62, 4, 2, 113, 61, 23],
                                     ['AVT032_SL', 62, 4, 3, 114, 61, 23],
                                     ['AVI805_L', 63, 4, 1, 115, 36, 64],
                                     ['AVI806_L', 64, 4, 1, 116, 63, 65],
                                     ['AVI802_FL', 65, 4, 2, 117, 64, 66],
                                     ['AVI802_SL', 65, 4, 3, 118, 64, 66],
                                     ['AVI801_FL', 66, 4, 2, 119, 65, 67],
                                     ['AVI801_SL', 66, 4, 3, 120, 65, 67],
                                     ['AVT021_FL', 67, 4, 2, 121, 66, 0],
                                     ['AVT021_SL', 67, 4, 3, 122, 66, 0],
                                     ['AVT004_L', 68, 4, 1, 123, 46, 0],
                                     ['AVT003_FL', 69, 4, 2, 124, 46, 0],
                                     ['AVT003_SL', 69, 4, 3, 125, 46, 0]
                                     ]
                                    , columns=['sub_section_name', 'section', 'direction', 'lanes', 'sub_section',
                                               'prev_section', 'next_section'])
    road_topology_df.set_index(['sub_section_name'], inplace=True)

    holyday_list = [['New Years Day', '2017-01-01', 1],
                    ['Australia Day', '2017-01-26', 2],
                    ['Good Friday', '2017-04-14', 3],
                    ['The day after Good Friday', '2017-04-15', 4],
                    ['Easter Sunday', '2017-04-16', 5],
                    ['Easter Monday', '2017-04-17', 6],
                    ['Aznac Day', '2017-04-25', 7],
                    ['Labour Day', '2017-05-01', 8],
                    ['Royal Queensland Show', '2017-08-16', 9],
                    ['Queens Birthday', '2017-10-2', 10],
                    ['Christmas Eve', '2017-12-24', 11],
                    ['Christmas Day', '2017-12-25', 12],
                    ['Boxing Day', '2017-12-26', 13],
                    ['New Years Day', '2018-01-01', 1],
                    ['Australia Day', '2018-01-26', 2],
                    ['Good Friday', '2018-03-30', 3],
                    ['The day after Good Friday', '2018-03-31', 4],
                    ['Easter Sunday', '2018-04-01', 5],
                    ['Easter Monday', '2018-04-02', 6],
                    ['Aznac Day', '2018-04-25', 7],
                    ['Labour Day', '2018-05-07', 8],
                    ['Royal Queensland Show', '2018-08-15', 9],
                    ['Queens Birthday', '2018-10-1', 10],
                    ['Christmas Eve', '2018-12-24', 11],
                    ['Christmas Day', '2018-12-25', 12],
                    ['Boxing Day', '2018-12-26', 13],
                    ['New Years Day', '2019-01-01', 1],
                    ['Australia Day', '2019-01-28', 2],
                    ['Good Friday', '2019-04-19', 3],
                    ['The day after Good Friday', '2019-04-20', 4],
                    ['Easter Sunday', '2019-04-21', 5],
                    ['Easter Monday', '2019-04-22', 6],
                    ['Aznac Day', '2019-04-25', 7],
                    ['Labour Day', '2019-05-06', 8],
                    ['Royal Queensland Show', '2019-08-14', 9],
                    ['Queens Birthday', '2019-10-7', 10],
                    ['Christmas Eve', '2019-12-24', 11],
                    ['Christmas Day', '2019-12-25', 12],
                    ['Boxing Day', '2019-12-26', 13]]
    holyday_df = pd.DataFrame(holyday_list, columns=['holyday_name', 'holyday_date', 'holyday_id'])
    holyday_df['holyday_date'] = pd.to_datetime(holyday_df['holyday_date']).dt.date
    holyday_df.set_index(['holyday_date'], inplace=True)

    days_dict = {0: 'Monday',
                 1: 'Tuesday',
                 2: 'Wednesday',
                 3: 'Thurday',
                 4: 'Friday',
                 5: 'Saturday',
                 6: 'Sunday'}




""" ============= DESCRIPTION ===============
    This script defines global variables, paths and settings


"""
data_path = 'D:\\INDRA\\Clarence-data\\'

data_path_analytics = 'D:\\dataanalytics\\dataanalytics.predictive\\data\\'
data_path_results = 'D:\\dataanalytics\\dataanalytics.predictive\\output\\'

project_path = 'C:\\Users\\javelascor\\OneDrive - Indra\\00-PROYECTOS Y OFERTAS\\05_Clarence_TTE'
data_path_TDM = project_path + '\\03_Documentacion_Tecnica\\08_Datos\\03_Suplementary_data\original\\'


def config_pandas():
    """
    Allows to show all the columns of a dataframe in the console
    Limit pandas warnings
    """
    import pandas as pd
    import numpy as np
    pd.options.mode.chained_assignment = None  # default='warn'
    desired_width = 350
    np.set_printoptions(linewidth=desired_width)  # show dataframes in console
    pd.set_option('display.max_columns', 10)


def config_plotly():
    import plotly.io as pio
    pio.renderers.default = "browser"


""" ============================
    (!) Change name of the road
"""
road = 'Legacy Way'

columns_ = ['flow', 'volume', 'speed', 'meanlength', 'gap', 'headway', 'occupancy']

all_columns_ = ['length', 'status', 'polltime', 'confidence',
                'lanes', 'periods', 'volume', 'flow', 'speed', 'meanlength', 'gap',
                'headway', 'levelofservice', 'occupancy', 'offset', 'datasetId',
                'ownSysId']

columns_group = ['length', 'status', 'polltime', 'confidence',
                 'lanes', 'periods', 'levelofservice', 'offset', 'datasetId',
                 'ownSysId']

units_ = {"flow": "(veh./h)",
          "volume": "(veh./15min)",
          "speed": '(km/h)',
          "meanlength": "(dm)",
          "gap": "(tenths of sec)",
          "headway": '(tenths of sec)',
          "occupancy": "(%)"}

dict_sub_sections = {1: 'Section 1 - Lane 1 - EB - Entry',
                     2: 'Section 1 - Lane 2 - EB - Entry',
                     3: 'Section 2 - Lane 1 - EB - Exit',
                     4: 'Section 2 - Lane 2 - EB - Exit',
                     5: 'Section 3 - Lane 1 - WB - Entry',
                     6: 'Section 3 - Lane 2 - WB - Entry',
                     7: 'Section 4 - Lane 1 - WB - Exit',
                     8: 'Section 4 - Lane 2 - WB - Exit'}

topology_dictionary = {'LIU1001_01': 'Lane 1 Entry EB Subsection 1',
                       'LIU1001_02': 'Lane 2 Entry EB Subsection 2',
                       'LIU1001_03': 'Lane 1 Exit WB Subsection 7',
                       'LIU1001_04': 'Lane 2 Exit WB Subsection 8',
                       'LIU2001_01': 'Lane 1 Entry WB Subsection 5',
                       'LIU2001_02': 'Lane 2 Entry WB Subsection 6',
                       'LIU2001_03': 'Lane 1 Exit EB Subsection 3',
                       'LIU2001_04': 'Lane 2 Exit EB Subsection 2'}
