
#exec(open("src/visualization/visualization_reporting_functions.py").read())
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots

def flatten(list_of_lists):
    if len(list_of_lists) == 0:
        return list_of_lists
    if isinstance(list_of_lists[0], list):
        return flatten(list_of_lists[0]) + flatten(list_of_lists[1:])
    return list_of_lists[:1] + flatten(list_of_lists[1:])

def st_plots_sp(dic, sp, obj):

    minmaxnum = 'max'
    data = dic[sp]
    number_seeds = data['number_seeds']
    n_initial = data['n_initial']
    n_queries = data['n_queries']

     #hist plot
    fig_his, ax = plt.subplots()
    his = data['y_dist']
    his_obj = his[obj]
    fig_his, ax = plt.subplots()
    plt.gca().title.set_text("Y distribution")
    ax.hist(his_obj, bins=20)


    #######################################################ANN
    ann = data['ann']
    focus = ann['max']
    # objective keys contains optimal points and all the values from random seeds.
    focus1 = focus[obj]
    time = focus1['Average Time']
    time = format(time, '.1f')
    optimal_point = focus1['optimal_point']
    opt = format(optimal_point, '.1f')
    opt = "Optimal Point: " + opt
    # Taking out the optimal point from the dictionary
    del focus1['optimal_point']
    del focus1['Average Time']
    df = pd.DataFrame.from_dict(focus1)
    df = df.drop(labels='plot_data', axis=0)
    df = df.drop(labels='initial_y', axis=0)

    # first compute the cumulative yields for each random seed
    cu_yields = []
    for seed in range(len(df.columns)):
        rseed = df.iloc[:, seed]
        yields = []
        if minmaxnum == 'max':
            current_opt = np.max(np.array(rseed[0]))
        elif minmaxnum == 'min':
            current_opt = np.min(np.array(rseed[0]))
        for qr in range(len(rseed)):
            query = np.array(rseed[qr])
            top_suggestion = np.max(query)
            if minmaxnum == "min":
                top_suggestion = np.min(query)
                if top_suggestion <= current_opt:
                    current_opt = top_suggestion
                yields.append(current_opt)
            if minmaxnum == "max":
                top_suggestion = np.max(query)
                if top_suggestion >= current_opt:
                    current_opt = top_suggestion
                yields.append(current_opt)
        cu_yields.append(yields)
    plot = np.array(cu_yields).T

    xvalues = list(range(n_initial + 1, (n_queries + n_initial + 1)))

    # averages by rows
    averages = np.mean(plot, axis=1)
    final_avg = averages[-1]
    final_avg = format(final_avg, '.1f')
    title = " ANN: " + \
            " Final Average: " + final_avg + "<br>" + time + " sec"

    fig = go.Figure()
    custom_template = {
        "layout": go.Layout(
            font={
                "family": "Nunito",
                "size": 12,
                "color": "#1f1f1f",
            },
            title={
                "font": {
                    "family": "Lato",
                    "size": 18,
                    "color": "#1f1f1f",
                },
            },
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            colorway=px.colors.qualitative.G10,
        )
    }

    for i in range(0, number_seeds):
        fig.add_trace(go.Scatter(x=xvalues, y=plot[:, i], mode="lines", line=dict(color='black', dash='dot',
                                                                                  width=4),
                                 name='Random Seed ' + str(i + 1),
                                 customdata=plot[:, i],
                                 hovertemplate='x:%{x}<br>y:%{y}<br>z:%{z}<br>target: %{customdata} <extra></extra> '))

    fig.add_trace(go.Scatter(x=xvalues, y=averages, mode="lines+markers", line=dict(color='navy', width=4),
                             marker=dict(size=2, color='navy'), name='Average'))

    fig.add_shape(type='line',
                  x0=xvalues[0],
                  y0=optimal_point,
                  x1=xvalues[-1],
                  y1=optimal_point,
                  line=dict(color='Red', width=4),
                  )
    fig.update_layout(xaxis={'title': 'Experiment', 'title_font_size': 15, 'tickfont_size': 15},
                      yaxis={'title': 'Cumulative Yield', 'title_font_size': 15, 'tickfont_size': 15},
                      title=title,
                      showlegend=False,
                      template=custom_template)
    fig.update_layout(
        xaxis=dict(mirror=True, ticks='inside', tickwidth=1.5, showline=True, linewidth=1, linecolor='black'),
        yaxis=dict(mirror=True, ticks='inside', tickwidth=1.5, showline=True, linewidth=1, linecolor='black'))
    fig.update_layout(width=800, height=600)

    return fig, opt, fig_his

def algo_plots(dic, sp, obj, seed):
    sizem = 7  # 2d use 12    #3d use 7
    sizef = 9  # 2d use 12     #3d use 9
    sizet = 11  # 3d use 11

    minmaxnum = 'max'
    data = dic[sp]
    ann = data['ann']
    focus = ann['max']
    # objective keys contains optimal points and all the values from random seeds.
    focus1 = focus[obj]
    rseed = focus1[seed]
    plotd = rseed['plot_data']

    train = plotd['training_plot']
    inputs = train['inp_names']
    df_training = train['inputs']
    df_training = pd.DataFrame(df_training, columns=inputs)
    df_training = df_training.sort_values(by=[inputs[3]])

    y_training = train['output']
    out_names = train['out_names']
    y_training = pd.DataFrame(y_training, columns=out_names)

    fig_train = px.scatter_3d(df_training, x=df_training[inputs[0]], y=df_training[inputs[1]],
                        z=df_training[inputs[2]], animation_frame=df_training[inputs[3]],
                        color_continuous_scale=px.colors.sequential.Viridis,
                              title = 'Training Data',
                        color=y_training.iloc[:, 0], opacity=0.8,
                        range_color=[min(y_training.iloc[:, 0]), max(y_training.iloc[:, 0])])

    fig_train.update_layout(scene=dict(
        xaxis_title=inputs[0],
        yaxis_title=inputs[1],
        zaxis_title=inputs[2],
        xaxis_tickfont_size=sizet, yaxis_tickfont_size=sizet, zaxis_tickfont_size=sizet),
        font=dict(size=sizef),
        width=700,
        height=500
    )

    pred_p = plotd['prediction_plot']

    df_training = pred_p['inputs']
    inp_names = pred_p['inp_names']
    x_pool = pred_p['x_pool']
    y_training = pred_p['output']
    y_pred = pred_p['pred']
    out_names = pred_p['out_names']

    y_pred_df = pd.DataFrame(y_pred, columns=out_names)
    x_pool_plot = x_pool
    x_pool_plot = pd.DataFrame(x_pool_plot, columns=inp_names)
    x_pool_plot = x_pool_plot.apply(pd.to_numeric)

    fig_pred = px.scatter_3d(x_pool_plot, x=x_pool_plot[inputs[0]], y=x_pool_plot[inputs[1]], z=x_pool_plot[inputs[2]],
                        animation_frame=x_pool_plot[inputs[3]], color=y_pred_df[out_names[0]], opacity=0.8,
                        color_continuous_scale=px.colors.sequential.Viridis, range_color=[min(y_pred_df.iloc[:, 0]),
                                                                                          max(y_pred_df.iloc[:, 0])],
                        title="Model PREDICTION")

    fig_pred.layout.coloraxis.colorbar.title = out_names[0]
    fig_pred.update_layout(scene=dict(
        xaxis_title=inputs[0],
        yaxis_title=inputs[1],
        zaxis_title=inputs[2],
        xaxis_tickfont_size=sizet, yaxis_tickfont_size=sizet, zaxis_tickfont_size=sizet,
    ))

    fig_pred.update_layout(font=dict(size=sizef))
    fig_pred.update_layout(width=700, height=500)

    return fig_train, fig_pred