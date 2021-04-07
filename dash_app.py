# imports
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd
from pandas.io import gbq
import pyodbc
from sklearn.preprocessing import StandardScaler
import random
from resources.algorithm_choosing import comparer

# alert var
alert = ""

# #############################################################################
# ############## Fetching and prepping data from local database  ##############
# #############################################################################

# connection string
con_str = 'Driver={ODBC Driver 17 for SQL Server};Server=localhost;DATABASE=NORTHWND;Trusted_Connection=yes;UID=FICKDICH\Hardy'

# connect to db
conxn = pyodbc.connect(con_str)
cursor = conxn.cursor()

query_str = """
SELECT[Order Details].UnitPrice, UnitsInStock, Quantity, Freight FROM [NORTHWND].[dbo].[Order Details]
JOIN Orders ON [Order Details].OrderID = Orders.OrderID
JOIN Products ON [Order Details].ProductID = Products.ProductID
"""
cursor.execute(query_str)
results = cursor.fetchall()

# make a data frame out of the test data
cols = ['UnitPrice', 'UnitsInStock', 'Quantity', 'Freight']
df = pd.DataFrame.from_records(results, columns=cols)

# clean data: remove NaN's
df = df.dropna()

# downcast decimals to float
df = df.apply(pd.to_numeric, downcast='float')

# #############################################################################
# ############################ Data preprocessing #############################
# #############################################################################

ss = StandardScaler()

# fit + transform data starting from row 2
scaled_data = ss.fit_transform(df.iloc[:, :])

# create data frame with scaled data, leave header untouched
df_scaled = df.copy()
df_scaled.iloc[:, :] = scaled_data

# Parameters that are read from the source
available_params = cols
available_params_prettified = ['Stückpreis', 'lagernd', 'Bestellmenge', 'Frachtkosten']

# Parameters to choose from (data base names & display names)
select_params = [x for x in available_params if x not in ['hat_kinder']]
select_params_prettified = [x for x in available_params_prettified if x not in ['Elternanteil']]

# color list to use for clusters
cluster_colors = ["brown", "darkorchid", "cornflowerblue", "forestgreen", "gold", "olive"]


# #############################################################################
# ###################### apply clustering algorithm ###########################
# #############################################################################

# function to apply algorithm on the data
def apply_cluster_alg(x, y, n, algorithm='kmeans', epsilon=0, min_pts=0):
    global alert
    dff = {}
    centers = None
    if algorithm == 'kmeans':
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=n)
        # apply algorithm on scaled data
        km.fit(df_scaled[[x, y]])
        # add cluster information to rows
        df_scaled["cluster"] = km.labels_
        daf = df.copy()
        # add cluster labels to copy of original dataframe
        daf["cluster"] = df_scaled.loc[:, "cluster"]
        centers = km.cluster_centers_
        alert = "Clustering done with kmeans"
    elif algorithm == 'dbscan':
        from sklearn.cluster import DBSCAN
        dbscan = DBSCAN(eps=epsilon, min_samples=min_pts)
        # apply clustering to parameters x and y
        dbscan.fit(df_scaled[[x, y]])
        # add cluster info to rows
        df_scaled["cluster"] = dbscan.labels_
        daf = df.copy()
        # add labels to non-scaled data
        daf["cluster"] = df_scaled.loc[:, "cluster"]
        # there is no center information for dbscan, leave empty
        centers = []
        alert = "Clustering done with dbscan"
    elif algorithm == 'aggl':
        from sklearn.cluster import AgglomerativeClustering
        aggl = AgglomerativeClustering(n_clusters=n)
        # apply clustering to the data
        aggl.fit(df_scaled[[x, y]])
        # add cluster information to rows
        df_scaled["cluster"] = aggl.labels_
        # copy original and add cluster labels
        daf = df.copy()
        daf["cluster"] = df_scaled.loc[:, "cluster"]
        # again, there is no center information
        centers = []
        alert = "Agglomerative Clustering applied"
    elif algorithm == 'gmm':
        from sklearn.mixture import GaussianMixture
        gmm = GaussianMixture(n_components=n)
        # apply mixture model to data
        cl_labels = gmm.fit_predict(df_scaled[[x, y]])
        # add cluster information to rows, we can directly add it to a copy of the original
        daf = df.copy()
        daf["cluster"] = cl_labels
        # we don't have center information from the mixture model
        centers = []
        alert = "Clustering done with Gaussian mixture model"
    elif algorithm == 'fuzz':
        import fcmeans as FCM
        fcm = FCM.FCM(n_clusters=n)
        # apply clustering to data
        fcm.fit(df_scaled[[x, y]])
        # get centers
        centers = fcm.centers
        # add labels to data
        fcm_labels = fcm.predict(df_scaled[[x, y]])
        daf = df.copy()
        daf["cluster"] = fcm_labels
        alert = "Clustering done with fuzzy c-means"
    return daf, centers


# function for turning a dataframe into a dbc table visual component
def df_to_table(table_data, cols):
    table_components = []  # list of tables, in bootstrap component form for adding them to the view
    for element, i in zip(table_data, range(len(table_data))):
        t = pd.DataFrame.from_dict(element, orient='index', columns=["Median", "Durchschnitt"])
        # set names for rows to prettified versions of selected attributes
        t = t.rename(index={j: available_params_prettified[available_params.index(available_params[j])]
                            for j in range(len(available_params))}).reset_index()
        # rename index header to parameter (selected attributes)
        t = t.rename(columns={'index': 'Parameter'})
        # transform into bootstrap component + add style
        t_done = dbc.Table.from_dataframe(t, striped=True, bordered=True, dark=True,
                                          style={"color": cluster_colors[i]})
        table_components.append(t_done)
    return table_components


# function for calculating averages (Mean and Median)
# noinspection SpellCheckingInspection
def perform_calculations(dframe, n):
    cframes = [dframe.loc[dframe["cluster"] == i] for i in range(n)]
    # bring data in order: sorted by cluster they belong to, ascending (that way the colors can be matched easily)
    cframes = sorted(cframes, key=lambda k: k['cluster'].min(axis=0))
    table_list = []  # list of tables
    # iterate over x and y per cluster to determine medians and averages
    for frame in cframes:
        del frame['cluster']
        table_dict = {}  # dictionary that holds all information per entry for one table
        for col, n in zip(frame, range(0, len(frame.index))):
            md = frame[col].median()
            m = round(frame[col].mean(), 2)
            table_dict[n] = [md, m]

        table_list.append(table_dict)
    return table_list


'''
# takes table data and creates info texts
def perform_nlp(table_data):
    info_texts = []
    r_match_spend = random.randint(0, len(phrases.spendings_match_start) - 1)
    for table, i in zip(table_data, range(len(table_data))):
        info_texts.append(
            html.P(
                phrases.age_start[random.randint(0, len(phrases.age_start) - 1)]
                + str(table[0][1])
                + phrases.age_end[random.randint(0, len(phrases.age_end) - 1)]
                + phrases.space
                + phrases.children[random.randint(0, len(phrases.children) - 1)]
                + str(round(table[1][1] * 100, 2))
                + phrases.dot
                + phrases.login_start[random.randint(0, len(phrases.login_start) - 1)]
                + str(table[2][1])
                + phrases.login_end[random.randint(0, len(phrases.login_end) - 1)]
                + phrases.login_perc[random.randint(0, len(phrases.login_perc) - 1)]
                + str(table[3][1])
                + phrases.dot
                # + phrases.spendings_match_start[r_match_spend]
                + str(table[4][1] * -1)
                # + phrases.spendings_match_end[r_match_spend]
            )
        )
    return info_texts
'''

# #############################################################################
# ############################## generate UI ###################################
# #############################################################################

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

# axis controls consisting of labels and dropdowns
controls = dbc.Card(
    [
        # axes dropdown w/ defaults
        dbc.FormGroup(
            [
                dbc.Label("X Achse"),
                dcc.Dropdown(
                    id="x-axis",
                    # internal values use the column names from the database, labels use 'prettified' names
                    options=[
                        {"label": col_p, "value": col} for col, col_p in zip(select_params, select_params_prettified)
                    ],
                    value=select_params[0],
                ),
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label("Y Achse"),
                dcc.Dropdown(
                    id="y-axis",
                    # same as x-axis
                    options=[
                        {"label": col_p, "value": col} for col, col_p in zip(select_params, select_params_prettified)
                    ],
                    value=select_params[1],
                ),
            ]
        ),
        # number of clusters as input field
        dbc.FormGroup(
            [
                dbc.Label("Cluster Anzahl"),
                dbc.Input(id="cluster-count", type="number", value=5, min=2, max=6),
            ]
        ),
    ],
    body=True,
)

# html
app.layout = dbc.Container(
    [
        html.H1("Data Sample Clustering - NORTHWIND"),
        html.Hr(),
        html.P(id="alg_p", children=["clarification_str"]),
        dbc.Row(
            [
                dbc.Col(controls, md=4),
                dbc.Col(dcc.Graph(id="cluster-graph"), md=8)
            ],
            align="center"
        ),
        dbc.Col(id='info-area')
    ],
    fluid=True
)


# when figure is updated, print alert showing which algorithm was used
@app.callback(
    Output("alg_p", "children"),
    Input("cluster-graph", "figure")
)
def send_alarm(graph_figure):
    return alert


# re-calc function for axes selection
@app.callback(
    [
        Output("cluster-graph", "figure"),
        # Output("table-container", "children")
        Output("info-area", "children")
    ],
    [
        Input("x-axis", "value"),
        Input("y-axis", "value"),
        Input("cluster-count", "value")
    ],
)
def apply_alg_to_view(x_value, y_value, n_clusters):
    cmp_result = comparer.chooseAlgorithm(df, x_value, y_value, n_clusters)
    cl_alg = cmp_result[0]
    eps = cmp_result[1]
    min_pts = cmp_result[2]
    dff, centers = apply_cluster_alg(x_value, y_value, n_clusters, algorithm=cl_alg, epsilon=eps, min_pts=min_pts)
    # create graph
    data = [
        go.Scatter(
            x=df_scaled.loc[dff.cluster == c, x_value],
            y=df_scaled.loc[dff.cluster == c, y_value],
            mode="markers",
            marker={"color": cluster_colors[c], "size": 8},
            name="Cluster {}".format(c),
        )
        for c in range(n_clusters)
    ]
    if len(centers) > 0:
        data.append(
            go.Scatter(
                x=centers[:, 0],
                y=centers[:, 1],
                mode="markers",
                marker={"color": "#000", "size": 12, "symbol": "diamond"},
                name="Cluster Zentren",
            ))
    fig = go.Figure(data=data)
    fig.update_xaxes(title_text=select_params_prettified[select_params.index(x_value)])
    fig.update_yaxes(title_text=select_params_prettified[select_params.index(y_value)])

    # create tables + text data
    table_data = perform_calculations(dff, n_clusters)

    table_components = df_to_table(table_data,
                                   [x_value, y_value])  # list of dbc table components for adding to the output Div
    # Text component erstellen und den Tabellen zuordnen
    info_area = []  # list of 'dbc.Row's + corresponding texts
    # text_components = perform_nlp(table_data)
    # combine dbc tables and texts
    # for table, info in zip(table_components, text_components):
    for table in table_components:
        info_area.append(
            dbc.Row(
                [
                    dbc.Col(table, md=6)
                ]
            )
        )
    return fig, info_area


# make sure that x and y values can't be the same variable
def filter_options(v):
    """Disable option v"""
    return [
        {"label": col_p, "value": col, "disabled": col == v}
        for col, col_p in zip(select_params, select_params_prettified)
    ]


# functionality is the same for both dropdowns, so we reuse filter_options
app.callback(Output("x-axis", "options"), [Input("y-axis", "value")])(
    filter_options
)
app.callback(Output("y-axis", "options"), [Input("x-axis", "value")])(
    filter_options
)

if __name__ == "__main__":
    app.run_server(debug=True, port=8888)
