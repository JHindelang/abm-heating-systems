import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_daq as daq
import os
import plotly.express as px
import pandas as pd
from pandas import to_datetime as dt
from math import sqrt
from model_geo import DiffusionModel
import webbrowser
import multiprocessing
from joblib import Parallel, delayed
from flask import request

num_cores = multiprocessing.cpu_count()

# initialize auxilary variables
num = 0
n_mods = 1
steps = 1
rmsd = 0

running = False

# style of app
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


##############################################################################
# model setup
##############################################################################

# change working directory
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

# input and output folder names
folder_input = os.path.join(dir_path, 'input')
folder_output = os.path.join(dir_path, 'output')
path_results = os.path.join(folder_output, 'simulation_results.xlsx')
path_params_save = os.path.join(folder_output, 'params_calib.xlsx')

# file names
shape_file = os.path.join(
    folder_input, 'geo_data', 'shape_file_households', 'households.shp')
tech_file = os.path.join(folder_input, 'heating_systems.xlsx')
build_stock_file = os.path.join(folder_input, 'building_stock.xlsx')
household_file = os.path.join(folder_input, 'household_characterization.xlsx')
timeseries_file = os.path.join(folder_input, 'heat_demand_timeseries.xlsx')
color_file = os.path.join(folder_input, 'colors_TUM_corporate.csv')
input_fact_file = os.path.join(folder_input, 'ind_fact.dat')
ca_file = os.path.join(folder_input, 'central_agents.xlsx')

# dictionary of file names
directories = {
    'shape': shape_file,
    'systems': tech_file,
    'build_stock': build_stock_file,
    'households': household_file,
    'timeseries': timeseries_file,
    'colors': color_file,
    'ca': ca_file,
    'input_f': input_fact_file
}

# read model parameters
parameter_file = os.path.join(dir_path, 'input', 'model_settings.xlsx')
params_df = pd.read_excel(
    io=parameter_file, sheet_name='parameter', header=0, index_col=0
)
params = params_df.to_dict('dict')['value']

for param in params:
    if params[param] == 'True':
        params[param] = True
    if params[param] == 'False':
        params[param] = False

# read simulation settings
settings_df = pd.read_excel(
    io=parameter_file, sheet_name='settings', header=0, index_col=0
)
settings = settings_df.to_dict('dict')['value']

# initialize model
diff_ms = [
    DiffusionModel(
        direct=directories,
        params=params,
        start_date=settings['start_date'],
        t_step=settings['time_step'])
    for i in range(n_mods)]

weights = diff_ms[0].hh_char['decision_weights']['C1']

# read color map
color_list = list(
    pd.read_csv('input\colors_TUM_corporate.csv', header=None, squeeze=True))

# read geo file and convert coordinates
df = diff_ms[0].df_geo
df = df.to_crs(epsg=4326)


def comp_steps(model, n_steps):
    # performs n_steps steps for model
    print('model steps')
    for i in range(n_steps):
        model.step()
    return model

# initialize dash app
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = html.Div(children=[
    html.Div([
        html.H1(
            children='Heat Pump Diffusion Model',
            className="nine columns"),
        html.Img(
            src="C:/Users/jakob/Desktop/Masterthesis/PycharmProjects/heat_pump_adoption/input/tum_logo.png",
            className="three columns",
            style={
                'heigt': '10%',
                'width': '10%',
                'float': 'realtive',
                'position': 'relative',
            },
        ),
        html.Div(
            children='Web Application',
            className="nine columns"
        )
    ], className="row"),
    html.Div([
        html.Button('Step', id='button_step', n_clicks=0),
        html.Button('Load', id='button_load', n_clicks=0),
        html.Button('Start', id='button_start', n_clicks=0),
        html.Button('Stop', id='button_stop', n_clicks=0),
        html.Button('Reset', id='button_reset', n_clicks=0),
        html.Button('save parameter', id='button_save', n_clicks=0),
        html.Button('exit', id='exit', n_clicks=0)
    ], className="row"),
    html.Div([
        html.Div(
            children='Model parameters',
            className='nine columns'
        )
    ], className="row"),
    html.Div([
            html.Div(id='w_cost_disp', style={'margin-top': 20},
                     className='three columns'),
            html.Div(id='w_gen_att_disp', style={'margin-top': 20},
                             className='three columns'),
            html.Div(id='w_ext_thread_disp', style={'margin-top': 20},
                             className='three columns'),
            html.Div(id='w_comf_disp', style={'margin-top': 20},
                             className='three columns'),
    ], className='row'),
    html.Div([
        dcc.Slider(
                id='w_cost',
                min=0,
                max=1,
                step=0.01,
                value=weights['cost_aspect'], className='three columns',
            ),
        dcc.Slider(
                id='w_gen_att',
                min=0,
                max=1,
                step=0.01,
                value=weights['general_attitude'], className='three columns',
            ),
        dcc.Slider(
                id='w_ext_thread',
                min=0,
                max=1,
                step=0.01,
                value=weights['external_threads'], className='three columns',
            ),
        dcc.Slider(
                id='w_comf',
                min=0,
                max=1,
                step=0.01,
                value=weights['comfort'], className='three columns',
            ),
    ], className='row'),
    html.Div([
            html.Div(id='w_peer_disp', style={'margin-top': 20},
                                     className='three columns'),
            html.Div(id='adoption_prob_disp', style={'margin-top': 20},
                                     className='three columns'),
            html.Div(id='age_dist_scale_disp', style={'margin-top': 20},
                                     className='three columns'),
            html.Div(id='age_dist_shape_disp', style={'margin-top': 20},
                                     className='three columns'),
    ], className='row'),
    html.Div([
        dcc.Slider(
                id='w_peer',
                min=0,
                max=1,
                step=0.01,
                value=weights['peers'], className='three columns',
            ),
        dcc.Slider(
                id='adoption_prob',
                min=0,
                max=1,
                step=0.01,
                value=params['adoption_prob'], className='three columns',
            ),
        dcc.Slider(
                id='age_dist_scale',
                min=20,
                max=40,
                step=1,
                value=params['age_dist_scale'], className='three columns',
            ),
        dcc.Slider(
                id='age_dist_shape',
                min=0,
                max=20,
                step=1,
                value=params['age_dist_shape'], className='three columns',
            ),
    ], className='row'),
    html.Div([
            html.Div(id='avg_inter_disp', style={'margin-top': 20},
                     className='three columns'),
            html.Div(id='info_lvl_fact_disp', style={'margin-top': 20},
                             className='three columns'),
            html.Div(id='info_th_disp', style={'margin-top': 20},
                             className='three columns'),
            html.Div(id='peer_th_disp', style={'margin-top': 20},
                             className='three columns'),
    ], className='row'),
    html.Div([
        dcc.Slider(
                id='avg_inter',
                min=0,
                max=1,
                step=0.01,
                value=params['avg_inter'], className='three columns',
            ),
        dcc.Slider(
                id='info_lvl_fact',
                min=0,
                max=1,
                step=0.01,
                value=params['info_lvl_fact'], className='three columns',
            ),
        dcc.Slider(
                id='info_th',
                min=0,
                max=2,
                step=0.01,
                value=params['info_th'], className='three columns',
            ),
        dcc.Slider(
                id='peer_th',
                min=0,
                max=1,
                step=0.01,
                value=params['peer_th'], className='three columns',
            ),
    ], className='row'),
    html.Div([
        daq.BooleanSwitch(
            id='dec_trig_peer',
            label='dec_trig_peer',
            on=params['dec_trig_peer'], className='two columns',
            ),
        daq.BooleanSwitch(
            label='dcm',
            id='dcm',
            on=params['dcm'], className='two columns',
        ),
        daq.BooleanSwitch(
            label='sys_groups',
            id='sys_groups',
            on=params['sys_groups'], className='two columns',
        ),
        html.Div(
            [html.P('Number of iterations'),
            dcc.Input(
                id='iterations',
                type='number',
                min=0, step=1,
                value=n_mods,
            )], className='three columns'
        ),

    ], className='row'),
    html.Div([
        html.Div(
            dcc.Graph(id='rates'),
            id='output-graph',
            className="twelve columns",
        ),
    ], className="row"),
    html.Div([
        html.Div([
            dcc.Graph(id='avg_age')
        ],
            id='graph_avg_age',
            className='eight columns'),
    html.Div([
            dcc.Graph(id='geoGraph')
        ],
            id='geo-graph-cont',
            className="four columns",
        )
    ], className='row'),
    html.Div([
        html.Div([
            dcc.Graph(id='dec_made')
        ],
            id='graph_dec_made',
            className='eight columns'),

    ], className='row'),
    html.Div([
        html.Div([
            dcc.Graph(id='co2')
        ],
            id='graph_co2',
            className='eight columns'),

    ], className='row'),
    dcc.Interval(
            id='interval-component',
            interval=1500,  # in milliseconds
            n_intervals=0),
    html.Div(id='hidden_div', style={'display':'none'}),
    html.Div(id='hidden_div2', style={'display':'none'}),
    html.Div(id='hidden_div3', style={'display':'none'}),
    html.Div(id='hidden_div4', style={'display':'none'}),
    html.Div(id='hidden_div5', style={'display':'none'}),
    html.Div(id='hidden_div6', style={'display':'none'})
])


@app.callback(Output('hidden_div3', 'children'),
              [Input('button_save',  component_property='n_clicks')])
def save_parameter(n):
    if n:
        params_save = {key: params[key] for key in params}
        params_save['w_cost'] = weights['cost_aspect']
        params_save['w_gen_att'] = weights['general_attitude']
        params_save['w_ext_thread'] = weights['external_threads']
        params_save['w_comf'] = weights['comfort']
        params_save['w_peer'] = weights['peers']
        params_df = pd.DataFrame.from_dict(params_save, orient='index')
        params_df.to_csv(path_params_save, sep=';', header=['value'], index=True)
    return 1

@app.callback(
    #Output(component_id='GeoGraph', component_property='mapboxAccessToken')],
    [Output(component_id='rates', component_property='figure'),
     Output(component_id='avg_age', component_property='figure'),
     Output(component_id='dec_made', component_property='figure'),
     Output(component_id='co2', component_property='figure')],
    [Input(component_id='button_step', component_property='n_clicks'),
     Input(component_id='button_start', component_property='n_clicks'),
     Input(component_id='button_reset', component_property='n_clicks'),
     Input(component_id='button_stop', component_property='n_clicks'),
     Input(component_id='interval-component', component_property='n_intervals')]
)
def update_value(input_data1, input_data2, input_data3, input_data4, input_data5):
    ctx = dash.callback_context

    callback_id = ctx.triggered[0]['prop_id'].split('.')[0]

    global diff_ms
    global p
    global running
    global params
    global n_mods

    if callback_id == 'button_step' and not input_data1 == 0:
        print('step start')
        diff_ms = Parallel(n_jobs=num_cores)(
            delayed(comp_steps)(model, steps) for
            model in diff_ms)
        print('step end')
    if callback_id == 'button_start' and not input_data2 == 0:
        running = True
        print(running)
    if callback_id == 'button_stop' and not input_data4 == 0:
        running = False
        print(running)
    if callback_id == 'button_reset' and not input_data3 == 0:
        diff_ms = Parallel(n_jobs=num_cores)(delayed(DiffusionModel)(directories, params, settings['start_date'], settings['time_step']) for i in range(n_mods_1))
        for i in range(n_mods_1):
            diff_ms[i].hh_char['decision_weights']['C1'] = weights
        print(params)
        n_mods = n_mods_1
    if callback_id == 'interval-component' and running:
        for model in diff_ms:
            model.step()

    n = len(diff_ms[0].systems)
    x = diff_ms[0].data_coll.model_vars['time']
    steps_comp = len(x)

    dec_made = [0]*steps_comp
    avg_ex_age = [0]*steps_comp
    avg_age = [0]*steps_comp
    co2_sh = [0]*steps_comp
    co2_dhw = [0]*steps_comp
    systems_ad = {sys.type: [0]*steps_comp for sys in diff_ms[0].systems}

    for model in diff_ms:
        vars = model.data_coll.model_vars
        dec_made = [dec_made[i] + vars['dec_made'][i]/n_mods for i in
                    range(steps_comp)]
        avg_ex_age = [avg_ex_age[i] + vars['avg_ex_age'][i]/n_mods for i in
                    range(steps_comp)]
        avg_age = [avg_age[i] + vars['avg_age'][i]/n_mods for i in
                    range(steps_comp)]
        co2_sh = [co2_sh[i] + vars['total_co2_room'][i] / n_mods for i in
                   range(steps_comp)]
        co2_dhw = [co2_dhw[i] + vars['total_co2_dhw'][i] / n_mods for i in
                   range(steps_comp)]
        for sys in systems_ad:
            systems_ad[sys] = [systems_ad[sys][i] + vars[sys][i]/n_mods for i in
                    range(steps_comp)]

    plot_data = [
        {'x': x,
         'y': systems_ad[diff_ms[0].systems[i].type],
         'type': 'line',
         'name': diff_ms[0].systems[i].description,
         'marker':{'color': diff_ms[0].color_list[str(i)]}
         } for i in range(n)]
    plot_ages = [
        {
            'x': x,
            'y': avg_age,
            'type': 'line',
            'name': 'average heating system age',
            'marker': {'color': diff_ms[0].color_list['0']}
        }, {
            'x': x,
            'y': avg_ex_age,
            'type': 'line',
            'name': 'average age at replacement',
            'marker': {'color': diff_ms[0].color_list['1']}
        }
    ]

    data_co2 = [
        {
            'x': x,
            'y': co2_sh,
            'type': 'line',
            'name': 'total co2 emissions for space heating in t/year',
            'marker': {'color': diff_ms[0].color_list['0']}
        }, {
            'x': x,
            'y': avg_ex_age,
            'type': 'line',
            'name': 'total co2 emissions for domestic hot water in t/year',
            'marker': {'color': diff_ms[0].color_list['1']}
        }
    ]

    plot_dec = [{
            'x': x,
            'y': dec_made,
            'type': 'line',
            'name': 'number of replacements',
            'marker': {'color': diff_ms[0].color_list['0']}
        }
    ]

    rates_plot = {
                'data': plot_data,
                'layout': {
                    'title': 'Share of adopters'
                }
    }
    ages_plot = {
                'data': plot_ages,
                'layout': {
                    'title': 'Average ages',
                    'legend': {
                        'orientation': 'horizontal'
                    }
                }
    }
    dec_plot = {
        'data': plot_dec,
        'layout': {
            'title': 'Number fo replacements'
        }
    }

    plot_co2 = {
        'data': data_co2,
        'layout': {
            'title': 'CO2 emissions'
        }
    }

    color_update = {'color': diff_ms[0].hhs_color}

    return rates_plot, ages_plot, dec_plot, plot_co2


@app.callback(
    Output(component_id='geoGraph', component_property='figure'),
    [Input(component_id='button_load', component_property='n_clicks'),
     Input(component_id='interval-component', component_property='n_intervals')],
    [State('geoGraph', 'relayoutData')]
)
def load_model(input_data, interval, relayout_data):

    if not interval == 0:
        geoFig = px.scatter_mapbox(df, lat='lat', lon='lon', color=diff_ms[0].hhs_color, color_discrete_map=diff_ms[0].color_list, zoom=10)
        geoFig.update_layout(mapbox_style='open-street-map')
        geoFig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
        if 'mapbox.zoom' in relayout_data:
            geoFig['layout']['mapbox.center'] = relayout_data['mapbox.center']
            geoFig['layout']['mapbox.zoom'] = relayout_data['mapbox.zoom']
    else:
        geoFig = px.scatter_mapbox(pd.DataFrame({'lat': [48.135125], 'lon': [11.581981]}), lat='lat', lon='lon', zoom=10)
        geoFig.update_layout(mapbox_style='open-street-map')
        geoFig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})

    return geoFig

@app.callback(
    [Output('w_cost_disp', 'children'),
     Output('w_gen_att_disp', 'children'),
     Output('w_ext_thread_disp', 'children'),
     Output('w_comf_disp', 'children'),
     Output('w_peer_disp', 'children')],
    [Input('w_cost', 'value'),
     Input('w_gen_att', 'value'),
     Input('w_ext_thread', 'value'),
     Input('w_comf', 'value'),
     Input('w_peer', 'value')
    ])
def display_value(w_cost, w_gen_att, w_ext, w_comf, w_peer):
    global weights
    update_val = {'cost_aspect': w_cost,
               'general_attitude': w_gen_att,
               'external_threads': w_ext,
               'comfort': w_comf,
               'peers': w_peer}

    trans = {
        'w_cost': 'cost_aspect',
        'w_gen_att': 'general_attitude',
        'w_ext_thread':  'external_threads',
        'w_comf':  'comfort',
        'w_peer': 'peers'}

    ctx = dash.callback_context

    callback_id = ctx.triggered[0]['prop_id'].split('.')[0]

    change_w = trans[callback_id]

    weight_new = update_val[change_w]
    frac_rest_old = 1 - weights[change_w]
    frac_rest_new = 1 - weight_new

    for weight in weights:
        frac_old = weights[weight] / frac_rest_old
        weights[weight] = round(frac_old * frac_rest_new, 2)

    weights[change_w] = weight_new

    return 'w_cost: {} '.format(weights['cost_aspect']), 'w_gen_att: {} '.format(weights['general_attitude']), 'w_ext_thread: {} '.format(weights['external_threads']), 'w_comf: {} '.format(weights['comfort']), 'w_peer: {} '.format(weights['peers'])

@app.callback(Output('adoption_prob_disp', 'children'),
              [Input('adoption_prob', 'value')])
def display_value(value):
    params['adoption_prob'] = value
    return 'adoption_prob: {} '.format(value)

@app.callback(Output('age_dist_scale_disp', 'children'),
              [Input('age_dist_scale', 'value')])
def display_value(value):
    params['age_dist_scale'] = value
    return 'age_dist_scale: {} '.format(value)

@app.callback(Output('age_dist_shape_disp', 'children'),
              [Input('age_dist_shape', 'value')])
def display_value(value):
    params['age_dist_shape'] = value
    return 'age_dist_shape: {} '.format(value)

@app.callback(Output('avg_inter_disp', 'children'),
              [Input('avg_inter', 'value')])
def display_value(value):
    params['avg_inter'] = value
    return 'avg_inter: {} '.format(value)

@app.callback(Output('info_lvl_fact_disp', 'children'),
              [Input('info_lvl_fact', 'value')])
def display_value(value):
    params['info_lvl_fact'] = value
    return 'info_lvl_fact: {} '.format(value)

@app.callback(Output('info_th_disp', 'children'),
              [Input('info_th', 'value')])
def display_value(value):
    params['info_th'] = value
    return 'info_th: {} '.format(value)

@app.callback(Output('peer_th_disp', 'children'),
              [Input('peer_th', 'value')])
def display_value(value):
    params['peer_th'] = value
    return 'peer_th: {} '.format(value)

@app.callback(Output('hidden_div', 'children'),
              [Input('dec_trig_peer', 'on')])
def display_value(value):
    params['dec_trig_peer'] = value
    return 'dec_trig_peer: {} '.format(value)

@app.callback(Output('hidden_div2', 'children'),
              [Input('dcm', 'on')])
def display_value(value):
    params['dcm'] = value
    return 'dcm: {} '.format(value)

@app.callback(Output('hidden_div5', 'children'),
              [Input('sys_groups', 'on')])
def display_value(value):
    params['sys_groups'] = value
    return 'sys_groups: {} '.format(value)

@app.callback(Output('hidden_div4', 'children'),
              [Input('iterations', 'value')])
def display_value(value):
    global n_mods_1
    n_mods_1 = int(value)
    return 'ok'

@app.callback(Output('hidden_div6', 'children'),
              [Input('exit', 'n_clicks')])
def exit(value):
    if value > 0:
        func = request.environ.get('werkzeug.server.shutdown')
        if func is None:
            raise RuntimeError('Not running with the Werkzeug Server')
        func()
    return 'ok'


url = 'http://127.0.0.1:8050'
print('Interface starting at {url}'.format(url=url))
webbrowser.open(url)
if __name__ == '__main__':
    app.run_server(debug=False, use_reloader=False)



