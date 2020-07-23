from model_geo import *
from matplotlib import pyplot as plt
import os
import time
from global_functions import *

# change working directory
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

# input and output folder names
folder_input = os.path.join(dir_path, 'input')
folder_output = os.path.join(dir_path, 'output')
path_results = os.path.join(folder_output, 'simulation_results.xlsx')

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
diffusionModel = DiffusionModel(
    direct=directories,
    params=params,
    start_date=settings['start_date'],
    t_step=settings['time_step']
)

# compute model steps
for i in range(settings['number_of_steps']):
    diffusionModel.step()

results_df = pd.DataFrame.from_dict(
    diffusionModel.data_coll.model_vars, orient='columns')
results_df.set_index(keys='time', drop=True, inplace=True)

with pd.ExcelWriter(path_results, engine='xlsxwriter', mode='w') as writer:
    results_df.to_excel(writer, sheet_name='results', index=True)
    writer.save()

print('finished')





