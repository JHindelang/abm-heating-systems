import os
from mesa_geo.geoagent import GeoAgent, AgentCreator
from mesa_geo import GeoSpace
from agents_geo import HouseholdAgent
import tkinter as tk
from tkinter import filedialog
import geopandas as gpd
from osgeo import gdal, ogr
import struct
import math
import pickle
from global_functions import *

# change working directory
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

path_geo = os.path.join(dir_path, 'input', 'geo_data')


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

# select shape file
root = tk.Tk()
root.withdraw()
# open dialogue for selecting shape file
shp_filename = filedialog.askopenfilename(
    title='Select shape file for preprocessing (.shp)',
    initialdir=os.path.join(path_geo, 'shape_file_households'),
    filetypes=(("shape files","*.shp"),("all files","*.*")))

shp_filename_save = os.path.join(
    dir_path, 'input', 'geo_data', 'shape_file_households', 'households.shp')

neighborhood_rad = params['radius']

# read shape file and convert coordinates
df_points = gpd.read_file(shp_filename)
df_points = df_points.to_crs(epsg=25832)

# read district heating shape file
dir_dh_grid = filedialog.askopenfilename(
    title='Select shape file of district heating grid (.shp)',
    initialdir=os.path.join(path_geo, 'shape_file_dh_grid'),
    filetypes=(("shape files","*.shp"),("all files","*.*")))

df_lines = gpd.read_file(dir_dh_grid)
df_lines = df_lines.to_crs(epsg=25832)

# compute distance to district heating grid for each agent
dist_dh = df_points.geometry.apply(lambda g: round(df_lines.distance(g).min(),2))
df_points['dist_dh'] = dist_dh

# convert coordinates of points
df_points = df_points.to_crs(epsg=4326)

# file names of raster files with geospatial data
src_filename_1 = os.path.join(
    dir_path, 'input', 'geo_data', 'groundwater_temperature.tif')
src_filename_2 = os.path.join(
    dir_path, 'input', 'geo_data', 'geothermal_power.tif')
src_filename_3 = os.path.join(
    dir_path, 'input', 'geo_data', 'avg_air_temp.tif')
src_filename_4 = os.path.join(
    dir_path, 'input', 'geo_data', 'solar_potential.tif')

# open shape file
ds = ogr.Open(shp_filename)
lyr = ds.GetLayer()

# pre allocation of result vectors
gw_temp = []
gw_power = []
avg_air_temp = []
solar_p = []

# get coordinates of agents
points = []
for feat in lyr:
    multipoint = feat.GetGeometryRef()

    point = multipoint.GetGeometryRef(0)  # <--- Get the point at index 0
    mx, my = point.GetX(), point.GetY()

    points.append([mx, my])

# open raster files
src_ds_1 = gdal.Open(src_filename_1)
gt_1 = src_ds_1.GetGeoTransform()
rb_1 = src_ds_1.GetRasterBand(1)

src_ds_2 = gdal.Open(src_filename_2)
gt_2 = src_ds_2.GetGeoTransform()
rb_2 = src_ds_2.GetRasterBand(1)

src_ds_3 = gdal.Open(src_filename_3)
gt_3 = src_ds_3.GetGeoTransform()
rb_3 = src_ds_3.GetRasterBand(1)

src_ds_4 = gdal.Open(src_filename_4)
gt_4 = src_ds_4.GetGeoTransform()
rb_4 = src_ds_4.GetRasterBand(1)

# read values from raster files at agents locations
for point in points:
    # Convert from map to pixel coordinates.
    # Only works for geotransforms with no rotation.
    px_1 = int((point[0] - gt_1[0]) / gt_1[1])  # x pixel
    py_1 = int((point[1] - gt_1[3]) / gt_1[5])  # y pixel

    px_2 = int((point[0] - gt_2[0]) / gt_2[1])  # x pixel
    py_2 = int((point[1] - gt_2[3]) / gt_2[5])  # y pixel

    px_3 = int((point[0] - gt_3[0]) / gt_3[1])  # x pixel
    py_3 = int((point[1] - gt_3[3]) / gt_3[5])  # y pixel

    px_4 = int((point[0] - gt_4[0]) / gt_4[1])  # x pixel
    py_4 = int((point[1] - gt_4[3]) / gt_4[5])  # y pixel

    structval_1 = rb_1.ReadRaster(px_1, py_1, 1, 1, buf_type=gdal.GDT_Float32)
    intval_1 = struct.unpack('f', structval_1)

    structval_2 = rb_2.ReadRaster(px_2, py_2, 1, 1, buf_type=gdal.GDT_Float32)
    intval_2 = struct.unpack('f', structval_2)

    structval_3 = rb_3.ReadRaster(px_3, py_3, 1, 1, buf_type=gdal.GDT_Float32)
    intval_3 = struct.unpack('f', structval_3)

    structval_4 = rb_4.ReadRaster(px_4, py_4, 1, 1, buf_type=gdal.GDT_Float32)
    intval_4 = struct.unpack('f', structval_4)

    gw_temp.append(round(intval_1[0],2))
    gw_power.append(round(intval_2[0],2))
    avg_air_temp.append(round(intval_3[0],2))
    solar_p.append(round(intval_4[0],2))

# store values in DataFrame
df_points['gw_temp'] = gw_temp
df_points['gw_power'] = gw_power
df_points['air_temp'] = avg_air_temp
df_points['solar_p'] = solar_p

# read agents as geoAgents
grid = GeoSpace()
AC = AgentCreator(HouseholdAgent, {"model": None})
households = AC.from_GeoDataFrame(df_points)

# pre allocation of result vectors
df_points['gas_grid'] = [1]*len(households)

neighbors = [None]*len(households)
num_neighbors = [0]*len(households)

# add agents from geo file to the grid
grid.add_agents(households)
for household in grid.agents:
    # get neighbors
    neighbor_list = [
        x.unique_id for x in grid.get_neighbors_within_distance(
            household, neighborhood_rad, center=False)]
    num_neighbors[household.unique_id] = len(neighbor_list)
    neighbor_list.remove(household.unique_id)
    # neighbor_list.insert(0, household.unique_id)
    list_str = ','.join([str(i) for i in neighbor_list])
    if not list_str:
        list_str = ''
    neighbors[household.unique_id] = list_str

# compute average number of agents
avg_neighbors = sum(num_neighbors) / len(households)

print('avg num neighbors: ', avg_neighbors)

# save neighbors
df_points['neighbors'] = neighbors

# save shape file

df_points.to_file(shp_filename_save)

##############################################################################
# get building stock characterizations (refurbishment status)

build_file = os.path.join(dir_path, 'input', 'building_stock.xlsx')
param_file = os.path.join(dir_path, 'input', 'heating_systems.xlsx')
ts_file = os.path.join(dir_path, 'input', 'heat_demand_timeseries.xlsx')
# temperature levels of heating
temps = get_build_stock(build_file)['temp_lvl']
# factors for computing efficiencies
fact = read_env_params(param_file)[5]
# time series of temperature and heat demand
ts = read_heat_dem_ts(ts_file)

# hourly fractions of total annual heat demand
factors_dhw = ts.dhw
factors_room = ts.room_heating

# gw_temp = [temp + 273.15 for temp in gw_temp]
# avg_air_temp = [temp + 273.15 for temp in avg_air_temp]

# convert temperatures to Kelvin
temps = {status: temps[status] + 273.15 for status in temps}

# compute temperatures
t_period = 8760  # [h]
delta_gw = 1.5  # [K] amplitude ground water temperature fluctuation
delta_f = 12  # [K] amplitude floor temperature fluctuation
t_0_gw = 1800  # [h] time constant groundwater
t_0_f = 840  # [h] time constant ground
a_e = 0.033  # [m^2/h] temperature conductivity


def temp_floor(t, t_m, d_t, t_0, t_p, z, a_e, ):
    """ ground temperatur in depth z """
    # t     time [h]
    # t_m   median air temperature [K]
    # d_t   amplitude of temperature fluctuation [K]
    # t_0   time constant [h]
    # t_p   time period [h]
    # z     depth [m]
    # a_e   temperature conductivity [m^2/h]

    temp = (t_m
            - d_t * math.exp(-z * math.sqrt(math.pi / (t_p * a_e)))
            * math.cos(2 * math.pi / t_p * (t - t_0 - (z / 2)
                                            * math.sqrt(
                        t_p / (math.pi * a_e))))
            )
    return temp


def inp_fact(temp_ts, heater):
    factors = {}
    if fact[heater]['type'] == 'ind':
        for refurb_status in temps:
            # compute cop for local temperature
            eff = comp_eff(fact[heater], temps[refurb_status], temp_ts)
            if refurb_status == 'dhw':
                input_factor_hour = [factors_dhw[i] / eff[i] for i
                                     in range(t_period)]
            else:
                input_factor_hour = [factors_room[i] / eff[i] for i
                                     in range(t_period)]
            factors[refurb_status] = sum(input_factor_hour)

    return factors


def comp_eff(eff_fact, load_temp, source_temps):
    # returns series of efficiency values
    # temps     load side temperature
    # temp_source  source temperatures
    # eff_fact  dictionary with coefficients

    if eff_fact['function'] == 'carnot':
        eff = [
            (eff_fact['a0']
             + (eff_fact['a1'] * load_temp / (load_temp - source_temp)))
            for source_temp in source_temps
        ]
    elif eff_fact['function'] == '2nd_order':
        eff = [
            (eff_fact['a0']
             + eff_fact['a1'] * source_temp
             + eff_fact['a2'] * load_temp
             + eff_fact['a3'] * source_temp ** 2
             + eff_fact['a4'] * source_temp * load_temp
             + eff_fact['a5'] * load_temp ** 2)
            for source_temp in source_temps
        ]

    return eff


def temp_gw(t, t_m, d_t, t_0, t_p):
    """ ground water temperautre """
    # t     time [h]
    # t_m   median ground water temperature [K]
    # d_t   amplitude of temperature fluctuation [K]
    # t_0   time constant [h]
    # t_p   time period [h]

    temp = (t_m
            - d_t * math.cos(2 * math.pi / t_p * (t - t_0))
            )
    return temp


temp_dev_gw = [temp_gw(i + 1, 0, delta_gw, t_0_gw, t_period)
               for i in range(t_period)]
temp_dev_vc = [temp_floor(i + 1, 0, delta_f, t_0_f, t_period, 20, a_e)
               for i in range(t_period)]
temp_dev_hc = [temp_floor(i + 1, 0, delta_f, t_0_f, t_period, 1.4, a_e)
               for i in range(t_period)]


# uniquify temperatures
unique_gw = list(set(gw_temp))
unique_air = list(set(avg_air_temp))

# compute timeseries
gw_ts = {}
hc_ts = {}
vc_ts = {}
for temp_gw in unique_gw:
    gw_ts[temp_gw] = [temp_gw + 273.15 + temp for temp in temp_dev_gw]
for temp_vc in unique_air:
    # assume pinchpoint at collector to be 2 K
    vc_ts[temp_vc] = [temp_vc + 273.15 + temp - 2 for temp in temp_dev_vc]
for temp_hc in unique_air:
    # assume pinch point at collector to be 2 K
    hc_ts[temp_hc] = [temp_hc + 273.15 + temp - 2 for temp in temp_dev_hc]

gw_fact = {ts: inp_fact(gw_ts[ts], 'heat_pump_gw') for ts in gw_ts}
vc_fact = {ts: inp_fact(vc_ts[ts], 'heat_pump_vc') for ts in vc_ts}
hc_fact = {ts: inp_fact(hc_ts[ts], 'heat_pump_hc') for ts in hc_ts}

ind_fact = {
    'heat_pump_gw': gw_fact,
    'heat_pump_vc': vc_fact,
    'heat_pump_hc': hc_fact
}

dir_ind_fact = os.path.join(dir_path, 'input', 'ind_fact.dat')
with open(dir_ind_fact, 'wb') as f_out:
    pickle.dump(ind_fact, f_out)

print('preprocessing completed')






