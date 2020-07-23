import numpy as np
import pandas as pd
from pandas import to_datetime as dt
from heating_system import HeatingSystem
from agents_geo import CentralAgent


def get_build_stock(file):
    # read sheets from xlsx file to DataFrame
    ref_stat_df = pd.read_excel(
        io=file, sheet_name='refurb_status', header=2, index_col=0
    )
    pers_SFH_df = pd.read_excel(
        io=file, sheet_name='persons_SFH',  header=2, index_col=0
    )
    temp_heat_df = pd.read_excel(
        io=file, sheet_name='temperature_heating', header=2,index_col=0
    )

    ref_stat_df.fillna('', inplace=True)

    # convert DataFrames to list
    pers_SFH_l = [pers_SFH_df.index.to_list(), pers_SFH_df['area'].to_list()]

    # convert DataFrames to dictionaries
    ref_stat_dict = ref_stat_df.to_dict('index')
    temp_heat_dict = temp_heat_df.to_dict('dict')['temperature']

    for status in ref_stat_dict:
        if ref_stat_dict[status]['upgrade_possibilities'] == '':
            ref_stat_dict[status]['upgrade_possibilities'] = []
        else:
            ref_stat_dict[status]['upgrade_possibilities'] = \
                ref_stat_dict[status]['upgrade_possibilities'].split(';')

    # map to one dictionary
    building_stock = {
        'refurb_stat': ref_stat_dict,
        'pers_SFH': pers_SFH_l,
        'temp_lvl': temp_heat_dict
    }

    return building_stock


def get_hh_char(file_directory):
    # read sheets from xlsx file to DataFrame
    cluster_fractions_df = pd.read_excel(
        io=file_directory, sheet_name='cluster', header=2, index_col=0
    )
    decision_weights_df = pd.read_excel(
        io=file_directory, sheet_name='weights', header=2, index_col=0
    )
    # opinion_init_df = pd.read_excel(
    #     io=file_directory, sheet_name='opinion', header=2, index_col=0
    # )

    # convert DataFrames to dictionaries
    cluster_fractions_dict = cluster_fractions_df.to_dict('index')
    decision_weights_dict = decision_weights_df.to_dict('dict')
    opinion_init_dict = {}  # opinion_init_df.to_dict('dict')

    # map to one dictionary
    household_characterization = {
        'clusters': cluster_fractions_dict,
        'decision_weights': decision_weights_dict,
        'opinion_init': opinion_init_dict
    }

    return household_characterization


def read_heat_sys(file):
    # read sheets from .xlsx files
    heating_system_df = pd.read_excel(
        io=file, sheet_name='heating_systems', header=0, index_col=0,
        dtype={'components first installation': np.str, 'requirements': np.str}
    )
    lifetime_df = pd.read_excel(
        io=file, sheet_name='lifetime', header=2, index_col=0
    )
    efficiency_df = pd.read_excel(
        io=file, sheet_name='efficiency', header=2, index_col=0
    )
    operational_time_df = pd.read_excel(
        io=file, sheet_name='operational_time', header=2, index_col=0, usecols='A:B'
    )
    fuels_df = pd.read_excel(
        io=file, sheet_name='fuels', header=2, index_col=0
    )
    maintenance_factors_df = pd.read_excel(
        io=file, sheet_name='maintenance_factors', header=2, index_col=0
    )
    technologies_df = pd.read_excel(
        io=file, sheet_name='technologies_available', skiprows=0, index_col=0
    )
    requ_df = pd.read_excel(
        io=file, sheet_name='requirements', header=0, index_col=0
    )
    # convert DataFrame content to strings
    requ_df.fillna('', inplace=True)
    heating_system_df.fillna('', inplace=True)

    # convert DataFrames to list
    technology_list = technologies_df.index.tolist()

    # get unique groups
    un_groups = heating_system_df.group.unique()
    groups = {group: [] for group in un_groups}

    # convert DataFrames to dictionaries
    heating_system_dict = heating_system_df.to_dict('index')
    lifetime_dict = lifetime_df.to_dict('dict')['lifetime']
    efficiency_dict = efficiency_df.to_dict('index')
    maintenance_factors_dict = maintenance_factors_df.to_dict('index')
    op_time_dict = operational_time_df.to_dict('dict')['operational_time']
    fuel_dict = fuels_df.to_dict('dict')['fuel']
    requ_dict = requ_df.to_dict('dict')['requirement']
    av_sys = technologies_df.to_dict('dict')['available']

    requ_dict = {requ: requ_dict[requ].split(';') for requ in requ_dict}

    heating_systems = []
    av_ids = []
    for i in range(len(technology_list)):
        heating_system_i = heating_system_dict[technology_list[i]]
        heating_system = HeatingSystem(
            name=technology_list[i],
            unique_id=i,
            exp_lifetime=lifetime_dict[heating_system_i['main system']]
        )
        heating_system.main_system = heating_system_i['main system']
        heating_system.dhw_system = heating_system_i['dhw system']
        heating_system.description = heating_system_i['description']
        heating_system.comps = {
            'main': heating_system_i['components main'].split(';'),
            'dhw': heating_system_i['components dhw'].split(';'),
            'first_installation': heating_system_i[
                'components first installation'].split(';')
        }
        heating_system.group = heating_system_i['group']
        # heating_system.requ = heating_system_i['requirements'].split(';')

        if not heating_system.dhw_system:
            heating_system.eff = {
                'main': efficiency_dict[heating_system.main_system],
                'dhw': efficiency_dict[heating_system.main_system]['dhw']
            }
            heating_system.fuels = {
                'main': fuel_dict[heating_system.main_system],
                'dhw': fuel_dict[heating_system.main_system]
            }
            heating_system.op_time = sum(
                op_time_dict[comp]
                for comp in heating_system.comps['main']
            )

            heating_system.maint_fact = {
                comp: maintenance_factors_dict[comp]
                for comp in heating_system.comps['main']
            }
        else:
            heating_system.eff = {
                'main': efficiency_dict[heating_system.main_system],
                'dhw': efficiency_dict[heating_system.dhw_system]['dhw']
            }
            heating_system.fuels = {
                'main': fuel_dict[heating_system.main_system],
                'dhw': fuel_dict[heating_system.dhw_system]
            }
            heating_system.op_time = (
                    sum(op_time_dict[comp]
                        for comp in heating_system.comps['main']
                        )
                    + sum(op_time_dict[comp]
                          for comp in heating_system.comps['dhw'])
            )
            heating_system.maint_fact = {
                comp: maintenance_factors_dict[comp]
                for comp
                in heating_system.comps['main'] + heating_system.comps['dhw']
            }
        if av_sys[heating_system.type]:
            av_ids.append(heating_system.unique_id)
        groups[heating_system.group].append(heating_system.unique_id)
        heating_systems.append(heating_system)

    return [heating_systems, efficiency_dict, requ_dict, groups, av_ids]


def read_param_ev(file):
    energy_p_ev_df = pd.read_excel(
        io=file, sheet_name='energy_price_evolution', header=2, index_col=0
    )
    emi_ev_df = pd.read_excel(
        io=file, sheet_name='emissions_evolution', header=2, index_col=0
    )
    emi_ev_type = pd.read_excel(
        io=file, sheet_name='emissions_evolution', usecols='B', header=0, nrows=1
    ).iloc[0, 0]
    energy_p_ev_type = pd.read_excel(
        io=file, sheet_name='energy_price_evolution', usecols='B', header=0, nrows=1
    ).iloc[0, 0]
    sub_scd_df = pd.read_excel(
        io=file, sheet_name='subsidies_schedule', header=2
    )
    ref_rate_ev_df = pd.read_excel(
        io=file, sheet_name='refurb_rate_evolution', header=2, index_col=0
    )

    # convert date input from string to datetime format
    sub_scd_df['start'] = dt(sub_scd_df['start'], format='%d/%m/%Y').dt.date
    sub_scd_df['end'] = dt(sub_scd_df['end'], format='%d/%m/%Y').dt.date
    emi_ev_df.index = dt(emi_ev_df.index, format='%d/%m/%Y').date
    energy_p_ev_df.index = dt(energy_p_ev_df.index, format='%d/%m/%Y').date
    ref_rate_ev_df.index = dt(ref_rate_ev_df.index, format='%d/%m/%y').date

    energy_p_ev_dict = energy_p_ev_df.to_dict('index')
    energy_p_ev_dict = dict(
        sorted(energy_p_ev_dict.items())
    )
    emi_ev_dict = emi_ev_df.to_dict('index')
    emi_ev_dict = dict(sorted(emi_ev_dict.items()))
    ref_rate_ev_dict = ref_rate_ev_df.to_dict('dict')['rate']

    # convert subsidies schedule
    # unique start and end dates
    unique_start = sub_scd_df.start.unique()
    unique_end = sub_scd_df.end.unique()

    dict_start = {
        date: list(sub_scd_df.loc[sub_scd_df['start'] == date].program)
        for date in unique_start
    }
    dict_start = dict(sorted(dict_start.items()))
    dict_end = {
        date: list(sub_scd_df.loc[sub_scd_df['end'] == date].program)
        for date in unique_end
    }
    dict_end = dict(sorted(dict_end.items()))
    sub_scd_dict = {'start': dict_start, 'end': dict_end}

    env_change_type = {
        'energy_price': energy_p_ev_type,
        'emissions': emi_ev_type
    }

    return [
        energy_p_ev_dict,
        emi_ev_dict,
        env_change_type,
        sub_scd_dict,
        ref_rate_ev_dict
    ]


def read_env_params(file):
    energy_p_df = pd.read_excel(
        io=file, sheet_name='energy_price', header=2, index_col=0,
        dtype={
            'energy_price': np.float64,
            'base_price': np.float64,
            'price_stability': np.float64
        }
    )
    inv_cost_df = pd.read_excel(
        io=file, sheet_name='investment_cost', header=2, index_col=0
    )
    lifetime_df = pd.read_excel(
        io=file, sheet_name='lifetime', header=2, index_col=0
    )
    subsidies_df = pd.read_excel(
        io=file, sheet_name='subsidies', header=2, index_col=0
    )
    emissions_df = pd.read_excel(
        io=file, sheet_name='emissions', header=2, index_col=0
    )
    cop_factors_df = pd.read_excel(
        io=file, sheet_name='factors_COP_hp', header=2, index_col=0
    )
    fin_params_df = pd.read_excel(
        io=file, sheet_name='financial_parameters', header=2, index_col=0
    )

    energy_p_dict = energy_p_df[
        ['energy_price', 'base_price', 'price_stability']
    ].to_dict('index')
    t_block_dict = energy_p_df[['blocking_time']].to_dict('dict')['blocking_time']
    price_ch_fact = energy_p_df[['price_ch_factor']].to_dict('dict')['price_ch_factor']
    lifetime_dict = lifetime_df.to_dict('dict')['lifetime']
    emissions_dict = emissions_df.to_dict('dict')['CO2_emissions']
    cop_factors_dict = cop_factors_df.to_dict('index')
    fin_params_dict = fin_params_df.to_dict('dict')['value']

    fin_params_dict['price_ch_dem'] = price_ch_fact

    x = list(inv_cost_df)
    investment_cost_dict = {
        index: [x, inv_cost_df.loc[index].tolist()]
        for index, row in inv_cost_df.iterrows()
    }

    # get unique index values from subsidies_df
    unique_program = subsidies_df.index.unique()
    # create dict with programs
    programs_dict = {
        program: subsidies_df[subsidies_df.index == program].set_index(
            'heat producer').to_dict('index')
        for program in unique_program
    }

    return [
        energy_p_dict,
        investment_cost_dict,
        lifetime_dict,
        programs_dict,
        emissions_dict,
        cop_factors_dict,
        t_block_dict,
        fin_params_dict
    ]


def read_heat_sys_init(file):
    technologies_df = pd.read_excel(
        io=file, sheet_name='technologies_available', skiprows=0, index_col=0
    )
    age_structure_df = pd.read_excel(
        io=file, sheet_name='age_structure', header=2
    )

    technology_fractions = technologies_df.to_dict('dict')['fraction']
    age_structure_dict = age_structure_df.to_dict('index')

    return [technology_fractions, age_structure_dict]


def read_heat_dem_ts(file_directory):
    timeseries_df = pd.read_excel(
        io=file_directory, sheet_name='timeseries', header=2, index_col=0
    )

    return timeseries_df


def read_ca(file_directory, model):
    central_agents_df = pd.read_excel(
        io=file_directory, sheet_name='central_agents', header=1
    )
    ca_scd_df = pd.read_excel(
        io=file_directory, sheet_name='central_agents_schedule', header=2
    )

    central_agents_df.replace('None', np.nan, True)

    ca_scd_df['start'] = dt(ca_scd_df['start'], format='%d/%m/%Y').dt.date
    ca_scd_df['end'] = dt(ca_scd_df['end'], format='%d/%m/%Y').dt.date

    unique_start = ca_scd_df.start.unique()
    unique_end = ca_scd_df.end.unique()

    dict_start = {
        date: list(ca_scd_df.loc[ca_scd_df['start'] == date].central_agent)
        for date in unique_start
    }
    dict_start = dict(sorted(dict_start.items()))
    dict_end = {
        date: list(ca_scd_df.loc[ca_scd_df['end'] == date].central_agent)
        for date in unique_end
    }
    dict_end = dict(sorted(dict_end.items()))
    central_agents_schedule_dict = {'start': dict_start, 'end': dict_end}

    dict_map = {True: 'active', False: 'passive'}
    central_agents = {}  # {'active': [], 'passive': []}
    central_agents_passive = []
    for index, row in central_agents_df.iterrows():
        ca = CentralAgent(index, model)
        ca.active = central_agents_df.active[index]
        ca.name = central_agents_df.name[index]
        ca.range = central_agents_df.range[index]
        ca.opinion = row.tolist()[3:]
        central_agents[ca.name] = ca

    return central_agents, central_agents_schedule_dict
