import networkx as nx
import math
import geopandas as gpd

import pickle
import numpy as np

from mesa import Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from mesa.batchrunner import BatchRunner

from mesa_geo import GeoSpace
from mesa_geo.geoagent import AgentCreator

from agents_geo import HouseholdAgent, interpol  # , CentralAgent
from global_functions import *

from scipy.stats import weibull_min, invweibull, nbinom
from operator import add

from datetime import datetime, timedelta
import time


def func_creator_tech(model, i):
    return lambda m = model: model.adopt_rates[i]/model.num_hh


def func_creator_stage(model, i):
    return lambda m = model: model.hh_stage[i]


def get_model_time(model):
    return model.time


def get_dec_made(model):
    return model.dec_made


def get_avg_exchange_age(model):
    return model.ex_cum / model.dec_made if model.dec_made else 0


def get_avg_age(model):
    cum_age = 0
    for hh in model.grid.agents:
        cum_age = cum_age + hh.heat_sys['age']

    avg_age = cum_age/model.num_hh

    return avg_age


def get_dec_trigger_f(model):
    return model.trigger['failure']


def get_dec_trigger_t(model):
    return model.trigger['think']


def get_total_heat(model):
    total_heat_room = 0
    total_heat_dhw = 0
    total_co2_room = 0
    total_co2_dhw = 0

    for hh in model.grid.agents:
        sys = model.systems[hh.heat_sys['id']]
        total_heat_room = total_heat_room + hh.heat_dem['input_room']
        total_heat_dhw = total_heat_dhw + hh.heat_dem['input_dhw']
        total_co2_room = (
            total_co2_room
            + hh.heat_dem['input_room'] * model.emis[sys.fuels['main']]
        )
        total_co2_dhw = (
            total_co2_dhw
            + hh.heat_dem['input_dhw'] * model.emis[sys.fuels['dhw']]
        )

    return {
        'total_heat_room': total_heat_room * 10 ** (-6),  # [GWh/a]
        'total_heat_dhw': total_heat_dhw * 10 ** (-6),  # [GWh/a
        'total_co2_room': total_co2_room * 10 ** (-6),  # [t/a]
        'total_co2_dhw': total_co2_dhw * 10 ** (-6)  # [t/a]
    }


def get_total_heat_room(model):
    return model.energy['total_heat_room']


def get_total_heat_dhw(model):
    return model.energy['total_heat_dhw']


def get_total_co2_room(model):
    return model.energy['total_co2_room']


def get_total_co2_dhw(model):
    return model.energy['total_co2_dhw']


def age_weibull_dist(exp_lifetime, mode, params):
    # dist = [
    #     [4, 14, 10.3],
    #     [4, 14, 10.3],
    #     [4, 14, 10.3],
    #     [4, 19, 7],
    #     [4, 24, 3.7],
    #     [4, 29, 3]
    # ]
    # if mode == 'init':
    #     [c, loc, scale] = dist[exp_lifetime]
    #
    # else:
    c = params['shape']
    loc = params['loc']
    scale = params['scale']

    return np.asscalar(weibull_min.rvs(c, loc, scale, 1))


def age_inv_weibull_dist(exp_lifetime):
    return np.asscalar(invweibull.rvs(5, 0, exp_lifetime, 1))


break_age_functions = {
    'weibull': age_weibull_dist,
    'inv_weibull': age_inv_weibull_dist
}


class AgentsGrid:

    def __init__(self):
        self.agents = []


class DiffusionModel(Model):
    """Model class for the Schelling segregation model."""

    def __init__(
            self,
            direct,
            params,
            start_date,
            t_step
    ):

        # model instances
        self.dir_shape = direct['shape']
        self.dir_ind = direct['input_f']
        self.schedule = RandomActivation(self)
        self.grid = AgentsGrid()  # GeoSpace()
        self.relations = None  # relationship graph

        self.color_list = pd.read_csv(
            direct['colors'],
            header=None,
            squeeze=True).to_dict()
        self.color_list = {
            str(key): val for key, val in self.color_list.items()
        }
        # select distribution of failure ages for heating systems
        self.failure_age = break_age_functions['weibull']
        self.age_dist_params = {
            'loc': params['age_dist_loc'],
            'scale': params['age_dist_scale'],
            'shape': params['age_dist_shape']
        }

        # model variables
        self.time = start_date.date()
        # time step size of simulation
        self.t_step = t_step  # 'y'-year, 'm'-month, 'd'-day
        # conversion factor depending on time step size
        t_fact = {'y': 1, 'm': 12, 'd': 365}
        self.t_fact = t_fact[self.t_step]

        # read heating systems available in model
        [
            self.systems,
            self.eff,
            self.sys_requ,
            self.groups,
            self.av_sys
        ] = read_heat_sys(direct['systems'])

        self.num_hh = 0  # number of households in model
        self.hh_upgrade = []  # list of upgradeable households

        # model parameters
        # share of contacts an agent interacts with in one timestep
        self.avg_inter = params['avg_inter']
        # threshold value for information level to reach stage 3
        self.info_th = params['info_th']
        # probability of adopting a system if decision was not triggered by
        # break down
        self.adoption_prob = params['adoption_prob']
        # factor weighting the influence of mismatching planned and actual cost
        # on the opinion
        self.feedb_fact = params['feedb_fact']
        # threshold for dissatisfaction trigger
        self.dis_sat_th = params['dis_sat_th']
        # threshold for peer pressure trigger
        self.peer_th = params['peer_th']
        # factor determinig loss of information per time step
        self.info_lvl_fact = params['info_lvl_fact']
        # dictionary for active decision triggers
        self.dec_trigger = {
            'failure': params['dec_trig_fail'],
            'sys_age': params['dec_trig_age'],
            'dis_sat': params['dec_trig_dis'],
            'peer_press': params['dec_trig_peer']
        }
        self.sys_groups = params['sys_groups']  # use grouped adoption rates for peer pressure
        self.dcm = params['dcm']  # use discrete choice model for selecting system

        # variables for initialisation
        self.sys_frac = {}
        self.sys_age_dist = {}
        [
            self.sys_frac,
            self.sys_age_dist
        ] = read_heat_sys_init(direct['systems'])

        # building stock data
        self.build_stock = get_build_stock(direct['build_stock'])
        # household characterization
        self.hh_char = get_hh_char(direct['households'])

        # environment parameters
        self.energy_p = {}  # energy prices
        self.inv_cost = {}  # investment costs of heating system components
        self.lifetime = {}  # expected lifetimes of components
        self.act_progs = {}  # active subsidy programs
        self.progs = {}  # subsidy programs
        self.emis = {}  # fuel specific GHG-emissions
        self.eff_fact = {}  # factors for calculation for COP of heat pumps
        self.fin_params = {}  # financial parameters

        # time series of heat demand and environment temperatures
        self.heat_dem_ts = read_heat_dem_ts(direct['timeseries'])
        self.refurb_rate = 0 # refurbishment rate
        self.refurb_surp = 0  # refurbishment surplus
        self.num_old_build = 0

        # changes in environmental parameters
        self.energy_p_ev = {}  # changes in energy prices
        # type of change in energy price and emissions (change rate, fixed value)
        self.env_change_type = {}
        self.emis_ev = {}  # changes is fuel specific emissions
        self.prog_scd = {}  # schedule for subsidy programs

        # initialize parameters
        [
            self.energy_p,
            self.inv_cost,
            self.lifetime,
            self.progs,
            self.emis,
            self.eff_fact,
            self.t_block,
            self.fin_params,
        ] = read_env_params(direct['systems'])

        [
            self.energy_p_ev,
            self.emis_ev,
            self.env_change_type,
            self.prog_scd,
            self.refurb_rate_ev
        ] = read_param_ev(direct['systems'])

        # actual efficiencies for technologies
        self.real_eff_global = self.compute_heat_input_factors()

        # model reporters
        #
        # counter variable for households in stage
        self.hh_stage = [0, 0, 0]
        # counter of households adopting each system
        self.adopt_rates = {sys.type: 0 for sys in self.systems}
        # counter for decisions made per time step
        self.dec_made = 0
        # counter for decision triggers
        self.trigger = {
            'failure': 0,
            'think': 0
        }
        # cumulative age of exchange
        self.ex_cum = 0

        # set model running
        self.running = True

        # initialize agents
        self.initialize_households()

        # get number of old buildings
        self.num_old_build = len(self.hh_upgrade)
        # dictionary for total energy consumption
        self.energy = get_total_heat(self)

        # create model reporter dictionary
        #
        # add reporters for adoption rates
        reporters = {
            sys.type: func_creator_tech(self, sys.type) for sys in self.systems
        }
        # dictionary for stage reporters
        stage_dict = {
            'Stage ' + str(i+1): func_creator_stage(self, i)
            for i in range(len(self.hh_stage))
        }
        reporters.update(stage_dict)
        reporters.update({
            'time': get_model_time,
            'dec_made': get_dec_made,
            'avg_age': get_avg_age,
            'trigger_f': get_dec_trigger_f,
            'trigger_t': get_dec_trigger_t,
            'avg_ex_age': get_avg_exchange_age,
            'total_heat_room': get_total_heat_room,
            'total_heat_dhw': get_total_heat_dhw,
            'total_co2_room': get_total_co2_room,
            'total_co2_dhw': get_total_co2_dhw
        })

        # initialize DataCollector
        self.data_coll = DataCollector(model_reporters=reporters)
        # set function for updating time based on time step
        self.update_time = update_date(self)

        # initialize central agent
        self.current_ca = {'active': {}, 'passive': {}}
        [
            self.ca,  # central agents
            self.ca_scd  # schedule for central agents
        ] = read_ca(direct['ca'], self)

        # set weights from input not from file for calibration and analysis

        print('Model init completed')

        # collect data of initial Timestep
        self.data_coll.collect(self)

    def initialize_households(self):
        # fractions of initial refurbishment status
        refurb_stat = self.build_stock['refurb_stat']
        # number of heating systems available in th model
        num_sys = len(self.systems)

        # load agent creator
        AC = AgentCreator(HouseholdAgent, {"model": self})
        # read shape file to GeoDataFrame
        self.df_geo = gpd.read_file(self.dir_shape)
        # create HouseHoldAgents from GeoDataFrame
        households = AC.from_GeoDataFrame(self.df_geo)

        # add agents to grid
        # self.grid.add_agents(households)
        self.grid.agents = households
        # compute total number of agents
        self.num_hh = len(households)

        # initialize vector for colors representing heating system
        self.hhs_color = np.asarray(['' for i in range(self.num_hh)])

        # initialize stage
        self.hh_stage[0] = self.num_hh

        # read individual input factors
        with open(self.dir_ind, 'rb') as f_in:
            ind_fact = pickle.load(f_in)

        # create random graph each household representing one edge
        # degree of family contacts
        deg_fam = nbinom.rvs(n=2.75, p=0.33, size=self.num_hh)
        # degree of friend contacts
        deg_fri = nbinom.rvs(n=1.76, p=0.23, size=self.num_hh)
        deg = [deg_fam[i] + deg_fri[i] for i in range(self.num_hh)]
        # self.relations = nx.gnp_random_graph(
        #     self.num_hh,
        #     self.avg_rel / self.num_hh,
        #     directed=False
        # )
        if sum(deg) % 2 > 0:
            deg[0] = deg[0] + 1

        self.relations = nx.configuration_model(deg)
        self.relations = nx.Graph(self.relations)

        # set up households
        for hh in self.grid.agents:  # loop through all households
            self.schedule.add(hh)  # add agent to scheduler

            # add positional information to graph edges
            self.relations.nodes[hh.unique_id]['pos'] = (
                hh.shape.centroid.x,
                hh.shape.centroid.y
            )

            # read household's neighbours
            # hh.neighbors = nbhd_data[str(hh.unique_id)]
            # hh.neighbors = [
            #     int(x) for x in hh.neighbors if str(x) != 'nan'
            # ]

            # convert neighbors ids to list
            if hh.neighbors:
                hh.neighbors = hh.neighbors.split(',')
                hh.neighbors = [
                    int(i) for i in hh.neighbors if i is not ''
                ]
            else:
                hh.neighbors = []

            # store unique_ids of related households in relation_graph
            hh.friends = [
                n for n in self.relations[hh.unique_id]
            ]

            # copy real efficiencies for household
            hh.real_eff = {
                key: self.real_eff_global[key] for key in self.real_eff_global
            }
            # compute real efficiencies depending on individual attributes
            ind_eff = self.comp_ind_eff(hh, ind_fact)
            for heater in ind_eff:
                hh.real_eff[heater] = ind_eff[heater]

            # initialise adopted heating system based on given distribution
            rand = self.random.random()  # random number for selection
            ub = 0
            for i in range(num_sys):  # loop through all heating systems
                # update upper bound each step
                ub = ub + self.sys_frac[self.systems[i].type]
                if rand <= ub:
                    # if rand_float less than upper bound assign current
                    # heating system
                    hh.heat_sys['id'] = i
                    # update color list
                    self.hhs_color[hh.unique_id] = str(i)
                    # update adoption rates
                    self.adopt_rates[self.systems[i].type] = (
                        self.adopt_rates[self.systems[i].type] + 1
                    )
                    break

            # assign number of people living in the household
            num_entr = len(self.build_stock['pers_SFH'][1])
            # search for index with minimum difference in household's living
            # area and average living area by people living in a SFH
            idx_people = min(
                range(num_entr),
                key=lambda x: abs(hh.area - self.build_stock['pers_SFH'][1][x])
            )
            # assign corresponding number of people
            hh.people = self.build_stock['pers_SFH'][0][idx_people]

            # initialise refurbishment status based on given distribution
            rand = self.random.random()  # random number for selection
            ub = 0
            for status in refurb_stat:  # loop over all possible status
                # update upper bound each step
                ub = ub + refurb_stat[status]['fraction']
                if rand <= ub:
                    # if rand_float less than upper bound assign current
                    # technology
                    hh.refurb_stat = status
                    break

            # if house can get refurbished, add possible upgrades to list
            if len(refurb_stat[status]['upgrade_possibilities']) is not 0:
                self.hh_upgrade.append(hh.unique_id)

            # compute annual heating demand and rated power of heating systems
            # for room heating and domestic hot water supply
            hh.heat_dem = self.household_heat_demand(hh)

            # initialise cluster (social group)
            rand = self.random.random()  # random number for selection
            ub = 0
            for cluster in self.hh_char['clusters']:
                ub = ub + self.hh_char['clusters'][cluster]['fraction']
                if rand <= ub:
                    # if rand_float less than upper bound assign
                    # current cluster
                    hh.cluster = cluster
                    break

            ## initialise opinions on heating systems with random  in a rang of
            ## +- 0.5 around specified opinion for cluster
            # opinion_val = self.hh_char['opinion_init'][hh.cluster]
            # hh.opinion = [
            #     self.random.uniform(
            #         opinion_val[self.systems[i].type] - 0.5,
            #         opinion_val[self.systems[i].type] + 0.5
            #     ) for i in range(num_sys)
            # ]

            # initialize opinions on heating systems randomly
            hh.opinion = [self.random.uniform(-1, 1) for i in range(num_sys)]

            # assign heating system age based on given distribution
            rand = self.random.random()
            ub = 0
            for i in range(len(self.sys_age_dist)):
                # update upper bound each step
                ub = ub + self.sys_age_dist[i]['fraction']
                if rand <= ub:
                    # if rand_float less than upper bound assign age within
                    # given range
                    hh.heat_sys['age'] = self.random.uniform(
                        self.sys_age_dist[i]['lower_limit'],
                        self.sys_age_dist[i]['upper_limit']
                    )
                    break

            # assign lifetime of heating system based on selected distribution
            # hh.heat_sys['lifetime'] = -1
            # hh.heat_sys['lifetime'] = self.failure_age(self.systems[hh.heat_sys['id']].exp_lifetime)
            it = 0
            hh.heat_sys['lifetime'] = self.failure_age(
                i, 'init', self.age_dist_params
            )
            while hh.heat_sys['lifetime'] < hh.heat_sys['age'] and it < 500:
                hh.heat_sys['lifetime'] = self.failure_age(
                    self.systems[hh.heat_sys['id']].exp_lifetime,
                    'init',
                    self.age_dist_params
                )
                it = it + 1

            # if hh.heat_sys['lifetime'] < hh.heat_sys['age']:
            #     rand = self.random.random()
            #     hh.heat_sys['lifetime'] = (
            #             hh.heat_sys['lifetime']
            #             + (1 + 0.2) * (hh.heat_sys['age'] - hh.heat_sys['lifetime'])
            #     )

            # initialise information level randomly
            hh.info_lvl = self.random.random()

            # initialise stage
            # hh.initialize_stage()

            # get favorite technology of agent
            # hh.fav_technology = hh.opinion.index(max(hh.opinion))

    def step(self):
        """Run one step of the model."""

        # refurbish houses and compare actual and planned cost for every agent
        if self.t_step == 'm' or self.t_step == 'd':
            # refurbish houses in month 6/june
            if self.time.month == 6 and self.time.day == 1:
                self.refurbish_houses()

            # check heating cost in moth 12/december
            if self.time.month == 12 and self.time.day == 1:
                for household in self.grid.agents:
                    self.compare_cost(household)

            if self.time.month == 1 and self.time.day == 1:
                self.energy = get_total_heat(self)

        else:
            self.refurbish_houses()
            for household in self.grid.agents:
                self.compare_cost(household)
            self.energy = get_total_heat(self)

        # update environmental parameters
        self.update_environment()

        # activate active central agents in random order
        ca_keys = self.current_ca['active'].keys()
        self.random.shuffle(ca_keys)
        for key in ca_keys:
            self.current_ca['active'][key].inform()

        # randomly activate household agents
        self.dec_made = 0
        self.ex_cum = 0
        self.trigger['failure'] = 0
        self.trigger['think'] = 0
        self.schedule.step()

        # remove subsidy programs and central agents, that are inactive in the
        # next time step
        self.remove_subsidies()
        self.remove_ca()

        # update model time
        self.time = self.update_time(self.time)

        # collect model data
        self.data_coll.collect(self)

    def compute_cost(self, household_id, heat_sys):
        # compute expected cost for technology for given household using annuity method
        hh = self.grid.agents[household_id]
        rat_p_room = hh.heat_dem['rat_p_room']
        rat_p_dhw = hh.heat_dem['rat_p_dhw']
        T = self.fin_params['obs_period']
        r_c = self.fin_params['price_ch_cap']
        r_d_main = self.fin_params['price_ch_dem'][heat_sys.fuels['main']]
        r_d_dhw = self.fin_params['price_ch_dem'][heat_sys.fuels['dhw']]
        r_o = self.fin_params['price_ch_op']
        q = 1 + self.fin_params['int_rate']

        # consider blocking times e.g. for heat pumps
        f_room = 24 / (24 - self.t_block[heat_sys.fuels['main']])
        f_dhw = 24 / (24 - self.t_block[heat_sys.fuels['dhw']])
        rat_p_room = math.ceil(rat_p_room * f_room)
        rat_p_dhw = math.ceil(rat_p_dhw * f_dhw)

        # annuity factor
        a = (q - 1) / (1 - q ** (-T))

        # cash value factors demand related cost
        #
        # room heating
        if q == r_d_main:
            b_d_room = T / q
        else:
            b_d_room = (1 - (r_d_main / q) ** T) / (q - r_d_main)
        # dhw
        if q == r_d_dhw:
            b_d_dhw = T / q
        else:
            b_d_dhw = (1 - (r_d_dhw / q) ** T) / (q - r_d_dhw)

        # cash value factor operation related cost
        if q == r_o:
            b_o = T/q
        else:
            b_o = (1 - (r_o / q) ** T) / (q - r_o)

        #######################################################################
        # capital related cost
        #######################################################################

        # determine the components, that need to be installed for the room
        # heating system and dhw system and assign its rated power/size
        new_components_dhw = []
        if heat_sys.dhw_system:
            new_components_dhw = [
                [comp, rat_p_dhw] for comp in heat_sys.comps['dhw']
                if comp not in self.systems[
                    hh.heat_sys['id']].comps['first_installation']
            ]
        else:
            rat_p_room = rat_p_room + rat_p_dhw

        new_components_rh = [
            [comp, rat_p_room] for comp
            in heat_sys.comps['main']
            if comp not in
               self.systems[hh.heat_sys['id']].comps['first_installation']
        ]

        new_components = new_components_rh + new_components_dhw

        # compute investment cost for all new components
        A_0_rh = [
            interpol(
                comp[1],
                self.inv_cost[comp[0]][0],
                self.inv_cost[comp[0]][1]
            ) for comp in new_components_rh
        ]
        A_0_dhw = [
            interpol(
                comp[1],
                self.inv_cost[comp[0]][0],
                self.inv_cost[comp[0]][1]
            ) for comp in new_components_dhw
        ]
        # list with investment cost for all systems
        A_0 = A_0_rh + A_0_dhw

        # compute subsidies
        A_0_sub_rh = self.compute_subsidies(
            new_components_rh,
            heat_sys.main_system,
            A_0_rh
        )
        A_0_sub_dhw = self.compute_subsidies(
            new_components_dhw,
            heat_sys.dhw_system,
            A_0_dhw
        )
        # list with subsidies for all components
        A_0_sub = A_0_sub_rh + A_0_sub_dhw

        # compute number of replacements in observation period for components
        n = [
            math.floor(T/self.lifetime[component[0]])
            if T/self.lifetime[component[0]] > 1 else 0
            for component in new_components
        ]

        # compute cash values of procured replacements
        A_n = [
            sum(
                [A_0[i] * (r_c ** (self.lifetime[new_components[i][0]] * k))
                 / (q ** (self.lifetime[new_components[i][0]] * k))
                 for k in range(n[i])]) for i in range(len(new_components))
        ]

        # compute residual values of all components
        R_w = [
            A_0[i] * (r_c ** (n[i] * self.lifetime[new_components[i][0]]))
            * (((n[i] + 1) * self.lifetime[new_components[i][0]] - T)
            / self.lifetime[new_components[i][0]]) * (1 / (q ** T))
            for i in range(len(new_components))
        ]

        # compute annuity of capital related cost
        A_NC = (sum(A_0) - sum(A_0_sub) + sum(A_n) - sum(R_w)) * a

        #######################################################################
        # demand related cost in first year
        #######################################################################

        # room heating
        A_D1_room = ((hh.heat_dem['demand_room']
                      * self.energy_p[heat_sys.fuels['main']]['energy_price']
                      / heat_sys.eff['main'][hh.refurb_stat])
                     + self.energy_p[heat_sys.fuels['main']][
                         'base_price'])
        # for dhw
        A_D1_dhw = ((hh.heat_dem['demand_dhw']
                     * self.energy_p[heat_sys.fuels['dhw']][
                         'energy_price']
                     / heat_sys.eff['dhw'])
                    +  self.energy_p[heat_sys.fuels['dhw']][
                        'base_price'])

        # annuity of demand related cost
        A_ND = (A_D1_room * b_d_room + A_D1_dhw * b_d_dhw) * a

        #######################################################################
        # operation related cost
        #######################################################################

        # operation related cost in first year for maintenance
        A_1M = [
            A_0[i] * (heat_sys.maint_fact[new_components[i][0]][
                          'f_maintenance']
                      + heat_sys.maint_fact[new_components[i][0]][
                          'f_servicing'])/100
            for i in range(len(new_components))
        ]
        A_NO = sum(A_1M) * a * b_o

        # annuity of total annual payments
        A_N = A_NC + A_ND + A_NO

        return A_N

    def compute_subsidies(self, components, heat_producer, A_0):
        # initialize list for subsidies
        A_0_subs = [0]*len(A_0)

        # get suitable subsidy programs from active programs
        programs_suit = [
            self.act_progs[program][heat_producer] for program
            in self.act_progs
            if heat_producer in self.act_progs[program]]

        # compute amount of subsidies
        for item in programs_suit:
            if item['type'] == 'fixed amount':
                # add fixed amount
                A_0_subs[0] = (A_0_subs[0] + item['funding rate'])
            if item['type'] == 'rate':
                for index, component in enumerate(components):
                    # compute amount for all components
                    A_0_subs[index] = (
                        A_0_subs[index] + A_0[index] * item['funding rate']
                    )

        return A_0_subs

    def update_environment(self):

        def update_val(value_old, parameter, val_type):
            updated_val = {
                'value': parameter,
                'change_rate': value_old * (1 + parameter)
            }
            return updated_val[val_type]

        # update environment variables

        # update energy prices
        if self.time in self.energy_p_ev:
            # change price for every fuel
            for fuel in self.energy_p_ev[self.time]:
                # update price according to relative change
                self.energy_p[fuel]['energy_price'] = update_val(
                    self.energy_p[fuel]['energy_price'],
                    self.energy_p_ev[self.time][fuel],
                    self.env_change_type['energy_price']
                )

        # update emissions
        if self.time in self.emis_ev:
            # change emissions for every fuel
            for fuel in self.emis_ev[self.time]:
                # update emissions according to relative change
                self.emis[fuel] = update_val(
                    self.emis[fuel],
                    self.emis_ev[self.time][fuel],
                    self.env_change_type['emissions']
                )

        # update refurbishment rate
        if self.time in self.refurb_rate_ev:
            # change rate
            self.refurb_rate = self.refurb_rate_ev[self.time]

        # update active subsidy programs and central agents
        self.add_subsidies()
        self.add_ca()

    def add_subsidies(self):
        # add valid subsidy programs
        start = self.prog_scd['start']

        # add programs
        if self.time in start:
            for program in start[self.time]:
                if program not in self.act_progs:
                    self.act_progs.update(
                        {program: self.progs[program]}
                    )

    def add_ca(self):
        start = self.ca_scd['start']

        # add central agents
        dict_map = {True: 'active', False: 'passive'}
        if self.time in start:
            for ca in start[self.time]:
                agent = self.ca[ca]
                if (ca not in self.current_ca['active']
                        and ca not in self.current_ca['passive']):
                    self.current_ca[dict_map[agent.active]].update(
                        {ca: agent}
                    )

    def remove_ca(self):
        end = self.ca_scd['end']

        # remove expired central agents
        if self.time in end:
            for ca in end[self.time]:
                if ca in self.current_ca['active']:
                    del self.current_ca['active'][ca]
                elif ca in self.current_ca['passive']:
                    del self.current_ca['passive'][ca]

    def remove_subsidies(self):
        end = self.prog_scd['end']

        # remove expired programs
        if self.time in end:
            for program in end[self.time]:
                if program in self.act_progs:
                    del self.act_progs[program]

    def refurbish_houses(self):
        # compute number of houses to refurbish
        # n_houses = self.refurb_rate * len(self.hh_upgrade)
        n_houses = self.refurb_rate * self.num_old_build
        # add surplus
        n_houses = n_houses + self.refurb_surp
        # compute number of houses where refurbishment is realized in
        # this time step
        n_houses_r = math.floor(n_houses)
        # compute new surplus
        self.refurb_surp = n_houses - n_houses_r

        if n_houses_r > len(self.hh_upgrade):
            n_houses_r = len(self.hh_upgrade)

        # select houses to refurbish
        hh_refurb = self.random.sample(self.hh_upgrade, n_houses_r)

        # perform refurbishment for selected houses
        for i in hh_refurb:
            hh = self.grid.agents[i]
            old_status = hh.refurb_stat
            # select new status
            new_status = self.random.choice(
                self.build_stock['refurb_stat'][old_status][
                    'upgrade_possibilities']
            )
            # upgrade household
            hh.refurb_stat = new_status
            hh.heat_dem = self.household_heat_demand(hh)

            if len(self.build_stock['refurb_stat'][new_status][
                       'upgrade_possibilities']) == 0:
                self.hh_upgrade.remove(hh.unique_id)

    def compute_heat_input_factors(self):
        # flow temperatures depending on refurbishment status
        temps = self.build_stock['temp_lvl']

        # hourly fractions of total annual heat demand
        factors_dhw = self.heat_dem_ts.dhw
        factors_room = self.heat_dem_ts.room_heating

        n = len(factors_dhw)
        input_factors = {}

        # convert temperatures to Kelvin
        temps = {status: temps[status] + 273.15 for status in temps}

        # compute input factor for every heater
        for heater in self.eff_fact:
            factors_year = {}
            for refurb_status in temps:
                if self.eff_fact[heater]['type'] == 'var':
                    # select time series of source temperatures
                    temp_source = (
                        self.heat_dem_ts[self.eff_fact[heater]['temperature']]
                        + 273.15
                    )
                    # compute real efficiency/cop time series
                    eff = self.comp_eff(
                        self.eff_fact[heater],
                        temps[refurb_status],
                        temp_source
                    )
                    if refurb_status == 'dhw':
                        input_factor_hour = [factors_dhw[i] / eff[i] for i
                                             in range(n)]
                    else:
                        input_factor_hour = [factors_room[i] / eff[i] for i
                                             in range(n)]
                    input_factor_year = sum(input_factor_hour)
                else:
                    input_factor_year = 1 / self.eff[heater][
                        refurb_status]

                factors_year[refurb_status] = input_factor_year

            input_factors[heater] = factors_year

        return input_factors

    def comp_eff(self, eff_fact, load_temp, source_temps):
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

    def comp_ind_eff(self, household, ind_fact):
        fact = self.eff_fact

        input_factors = {}
        # compute input factor for every heater
        for heater in fact:
            if fact[heater]['type'] == 'ind':
                input_factors[heater] = ind_fact[heater][
                    getattr(household, fact[heater]['temperature'])]

        return input_factors

    def household_heat_demand(self, household):
        # get variables
        ref_status = self.build_stock['refurb_stat']
        status = household.refurb_stat
        area = household.area
        sys_id = household.heat_sys['id']
        in_fact = household.real_eff
        heat_sys = self.systems[sys_id]

        # compute annual heating demand for room heating
        # [m^2 * kWh/m^2a = kWh/a]
        an_demand_room = area * ref_status[status]['heating_demand']
        # compute rated power of heating system for room heating
        # [kWh/m^2a / h/a = kW]
        rated_power_room = (
                an_demand_room / ref_status[status]['peak_load_hours']
        )

        # compute annual heating demand for dhw
        # [kWh/m^2a * m^2 = kWH/m^2a ]
        an_demand_dhw = max(7, (15 - (area * 0.04))) * area
        # compute rated power of heating system for dhw
        # [kWh/m^2a / (d/a * h/d ) = kW]
        rated_power_dhw = an_demand_dhw / (365 * 4)

        # compute required annual fuel input in [kWh/a] for room heating
        an_in_room = (an_demand_room
                      * in_fact[heat_sys.main_system][status]
        )
        # compute required annual fuel input in [kWh/a] for dhw
        if not self.systems[sys_id].dhw_system:
            an_in_dhw = (an_demand_dhw
                         * in_fact[heat_sys.main_system]['dhw'])
        else:
            an_in_dhw = (an_demand_dhw
                         * in_fact[heat_sys.dhw_system]['dhw'])

        # compute planned annual cost for heating in [euro/a]
        A_D1_room = ((an_demand_room
                      * self.energy_p[heat_sys.fuels['main']]['energy_price']
                      / heat_sys.eff['main'][status])
                     + self.energy_p[heat_sys.fuels['main']]['base_price'])
        A_D1_dhw = ((an_demand_dhw
                     * self.energy_p[heat_sys.fuels['dhw']]['energy_price']
                     / heat_sys.eff['dhw'])
                    + self.energy_p[heat_sys.fuels['dhw']]['base_price'])
        planned_annual_cost = A_D1_room + A_D1_dhw

        # store in dictionary
        heat_demand = {
            'demand_room': an_demand_room,  # annual demand for room heating
            'rat_p_room': rated_power_room,  # rated power of room heating sys
            'demand_dhw': an_demand_dhw,  # annual demand for hot water
            'rat_p_dhw': rated_power_dhw,  # rated power of hot water system
            'input_room': an_in_room,  # annual input of energy
            'input_dhw': an_in_dhw,  # annual input of energy
            'planned_cost': planned_annual_cost,  # planned annual cost
        }

        return heat_demand

    def compare_cost(self, household):
        an_in_room = household.heat_dem['input_room']
        an_in_dhw = household.heat_dem['input_dhw']
        planned_cost = household.heat_dem['planned_cost']
        heat_sys = self.systems[household.heat_sys['id']]

        # compute actual cost for heating
        A_D1_room = ((an_in_room
                      * self.energy_p[heat_sys.fuels['main']]['energy_price'])
                     + self.energy_p[heat_sys.fuels['main']]['base_price'])
        A_D1_dhw = ((an_in_dhw *
                     self.energy_p[heat_sys.fuels['dhw']]['energy_price'])
                    + self.energy_p[heat_sys.fuels['dhw']]['base_price'])

        actual_cost = A_D1_room + A_D1_dhw

        rel_dev = (planned_cost - actual_cost) / planned_cost

        # update opinion on own heating system
        household.opinion[household.heat_sys['id']] = (
                household.opinion[household.heat_sys['id']]
                + rel_dev * self.feedb_fact)
        if household.opinion[household.heat_sys['id']] > 1:
            household.opinion[household.heat_sys['id']] = 1
        if household.opinion[household.heat_sys['id']] < -1:
            household.opinion[household.heat_sys['id']] = -1


def update_date(model):

    def day(time):
        return time + timedelta(days=1)

    def month(time):
        date = time.replace(day=1)
        date += timedelta(32)
        return date.replace(day=1)

    def year(time):
        return time.replace(year=time.year + 1)

    time_step_dict = {
        'd': day,
        'm': month,
        'y': year
    }
    return time_step_dict.get(model.t_step, 'nothing')
