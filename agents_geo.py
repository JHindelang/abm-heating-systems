from mesa_geo.geoagent import GeoAgent
from mesa import Agent
from math import exp, cos, pi, sqrt
import numpy as np
from scipy.stats import weibull_min


def compr(alpha, v):
    # compromise value
    return alpha * cos(pi * v) + alpha


def diffu(gamma, v):
    # diffusion value
    return gamma * cos(pi * v) + gamma


def exper(age):
    # convert age to months
    age = age * 12
    # compute weighting function accounting for experience of partner
    h2 = 10 * prob_den_normal(age, 12, 12) + 1

    return h2


def robust(age, adopter):
    # convert age into months
    age = age * 12
    # compute robustness of change in opinion
    if adopter:
        # if agent is a adopter of the technology
        h1 = (1 + exp(-age / 4 + 5.5)) ** (-1)
    else:
        # if agent is not a adopter of the technology
        h1 = 1
    return h1


def prob_den_normal(x, mu, sigma_sq):
    # compute probability density function of normal distribution with
    # mean mu and bounded variance sigma^2
    prob_density = (1 / sqrt(2 * pi * sigma_sq) *
                    exp(-0.5 * ((x - mu) ** 2 / sigma_sq)))
    return prob_density


class HouseholdAgent(GeoAgent):
    """Household agent."""

    def __init__(self, unique_id, model, shape):
        """Create a new Household agent.

        Args:
            unique_id: Unique identifier for the agent.

        """
        super().__init__(unique_id, model, shape)
        self.active = False
        self.active_friend = False
        self.active_neighbor = False
        self.opinion = []
        self.heat_sys = {
            'id': None,
            'age': 0,  # in [a]
            'lifetime': 0  # in [a]
        }  # age in month
        self.heat_dem = {
            'demand_room': None,  # annual demand room heating [kWh/a]
            'rat_p_room': None,  # rated power room heating system [kW]
            'demand_dhw': None,  # annual demand hot water [kWh/a]
            'rat_p_dhw': None,  # rated power hot water [kW]
            'input_room': None,  # annual input of energy
            'input_dhw': None,  # annual input of energy
            'planned_cost': None,  # planned annual cost
        }
        self.info_lvl = None
        self.cluster = None
        self.friends = []
        self.neighbors = []
        self.nbhd_adopt_rates = []  # neighborhood adoption rates
        self.stage = 0
        self.refurb_stat = None
        self.people = None
        self.real_eff = None

    def step(self):
        """Advance agent one step."""
        # increase heating system age
        self.heat_sys['age'] = self.heat_sys['age'] + 1 / self.model.t_fact
        # get neighborhood adoption rates
        self.nbhd_adopt_rates = self.get_nbhd_adop_nr()

        def decision():
            self.stage_2(break_down=False)

        def stage_switch(stage):
            stage_func = {
                0: self.stage_0,
                1: self.stage_1,
                2: decision
            }
            func = stage_func.get(stage, 'nothing')
            return func()

        # activate action depending on stage
        stage_switch(self.stage)

    def stage_0(self):
        model = self.model
        triggers = model.dec_trigger
        exp_lt = self.heat_sys['lifetime']  # expected lifetime
        age = self.heat_sys['age']
        rand = model.random.random()
        nbhd_ad = self.nbhd_adopt_rates

        # update information level
        self.info_lvl = self.info_lvl * model.info_lvl_fact

        # probability of starting to think about new heating system
        def weibull_pdf(x, c, loc, scale):
            x_n = (x - loc) / scale
            if x < loc:
                return 0
            else:
                return c / scale * pow(x_n, c - 1) * np.exp(-pow(x_n, c))

        prob_think = weibull_pdf(
            self.heat_sys['age'], 3.4, 3, 19
        )

        # check trigger for decision making process
        if triggers['failure'] and (age > self.heat_sys['lifetime']):
            # if heating system failed set household to decision stage
            self.stage = 2
            model.hh_stage[0] = model.hh_stage[0] - 1
            model.hh_stage[2] = model.hh_stage[2] + 1
            self.stage_2(break_down=True)
            model.trigger['failure'] = model.trigger['failure'] + 1
        elif ((triggers['sys_age'] and (rand <= prob_think))
                or (triggers['dis_sat'] and
                    (self.opinion[self.heat_sys['id']] < model.dis_sat_th))
                or (triggers['peer_press'] and
                    nbhd_ad[self.heat_sys['id']] < 1 - model.peer_th)):
            # household starts to think about new heating system
            # set agent to information stage
            self.stage = 1
            model.hh_stage[0] = model.hh_stage[0] - 1
            model.hh_stage[1] = model.hh_stage[1] + 1
            self.stage_1()
            model.trigger['think'] = model.trigger['think'] + 1

    def stage_1(self):
        # assign variables
        model = self.model

        # check weather information level high enough to reach decision stage
        if self.info_lvl >= self.model.info_th:
            # set to decision stage
            self.stage = 2
            model.hh_stage[1] = model.hh_stage[1] - 1
            model.hh_stage[2] = model.hh_stage[2] + 1
        else:
            # contact friends to gather information
            num_friends = len(self.friends)

            # determine number of interactions for this time step
            # num_inter = model.random.randint(
            #     0,
            #     min(num_friends, model.avg_inter)
            # )
            num_inter = int(round(model.avg_inter * num_friends))

            # randomly select friends to interact with
            interact_friends = model.random.sample(self.friends, num_inter)
            # if num_inter:
            #
            # else:
            #     interact_friends = []

            # perform interaction with each friend
            for friend in interact_friends:
                self.interact(friend)

            # interact with passive central agents
            prob_ca = model.random.random()
            if prob_ca >= 0.5:
                if model.current_ca['passive']:
                    ca_name = model.random.sample(
                        list(model.current_ca['passive']), 1
                    )
                    ca = model.current_ca['passive'][ca_name[0]]
                    ca.ca_interact(self)

    def stage_2(self, break_down):
        # assign variables
        model = self.model
        adoption_prob = model.adoption_prob
        rand = model.random.random()
        sys_id = self.heat_sys['id']
        weights = model.hh_char['decision_weights'][self.cluster]

        # check if household will adopt heating system
        if rand < adoption_prob or break_down:
            # create list of viable heating systems
            # viable_sys = [
            #     sys for sys in model.systems
            #     if all([
            #         getattr(self, requ) for requ in model.requ[sys.type]
            #     ])
            # ]
            viable_sys = self.get_viable_sys()

            # compute cost for all options
            cost = [
                model.compute_cost(self.unique_id, sys) for sys in viable_sys
            ]

            # compute greenhouse gas emissions for all options
            rh_demand = self.heat_dem['demand_room']
            dhw_demand = self.heat_dem['demand_dhw']

            # ghg emission caused by room heating
            ghg_emi_room = [
                (rh_demand * model.emis[sys.fuels['main']]
                 / (sys.eff['main'][self.refurb_stat] * 1000))
                for sys in viable_sys
            ]
            # ghg emissions caused by hot water supply
            ghg_emi_dhw = [
                (dhw_demand * model.emis[sys.fuels['dhw']]
                 / (sys.eff['dhw']*1000))
                for sys in viable_sys
            ]
            # total ghg emissions
            ghg_emi = [
                ghg_emi_room[i] + ghg_emi_dhw[i]
                for i in range(len(ghg_emi_dhw))
            ]

            # compute peer influence
            nbhd_adopt = self.nbhd_adopt_rates

            # normalize vectors
            #
            # normalize opinion vector
            opinion_norm = [(self.opinion[sys.unique_id] + 1) / 2 for sys in viable_sys]

            # normalize cost, 0€ = 1, maximum_cost = 0
            max_cost = max(cost)
            min_cost = min(cost)
            cost_norm = [interpol(val, [0, max_cost], [1, 0]) for val in cost]

            # normalize maintenance effort, 0h = 1, max_h = 0
            op_times = [sys.op_time for sys in viable_sys]
            max_op_time = max(op_times)
            min_op_time = min(op_times)
            comf_norm = [
                interpol(sys.op_time, [0, max_op_time], [1, 0])
                for sys in viable_sys
            ]

            # normalize ghg-emission, 0kg/a  = 1, max_emissions = 0
            max_emi = max(ghg_emi)
            min_emi = min(ghg_emi)
            ghg_emi_norm = [
                interpol(emi, [0, max_emi], [1, 0]) for emi in ghg_emi
            ]

            # compute factor for reactions to external threads
            # get price stability factors for fuels of dhw and room heating
            energy_p = model.energy_p
            price_stab = [
                [energy_p[sys.fuels['main']]['price_stability'],
                 energy_p[sys.fuels['dhw']]['price_stability']]
                for sys in viable_sys
            ]

            def ext_threads(index):
                price_stab_et = (
                        (price_stab[index][0] * rh_demand
                         + price_stab[index][1] * dhw_demand)
                        / (rh_demand + dhw_demand)
                )
                return (price_stab_et + ghg_emi_norm[index]) / 2

            ext_th = [
                ext_threads(i) for i, sys in enumerate(viable_sys)
            ]

            max_ext_th = max(ext_th)
            ext_threads_norm = ext_th
            # ext_threads_norm = [
            #     interpol(ext_th, [0, max_ext_th], [0, 1]) for ext_th in ext_th
            # ]

            # compute utility function value for all technologies
            def uf(u_id):
                uf_val = [0]*5
                uf_val[0] = weights['cost_aspect']*cost_norm[u_id]
                uf_val[1] = weights['general_attitude'] * opinion_norm[u_id]
                uf_val[2] = weights['external_threads'] * ext_threads_norm[u_id]
                uf_val[3] = weights['comfort'] * comf_norm[u_id]
                uf_val[4] = weights['peers'] * nbhd_adopt[u_id]
                return sum(uf_val)

            utility = [uf(index) for index, system in enumerate(viable_sys)]

            if model.dcm:
                # selection of option following the discrete choice model
                # logit model
                # compute probability of choosing each technology
                sum_prob = sum(np.exp(utility))
                prob_tech = [np.exp(uf)/sum_prob for uf in utility]
                idx = np.random.choice(a=range(len(viable_sys)), p=prob_tech)
            else:
                # find system with maximum utility function value
                val, idx = max((val, idx) for (idx, val) in enumerate(utility))

            # update adoption rates and heating system
            sys = viable_sys[idx]
            model.adopt_rates[model.systems[sys_id].type] = (
                model.adopt_rates[model.systems[sys_id].type] - 1
            )
            model.adopt_rates[sys.type] = model.adopt_rates[sys.type] + 1
            # set unique id of new system
            self.heat_sys['id'] = sys.unique_id
            # update color of agent
            model.hhs_color[self.unique_id] = str(sys.unique_id)
            # add age of old system to cumulative age
            model.ex_cum = model.ex_cum + self.heat_sys['age']
            # reset age
            self.heat_sys['age'] = 0
            # compute lifetime
            self.heat_sys['lifetime'] = model.failure_age(
                sys.exp_lifetime, 'step', model.age_dist_params
            )

            # update heat demand dictionary
            self.heat_dem = model.household_heat_demand(self)
            # count decision
            model.dec_made = model.dec_made + 1

            # if sys.type == 'pellet_b' or sys.type == 'pellet_bs':
            #     print('##############')
            #     for k, ubtm in enumerate(viable_sys):
            #         print(ubtm.type, 'uf: ', utility[k],
            #             'cost:',  weights['cost_aspect']*cost_norm[k],
            #             'opinion:', weights['general_attitude'] * opinion_norm[k],
            #             'ext: ', weights['external_threads'] * ext_threads_norm[k],
            #             'comf:', weights['comfort'] * comf_norm[k],
            #             'peers:', weights['peers'] * nbhd_adopt[k],)

        # update stage
        self.stage = 0
        model.hh_stage[2] = model.hh_stage[2] - 1
        model.hh_stage[0] = model.hh_stage[0] + 1

    def interact(self, agent2):
        model = self.model
        agent_1 = self
        agent_2 = model.grid.agents[agent2]

        # select technologies agents are talking about
        # technologies = self.random.sample(
        #     range(0, len(self.systems)),
        #     self.avg_sys
        # )

        num_sys = len(model.systems)
        # select random number of systems agent has best opinion about
        n = model.random.randint(0, num_sys)
        technologies = sorted(
            range(num_sys),
            key=lambda sub: agent_1.opinion[sub]
        )[-n:]
        # add systems agents are currently using
        if agent_1.heat_sys['id'] not in technologies:
            technologies.append(agent_1.heat_sys['id'])
        if agent_2.heat_sys['id'] not in technologies:
            technologies.append(agent_2.heat_sys['id'])

        # perform interaction for each technology
        age_1 = agent_1.heat_sys['age']
        age_2 = agent_2.heat_sys['age']

        for tech in technologies:
            adopter_1 = (tech == agent_1.heat_sys['id'])
            adopter_2 = (tech == agent_2.heat_sys['id'])

            # get current opinions on technology
            v_1_old = agent_1.opinion[tech]
            v_2_old = agent_2.opinion[tech]
            # update opinions on technology
            compr_1 = (robust(age_1, adopter_1) * exper(age_2)
                       * compr(0.05, v_1_old) * (v_2_old - v_1_old))
            compr_2 = (robust(age_1, adopter_2) * exper(age_1)
                       * compr(0.05, v_2_old) * (v_1_old - v_2_old))

            diff_1 = diffu(0.05, v_1_old) * np.random.normal(0, 0.01)
            diff_2 = diffu(0.05, v_2_old) * np.random.normal(0, 0.01)

            agent_1.opinion[tech] = agent_1.opinion[tech] + compr_1 + diff_1
            agent_2.opinion[tech] = agent_2.opinion[tech] + compr_2 + diff_2

            if agent_1.opinion[tech] > 1:
                agent_1.opinion[tech] = 1
            if agent_1.opinion[tech] < -1:
                agent_1.opinion[tech] = -1

            if agent_2.opinion[tech] > 1:
                agent_2.opinion[tech] = 1
            if agent_2.opinion[tech] < -1:
                agent_2.opinion[tech] = -1

            # increase information level
            agent_1.info_lvl = agent_1.info_lvl + 0.01
            agent_2.info_lvl = agent_2.info_lvl + 0.01

    def __repr__(self):
        return 'Agent ' + str(self.unique_id)

    def initialize_stage(self):
        model = self.model
        exp_lt = self.heat_sys['lifetime']
        rand = self.model.random.random()
        prob = (1
                + exp(-0.4 * (self.heat_sys['age']
                              / model.t_fact - (exp_lt - 5)))
                ) ** (-1)
        # check if agent starts to think about new system
        if rand <= prob:
            # set agent to information stage
            self.stage = 1
            model.hh_stage[0] = model.hh_stage[0] - 1
            model.hh_stage[1] = model.hh_stage[1] + 1
            if self.info_lvl >= model.info_th:
                # set agent to decision stage
                self.stage = 2
                model.hh_stage[1] = model.hh_stage[1] - 1
                model.hh_stage[2] = model.hh_stage[2] + 1

    def get_nbhd_adop_nr(self):
        agents = self.model.grid.agents
        adopt_nr = [0] * len(self.model.systems)
        nr_neighbors = len(self.neighbors)

        # loop through all neighbors and count their heating systems
        if nr_neighbors:
            for n in self.neighbors:
                adopt_nr[agents[n].heat_sys['id']] = (
                    adopt_nr[agents[n].heat_sys['id']] + 1
                )
            # convert total numbers of adopters to fractions
            adopt_nr = [number / nr_neighbors for number in adopt_nr]

        # if peer pressure is computed by groups
        if self.model.sys_groups:
            grp_adopt = {}
            for group in self.model.groups:
                adopt_nr_g = 0
                # compute total fraction of group
                for sys in self.model.groups[group]:
                    adopt_nr_g = adopt_nr_g + adopt_nr[sys]
                grp_adopt[group] = adopt_nr_g
                # set total fraction for all group members
                for sys in self.model.groups[group]:
                    adopt_nr[sys] = adopt_nr_g

        return adopt_nr

    def get_viable_sys(self):
        model = self.model
        systems = model.systems
        requ = model.sys_requ
        heat_dem = self.heat_dem
        refurb = self.refurb_stat
        viable_sys = []

        # routine for checking weather requirements are fulfilled
        def check_requ(requ_list, mode):
            # loop through all requirements
            for r in requ_list:
                if r == '':
                    # if no requirements set system as viable
                    viable = True
                elif r == 'ground_wat':
                    thermal_room = (
                        heat_dem['rat_p_room'] * (1 + 1 / eff['main'][refurb])
                    )
                    thermal_dhw = heat_dem['rat_p_dhw'] * (1 + 1 / eff['dhw'])
                    if mode == 'comb':
                        # for room heating and dhw
                        thermal_p = thermal_room + thermal_dhw
                    if mode == 'main':
                        # for room heating only
                        thermal_p = thermal_room
                    if mode == 'dhw':
                        # for dhw only
                        thermal_p = thermal_dhw
                    # check for geothermal power at location
                    viable = (thermal_p <= getattr(self, 'gw_power'))
                elif r == 'dh_grid':
                    viable = (getattr(self, 'dist_dh') <= 50)
                elif r == 'solar':
                    thermal_room = heat_dem['demand_room'] / eff['main'][refurb]
                    thermal_dhw = heat_dem['demand_dhw'] / eff['dhw']
                    if mode == 'comb':
                        # for room heating and dhw
                        thermal_p = thermal_room + thermal_dhw
                    if mode == 'main':
                        # for room heating only
                        thermal_p = thermal_room
                    if mode == 'dhw':
                        # for dhw only
                        thermal_p = thermal_dhw
                    # check for geothermal power at location
                    viable = (thermal_p <= getattr(self, 'solar_p'))
                else:
                    viable = bool(getattr(self, r))
            return viable

        # loop through available heating systems
        for i in model.av_sys:
            sys = systems[i]
            eff = sys.eff
            requ_main = requ[sys.main_system]
            if sys.dhw_system is not '':
                requ_dhw = requ[sys.dhw_system]
                # evaluate requirment conditions
                main = check_requ(requ_main, 'main')
                dhw = check_requ(requ_dhw, 'dhw')
                #
                if main and dhw:
                    viable_sys.append(sys)
            else:
                main = check_requ(requ_main, 'comb')
                if main:
                    viable_sys.append(sys)

        return viable_sys


class CentralAgent(Agent):
    """Household agent."""

    def __init__(self, unique_id, model):
        """Create a new central agent.

        Args:
            unique_id: Unique identifier for the agent.

        """
        super().__init__(unique_id, model)
        self.active = False
        self.name = None
        self.range = 1.0
        self.opinion = [] * len(self.model.systems)

    def inform(self):
        # compute number of households getting the information
        num_inform = int(round(self.range*self.model.num_hh))

        # select a subset of households
        hhs = self.model.random.sample(self.model.grid.agents, num_inform)

        for hh in hhs:
            self.ca_interact(hh)

    def ca_interact(self, hh):
        num_tech = len(self.model.systems)

        for i in range(num_tech):
            if not np.isnan(self.opinion[i]):
                # get current opinion on technology
                v_i_old = hh.opinion[i]

                # compromise
                comp = (robust(hh.heat_sys['age'], False)
                        * compr(0.05, v_i_old) * (self.opinion[i] - v_i_old))
                # diffusion
                diff = diffu(0.05, v_i_old) * np.random.normal(0, 0.01)

                # update opinions on technology
                hh.opinion[i] = hh.opinion[i] + comp + diff
                if hh.opinion[i] > 1:
                    hh.opinion[i] = 1
                if hh.opinion[i] < -1:
                    hh.opinion[i] = -1
                hh.info_lvl = hh.info_lvl + 0.01


def interpol(x_n, x_d, y_d):
    # returns an interpolated or extrapolated value of y for x_new for
    # the given data set x_data, y_data
    if x_n < x_d[0]:
        # compute extrapolated value
        y_ex = (y_d[0]
                  + (x_n - x_d[0]) * (y_d[1] - y_d[0]) / (x_d[1] - x_d[0])
                  )
        return y_ex
    if x_n > x_d[-1]:
        # compute extrapolated value
        y_ex = (y_d[-1]
                + (x_n - x_d[-1]) * (y_d[-1] - y_d[-2]) / (x_d[-1] - x_d[-2])
                )
        return y_ex
    else:
        # This passage is adopted from scipy.interpolate
        # Find where in the range of original data, the value to interpolate
        # would be inserted
        x_new_indices = np.searchsorted(x_d, x_n)
        # Clip x_new_indices so that they are within the range of x_data
        # indices and at least 1
        x_new_indices = x_new_indices.clip(1, len(x_d) - 1).astype(int)
        lo = x_new_indices - 1
        hi = x_new_indices
        # compute interpolated value
        y_in = (y_d[lo]
                + (x_n - x_d[lo]) * (y_d[hi] - y_d[lo]) / (x_d[hi] - x_d[lo])
                )
        return y_in


