class HeatingSystem:

    def __init__(self, name, unique_id, exp_lifetime):
        self.type = name
        self.unique_id = unique_id
        self.fuels = {}
        self.exp_lifetime = exp_lifetime
        self.eff = {}  # efficiency
        self.emis = {}  # emissions
        self.invest_cost = {}  # investment cost
        self.maint_cost = {}  # maintenance cost
        self.maint_fact = {}  # maintenance factors
        self.op_time = None  # operational time
