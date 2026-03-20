import starsim as ss
import numpy as np
import matplotlib.pyplot as plt

__all__ = ['Measles', 'custom_seeding', 'plot_measles', 'Quarantine_inf', 'get_children', 'Quarantine_sib', 'Quarantine_contacts', 'CloseSchools']


class Measles(ss.SIR):
    '''
    This class is a modified version of the Measles class in Starsim examples:
    https://github.com/starsimhub/starsim/blob/main/starsim_examples/diseases/measles.py
    It is adapted to model measles on a GeoPops population of Spartanburg County, SC.
    '''
    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.name = 'measles'  # Set a stable name so results live under res['measles']
        
        # Define parameters
        self.define_pars(
                          beta      = 0.9, # disease's base infectiousness (scaled by rel_trans, rel_sus, and edge_weight) # see custom_seeding function below
                          init_prev = ss.bernoulli(p=custom_seeding()), # Initial prevalence, drawn from Bernoulli distribution
                          dur_exp   = ss.normal(10.0), # duration of exposed state, in this model not infectious, drawn from normal distribution
                          dur_inf   = ss.normal(8.0), # duration of infectious state, drawn from normal distribution
                          p_death   = ss.bernoulli(p=0.0), # probability of death, drawn from Bernoulli distribution
                          vax_eff   = 0.97, # effectiveness of vaccine, scales rel_sus by 1-vax_eff
                          rel_sus_babies   = 1.0, # relative susceptability of ages <2
                          rel_sus_young    = 1.0, # relative susceptability of ages 2-5
                          rel_sus_school   = 1.0, # relative susceptability of ages 5-18    
                          rel_sus_adults   = 0.0, # relative susceptability of ages >18
                          rel_trans_babies = 1.0, # relative transmissability of ages <2
                          rel_trans_young  = 1.0, # relative transmissability of ages 2-5
                          rel_trans_school = 1.0, # relative transmissability of ages 5-18
                          rel_trans_adults = 1.0, # relative transmissability of ages >18
                        )
        self.update_pars(pars, **kwargs)

        self.define_states(
            ss.BoolState('exposed', label='Exposed'),
            ss.FloatArr('ti_exposed', label='Time of exposure'),
        )

    def init_results(self):
        """
        Extend default Infection/SIR results with per-network infection counts.
        """
        # First create the standard results: prevalence, new_infections, cum_infections, etc.
        super().init_results()

        # Then add a custom Result to track new infections by network at each timestep
        n_steps = self.sim.t.npts                  # Number of time steps in the simulation
        n_nets = len(self.sim.networks)            # Number of networks (e.g. home, school, work, gq)

        self.define_results(
            ss.Result(
                name='new_infections_by_network',
                module=self.name,
                dtype=int,
                shape=(n_steps, n_nets),          # time x network matrix of counts
                scale=True,
                auto_plot=False,
                label='New infections by network',
            )
        )
        return

    def step(self):
        """
        Override Infection.step() so we can record which network each infection came from.
        """
        # Create new cases using the base Infection logic
        new_cases, sources, networks = self.infect()

        # Compute counts of new infections per network for this timestep
        n_nets = len(self.sim.networks)
        if len(new_cases):
            net_arrays = []
            for arr in networks:
                arr = np.atleast_1d(arr)          # Ensure we always have at least 1-D arrays
                if arr.size:
                    net_arrays.append(arr)        # Keep only non-empty arrays
            if net_arrays:
                all_nets = np.concatenate(net_arrays)          # Network index for each new case
                counts = np.bincount(all_nets, minlength=n_nets)
            else:
                counts = np.zeros(n_nets, dtype=int)
        else:
            counts = np.zeros(n_nets, dtype=int)  # No new cases on this timestep

        # Store per-network counts into our custom result
        self.results.new_infections_by_network[self.ti, :] = counts

        # Apply prognoses/outcomes using the usual Infection step behavior
        if len(new_cases):
            self.set_outcomes(new_cases, sources)

        return new_cases, sources, networks

    def init_post(self):
        ''' Initialize post-infection states and relative susceptibility/transmissibility by age group'''
        # Run the usual Infection/SIR initialization first (seeding infections, etc.)
        super().init_post()

        # Starsim stores attributes in its own array type; convert to a NumPy array
        # so masks like np.isnan(...) are per-person boolean arrays (not scalars).
        ages = np.asarray(self.sim.people.age, dtype=float)  # per-agent ages

        # Example age groups (in years)
        dummy   = np.isnan(ages)  # agents who live outside the geo area (age is NaN)
        babies  = (ages < 2) & ~dummy
        young   = (ages >= 2) & (ages < 5) & ~dummy
        school  = (ages >= 5) & (ages < 18) & ~dummy
        adults  = (ages >= 18) & ~dummy

        # Set relative susceptibility
        self.rel_sus[babies] = self.pars.rel_sus_babies   # babies: 0.5x susceptibility
        self.rel_sus[young]  = self.pars.rel_sus_young   # young kids: 2x susceptibility
        self.rel_sus[school] = self.pars.rel_sus_school   # school-age: 1.5x
        self.rel_sus[adults] = self.pars.rel_sus_adults   # adults: baseline
        # agents who life outside geo area but commute in for work. Assume they are all adults and there fore not susceptible
        self.rel_sus[dummy] = 0.0 
        
        # If vaccinated (vax_status == 1.0), reduce susceptibility 
        vax = (self.sim.people.vax_status == 1.0)
        self.rel_sus[vax] *= (1-self.pars.vax_eff)

        # Optionally, adjust relative transmissibility similarly
        self.rel_trans[babies] = self.pars.rel_trans_babies
        self.rel_trans[young]  = self.pars.rel_trans_young
        self.rel_trans[school] = self.pars.rel_trans_school
        self.rel_trans[adults] = self.pars.rel_trans_adults

        return

    def step_state(self):
        ''' Progress exposed -> infected -> recovered -> dead '''
        # Progress exposed -> infected
        ti = self.ti
        infected = (self.exposed & (self.ti_infected <= ti)).uids
        self.exposed[infected] = False
        self.infected[infected] = True

        # Progress infected -> recovered
        recovered = (self.infected & (self.ti_recovered <= ti)).uids
        self.infected[recovered] = False
        self.recovered[recovered] = True

        # Trigger deaths
        deaths = (self.ti_dead <= ti).uids
        if len(deaths):
            self.sim.people.request_death(deaths)
        return

    def set_prognoses(self, uids, sources=None):
        """ Set prognoses for those who get infected """
        super().set_prognoses(uids, sources)
        # Skip Infection/SIR's infected flag, or undo it
        super().set_prognoses(uids, sources)
        ti = self.ti
        # Ensure they are not yet in the infected state at seeding
        self.infected[uids] = False
        self.susceptible[uids] = False
        self.exposed[uids] = True
        self.ti_exposed[uids] = ti
        self.ti_infected[uids] = ti + self.pars.dur_exp.rvs(uids)

        p = self.pars

        # Determine when exposed become infected
        self.ti_infected[uids] = ti + p.dur_exp.rvs(uids)

        # Sample duration of infection, being careful to only sample from the
        # distribution once per timestep.
        dur_inf = p.dur_inf.rvs(uids)

        # Determine who dies and who recovers and when
        will_die = p.p_death.rvs(uids)
        dead_uids = uids[will_die]
        rec_uids = uids[~will_die]
        self.ti_dead[dead_uids] = self.ti_infected[dead_uids] + dur_inf[will_die]
        self.ti_recovered[rec_uids] = self.ti_infected[rec_uids] + dur_inf[~will_die]

        return

    def step_die(self, uids):
        ''' Reset infected/recovered flags for dead agents '''
        # Reset infected/recovered flags for dead agents
        for state in ['susceptible', 'exposed', 'infected', 'recovered']:
            self.state_dict[state][uids] = False
        return

def get_children(ppl):
    sim = ss.Sim(people=ppl).init()
    ppl_df = sim.people.to_df()
    children = ppl_df.loc[ppl_df['age'] < 18].copy()
    children = children.loc[children['household'] != 0, ['uid', 'household','sch_code','age']]
    children['uid'] = children['uid'].astype(int)
    children['household'] = children['household'].astype(int)
    children['sch_code'] = children['sch_code'].astype(int)
    children = children.sort_values(by='household')
    children = children.reset_index(drop=True)
    return children
  

def custom_seeding(max_age=18, sch_codes=[450363001314], n_seeds=30):
    """Custom seeding function to seed specified number of measles infections
    in a specific age group at a specific school(s)"""
    sch_codes = np.atleast_1d(sch_codes)
    def seeding(sim, uids):
        ages   = sim.people.age[uids]
        person_sch_codes = sim.people.sch_code[uids]
        p = np.zeros(len(uids), dtype=float)
        # indices of uids that meet the criteria (age < max_age AND sch_code in sch_codes)
        eligible_idx = np.where((ages < max_age) & np.isin(person_sch_codes, sch_codes))[0]
        if len(eligible_idx):
            k = min(n_seeds, len(eligible_idx))
            chosen = np.random.choice(eligible_idx, size=k, replace=False)
            p[chosen] = 1.0
        return p
    return seeding


def plot_measles(sim, res, label=None):
    if label is None:
        label = ''

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()  # axes[0]..axes[5]

    # 1. Currently infectious over time
    ax = axes[0]
    ax.plot(res.timevec, res.measles.n_infected)
    ax.set_title(f'Currently infectious over time\n{label}')
    ax.set_xlabel('Day')
    ax.set_ylabel('Count')

    # 2. Cumulative infections over time
    ax = axes[1]
    ax.plot(res.timevec[1:], res.measles.new_infections[1:].cumsum())
    ax.set_title(f'Cumulative infections over time\n{label}')
    ax.set_xlabel('Day')
    ax.set_ylabel('Count')

    # 3. Active school edges over time
    ax = axes[2]
    q = next((iv for iv in sim.interventions.values() if hasattr(iv, 'school_edges')), None)
    if q is not None:
        school_edges_over_time = list(q.school_edges)
    else:
        baseline = float(sim.networks['schoolnet'].edges.beta.sum())
        school_edges_over_time = [baseline] * len(sim.results.timevec)

    ax.plot(range(len(school_edges_over_time)), school_edges_over_time, color='#1a365d')
    ax.set_title(f'Active school edges over time\n{label}')
    ax.set_xlabel('Day')
    ax.set_ylabel('Active edges')

    # 4. Cumulative infections by age group
    ax = axes[3]
    age_inf = sim.people.to_df()
    age_inf = age_inf.loc[age_inf['measles.ti_infected'] > 0.0]
    age_inf.loc[(age_inf['age'] > 0) & (age_inf['age'] < 5), 'ag_bar'] = '0-4'
    age_inf.loc[(age_inf['age'] >= 5) & (age_inf['age'] < 12), 'ag_bar'] = '5-11'
    age_inf.loc[(age_inf['age'] >= 12) & (age_inf['age'] < 18), 'ag_bar'] = '12-17'
    age_inf.loc[(age_inf['age'] >= 18) & (age_inf['age'] < 30), 'ag_bar'] = '18-29'
    age_inf.loc[(age_inf['age'] >= 30) & (age_inf['age'] < 49), 'ag_bar'] = '30-49'
    age_inf.loc[(age_inf['age'] >= 49), 'ag_bar'] = '50+'
    age_inf = age_inf.groupby('ag_bar').size()

    order = ['0-4', '5-11', '12-17', '18-29', '30-49', '50+']
    age_inf = age_inf.reindex(order, fill_value=0)

    ax.bar(age_inf.index, age_inf.values, color='#1a365d', edgecolor='none')
    ax.set_title(f'Cum infections by age\n{label}')
    ax.set_xlabel('Age group')
    ax.set_ylabel('Count')
    ax.tick_params(axis='x', rotation=45)

    # 5. Cumulative infections by vax_status (counts)
    ax = axes[4]
    vax_inf = sim.people.to_df()
    vax_inf = vax_inf.loc[vax_inf['measles.ti_infected'] > 0.0]
    counts = vax_inf.groupby('vax_status').size().reindex([0, 1], fill_value=0)

    ax.bar([0, 1], counts.values, color='#1a365d', edgecolor='none')
    ax.set_xticks([0, 1])
    ax.set_title(f'Cumulative infections by vax_status\n{label}')
    ax.set_xlabel('Vax status')
    ax.set_ylabel('Count')

    # 6. Attack rate by vax_status
    ax = axes[5]
    df = sim.people.to_df()
    total_by_vax = df.groupby('vax_status').size().reindex([0, 1], fill_value=0)
    infected_by_vax = df.loc[df['measles.ti_infected'] > 0.0].groupby('vax_status').size().reindex([0, 1], fill_value=0)

    attack_rate_by_vax = infected_by_vax / total_by_vax.replace(0, np.nan)  # avoid div-by-zero
    ax.bar([0, 1], attack_rate_by_vax.values, color='#1a365d', edgecolor='none')
    ax.set_xticks([0, 1])
    ax.set_title(f'Attack rate by vax_status\n{label}')
    ax.set_xlabel('Vax status')
    ax.set_ylabel('Attack rate')

    fig.tight_layout()
    plt.show()

class Quarantine_inf(ss.Intervention):
    def __init__(self, *args,
                 quarantine_start=0,
                 days_since_infectious=1,
                 dur_quarantine=5,
                 compliance=1.0,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.quarantine_start = quarantine_start
        self.days_since_infectious = days_since_infectious
        self.dur_quarantine = dur_quarantine
        self.compliance = compliance

        self.school_edges = []
        self.quarantined = []   # <- add this
        self._beta0 = None      # baseline school network edge weights

    def init_pre(self, sim):
        super().init_pre(sim)
        # Snapshot original edge weights so we can restore them each step
        net = sim.networks['schoolnet']
        self._beta0 = net.edges.beta.copy()
        return

    def step(self):
        sim = self.sim
        net = sim.networks['schoolnet']
        ti = sim.ti

        # Restore baseline before applying any quarantine edge changes
        if self._beta0 is None:
            self._beta0 = net.edges.beta.copy()
        net.edges.beta[:] = self._beta0

        # Before quarantine_start or if quarantine disabled: record and return
        if ti < self.quarantine_start or self.dur_quarantine <= 0:
            self.school_edges.append(np.count_nonzero(net.edges.beta))
            self.quarantined.append(0)  # nobody quarantining
            return

        infected = sim.people.measles.infected
        ti_infected = sim.people.measles.ti_infected

        days_since_infected = ti - ti_infected
        quarantined = (
            infected
            & (days_since_infected >= self.days_since_infectious)
            & (days_since_infected < self.days_since_infectious + self.dur_quarantine)
        )
        quarantined_uids = ss.uids(quarantined)

        # Apply compliance
        if len(quarantined_uids) > 0:
            comply = np.random.random(len(quarantined_uids)) < self.compliance
            actually_quarantined_uids = quarantined_uids[comply]
        else:
            actually_quarantined_uids = np.array([], dtype=int)

        # Track how many are actually quarantining this step
        self.quarantined.append(len(actually_quarantined_uids))

        p1 = net.edges.p1
        p2 = net.edges.p2
        mask_disable = np.isin(p1, actually_quarantined_uids) | np.isin(p2, actually_quarantined_uids)
        net.edges.beta[mask_disable] = np.zeros_like(net.edges.beta[mask_disable])

        self.school_edges.append(np.count_nonzero(net.edges.beta))
        return
    
class Quarantine_sib(ss.Intervention):
    def __init__(self, *args,
                 quarantine_start=0,
                 days_since_infectious=1,
                 days_quarantine=7,
                 compliance=0.5,
                 children_df=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.quarantine_start = quarantine_start
        self.days_since_infectious = days_since_infectious
        self.days_quarantine = days_quarantine
        self.compliance = compliance
        self.children = children_df

        self.school_edges = []
        self.quarantined = []
        self.quarantined_uids = []
        self.infected_uids = []
        self._beta0 = None      # baseline school network edge weights

    def init_pre(self, sim):
        super().init_pre(sim)
        net = sim.networks['schoolnet']
        self._beta0 = net.edges.beta.copy()
        return

    def step(self):
        sim = self.sim
        net = sim.networks['schoolnet']
        ti = sim.ti

        if self.children is None:
            raise RuntimeError("Pass children_df=children when constructing Quarantine_sib")

        # Restore baseline before applying any quarantine edge changes
        if self._beta0 is None:
            self._beta0 = net.edges.beta.copy()
        net.edges.beta[:] = self._beta0

        # Before quarantine_start or if disabled: just record and return
        if ti < self.quarantine_start or self.days_quarantine <= 0:
            self.school_edges.append(np.count_nonzero(net.edges.beta))
            self.quarantined.append(0)
            return

        infected = sim.people.measles.infected          # Starsim BoolState
        ti_infected = sim.people.measles.ti_infected    # Starsim FloatArr
        days_since_infected = ti - ti_infected
        self.infected_uids.append(np.where(infected)[0])

        # Base infected kids to quarantine (exactly like Quarantine_inf)
        base_quarantined = (
            infected
            & (days_since_infected >= self.days_since_infectious)
            & (days_since_infected < self.days_since_infectious + self.days_quarantine)
        )

        base_uids = ss.uids(base_quarantined)  # Starsim UIDs
        # "Index" cases are those infectious children in the quarantine window.
        # Per updated requirement, the index always quarantines; compliance applies only to siblings.
        index_uids = np.asarray(base_uids, dtype=int)

        # Add siblings (using your children_df)
        if len(base_uids):
            children = self.children
            infected_children = children[children['uid'].isin(base_uids)]
            hh_ids = infected_children['household'].unique()
            sib_uids = children[children['household'].isin(hh_ids)]['uid'].to_numpy(dtype=int)

            # union of base_uids and sib_uids
            all_uids = np.union1d(np.array(base_uids, dtype=int), sib_uids).astype(int)
        else:
            all_uids = np.array([], dtype=int)

        # Compliance applies only to siblings (contacts excluding the index)
        if all_uids.size == 0:
            actually_quarantined_uids = np.array([], dtype=int)
        else:
            sib_only_uids = np.setdiff1d(all_uids, index_uids)

            if self.compliance >= 1.0 or sib_only_uids.size == 0:
                actually_quarantined_uids = all_uids
            else:
                comply = np.random.random(sib_only_uids.size) < self.compliance
                actually_quarantined_uids = np.union1d(index_uids, sib_only_uids[comply]).astype(int)

        self.quarantined_uids.append(actually_quarantined_uids)

        self.quarantined.append(int(actually_quarantined_uids.size))

        # Disable school edges touching quarantined UIDs
        p1 = net.edges.p1
        p2 = net.edges.p2
        if actually_quarantined_uids.size > 0:
            mask_disable = np.isin(p1, actually_quarantined_uids) | np.isin(p2, actually_quarantined_uids)
        else:
            mask_disable = np.zeros(len(p1), dtype=bool)
        net.edges.beta[mask_disable] = np.zeros_like(net.edges.beta[mask_disable])

        self.school_edges.append(np.count_nonzero(net.edges.beta))
        return
    
class Quarantine_contacts(ss.Intervention):
    """
    Quarantine infectious index children + their home (household) + school (sch_code) contacts
    using children_df, BUT only disable SCHOOL network edges for quarantined agents.

    Timing logic matches the usual pattern:
      - identify "index" quarantining children when
          days_since_infected in [days_since_infectious, days_since_infectious + dur_quarantine)
    """
    def __init__(self, *args,
                 quarantine_start=0,
                 days_since_infectious=1,
                 dur_quarantine=5,
                 compliance=1.0,
                 children_df=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.quarantine_start = quarantine_start
        self.days_since_infectious = days_since_infectious
        self.dur_quarantine = dur_quarantine
        self.compliance = compliance
        self.children = children_df  # expects columns: ['uid','household','sch_code',...]
        self.school_edges = []
        # Keep these attribute names consistent with other quarantine interventions
        # so the notebook can plot `qX.quarantined` and `qX.school_edges`.
        self.quarantined = []
        self._beta0 = None      # baseline school network edge weights

    def init_pre(self, sim):
        super().init_pre(sim)
        net = sim.networks['schoolnet']
        self._beta0 = net.edges.beta.copy()
        return

    def step(self):
        sim = self.sim
        ti = sim.ti

        if self.children is None:
            raise RuntimeError("Pass children_df=children_df when constructing this intervention")

        net = sim.networks['schoolnet']
        p1 = net.edges.p1
        p2 = net.edges.p2

        # Restore baseline before applying any quarantine edge changes
        if self._beta0 is None:
            self._beta0 = net.edges.beta.copy()
        net.edges.beta[:] = self._beta0

        # If not started or disabled: just record and do nothing
        if ti < self.quarantine_start or self.dur_quarantine <= 0:
            self.school_edges.append(np.count_nonzero(net.edges.beta))
            self.quarantined.append(0)
            return

        infected = sim.people.measles.infected
        ti_infected = sim.people.measles.ti_infected
        days_since_infected = ti - ti_infected

        # Index infectious children in the quarantine window
        base_quarantined = (
            infected
            & (days_since_infected >= self.days_since_infectious)
            & (days_since_infected < self.days_since_infectious + self.dur_quarantine)
        )
        base_uids = ss.uids(base_quarantined)  # Starsim UIDs as numpy ints

        if len(base_uids) == 0:
            self.school_edges.append(np.count_nonzero(net.edges.beta))
            self.quarantined.append(0)
            return

        children = self.children
        index_children = children[children['uid'].isin(base_uids)]

        # Home contacts (household mates) + school contacts (same school code)
        hh_ids = index_children['household'].unique()
        sch_ids = index_children['sch_code'].unique()

        home_contact_uids = children[children['household'].isin(hh_ids)]['uid'].to_numpy(dtype=int)
        school_contact_uids = children[children['sch_code'].isin(sch_ids)]['uid'].to_numpy(dtype=int)

        # Quarantined set = base index children + home contacts + school contacts
        all_uids = np.unique(
            np.concatenate([np.asarray(base_uids, dtype=int), home_contact_uids, school_contact_uids])
        ).astype(int)

        # Compliance applies only to contacts excluding the index
        index_uids = np.asarray(base_uids, dtype=int)
        contact_only_uids = np.setdiff1d(all_uids, index_uids)

        if self.compliance >= 1.0 or contact_only_uids.size == 0:
            quarantined_uids = all_uids
        else:
            comply = np.random.random(contact_only_uids.size) < self.compliance
            quarantined_uids = np.union1d(index_uids, contact_only_uids[comply]).astype(int)

        self.quarantined.append(int(quarantined_uids.size))

        # IMPORTANT: disable ONLY schoolnet edges touching quarantined agents
        if quarantined_uids.size > 0:
            mask_disable = np.isin(p1, quarantined_uids) | np.isin(p2, quarantined_uids)
        else:
            mask_disable = np.zeros(len(p1), dtype=bool)

        net.edges.beta[mask_disable] = np.zeros_like(net.edges.beta[mask_disable])

        self.school_edges.append(np.count_nonzero(net.edges.beta))

class CloseSchools(ss.Intervention):
    def __init__(self, *args,
                 days_since_infectious=1,
                 days_quarantine=7,
                 children_df=None,
                 **kwargs):
        """
        Close entire schools for a fixed window after any student becomes infectious.

        Args:
            days_since_infectious: offset (in days) after infection when school closure starts
            days_quarantine: duration (in days) that affected schools remain closed
            children_df: DataFrame with at least ['uid', 'sch_code'] for school-attending agents
        """
        super().__init__(*args, **kwargs)
        self.days_since_infectious = days_since_infectious
        self.days_quarantine = days_quarantine
        self.children = children_df  # uid, household, sch_code (uid as int)

        # Tracking
        self.school_edges = []          # number of active school edges per day
        # Number of students quarantined each day (i.e., students attending schools that are closed today)
        # Kept for compatibility with notebook plots.
        self.quarantined = []
        self.closed_schools = []        # list of sets of closed schools per day (optional)

        # Internal: map uid -> school id, and school id -> student uids
        self._uid_to_school = None
        self._school_to_uids = None
        self._school_closed_until = {}  # sch_code -> day (float) until which it remains closed
        self._beta0 = None               # baseline school network edge weights

    def init_pre(self, sim):
        super().init_pre(sim)
        if self.children is None:
            raise RuntimeError("CloseSchool requires children_df with columns ['uid', 'sch_code'].")

        df = self.children.copy()
        df = df.dropna(subset=['sch_code'])
        df['uid'] = df['uid'].astype(int)
        df['ssch_codeh_id'] = df['sch_code'].astype(int)

        # uid -> sch_code
        self._uid_to_school = dict(zip(df['uid'].values, df['sch_code'].values))

        # sch_code -> np.ndarray of uids
        self._school_to_uids = {
            sch: grp['uid'].to_numpy(dtype=int)
            for sch, grp in df.groupby('sch_code')
        }

        # Snapshot baseline edge weights so we can restore them each step
        net = sim.networks['schoolnet']
        self._beta0 = net.edges.beta.copy()
        return

    def step(self):
        sim = self.sim
        net = sim.networks['schoolnet']
        ti = sim.ti

        # Shortcuts to network arrays
        p1 = net.edges.p1
        p2 = net.edges.p2
        beta = net.edges.beta

        # Restore baseline before applying any school closure edge changes
        if self._beta0 is None:
            self._beta0 = beta.copy()
        beta[:] = self._beta0

        # --- 1) Update which schools should be closed based on current infections ---

        infected = sim.people.measles.infected
        ti_infected = sim.people.measles.ti_infected
        days_since_infected = ti - ti_infected

        # Base infectious students in the quarantine window
        base_quarantined = (
            infected
            & (days_since_infected >= self.days_since_infectious)
            & (days_since_infected < self.days_since_infectious + self.days_quarantine)
        )
        base_uids = ss.uids(base_quarantined)  # Starsim UIDs

        # For any such infected student with a known school, extend that school’s closure window
        if len(base_uids):
            for uid in base_uids:
                uid_int = int(uid)
                sch_code = self._uid_to_school.get(uid_int, None)
                if sch_code is None:
                    continue
                # Close this school until (ti + days_quarantine) at least
                current_until = self._school_closed_until.get(sch_code, -np.inf)
                self._school_closed_until[sch_code] = max(current_until, ti + self.days_quarantine)

        # --- 2) Determine which schools are closed on this day ---

        closed_schools_today = {
            sch for sch, t_until in self._school_closed_until.items() if ti < t_until
        }

        # Optionally record closed school set
        self.closed_schools.append(closed_schools_today.copy())

        # --- 3) Compute which UIDs are in closed schools today ---

        if closed_schools_today:
            # Union of all student UIDs for closed schools
            closed_uids_list = [
                self._school_to_uids[sch]
                for sch in closed_schools_today
                if sch in self._school_to_uids
            ]
            if closed_uids_list:
                closed_uids = np.unique(np.concatenate(closed_uids_list))
            else:
                closed_uids = np.empty(0, dtype=int)
        else:
            closed_uids = np.empty(0, dtype=int)

        # --- 3b) Track quarantined students today ---
        self.quarantined.append(int(closed_uids.size))

        # --- 4) Apply closure to the school network edges ---

        if closed_uids.size > 0:
            mask_disable = np.isin(p1, closed_uids) | np.isin(p2, closed_uids)
        else:
            mask_disable = np.zeros(len(p1), dtype=bool)

        beta[mask_disable] = np.zeros_like(beta[mask_disable])

        # --- 5) Track active school edges ---

        self.school_edges.append(int(np.count_nonzero(beta)))
        return
