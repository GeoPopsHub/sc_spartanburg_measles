import marimo

__generated_with = "0.21.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import geopops
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    import scipy.stats as stats
    import networkx as nx
    import starsim as ss
    import sciris as sc

    return geopops, np, nx, pd, plt, sc, sns, ss


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 3.0 Explore Networks
    In this notebook, we'll:
    * 3.1 Calculate and compare statistics of each network (number of nodes and edges, mean degree)
    * 3.2 Run a simulation and look at infections by network
    * 3.3 Compare simulations when changing edge weight on a network
    * 3.4 Compare simulations when changing mean degree on a network

    ## 3.1 Compare network statistics
    `geopops.ForStarsim.GPNetwork()` takes the upper adjacency matrices for each network and stores them as edge lists, which can be turned into Starsim networks and passed into a simulation. In each edge list, a value in columns p1 or p2 corresponds to the agent uid in people_all.csv and the Starsim people object.
    """)
    return


@app.cell
def _(pd):
    # Read in edge list dataframes
    #  p1=person 1, p2=person 2, beta=edge weight
    h_df = pd.read_csv("data/pop_export/starsim/net_h.csv").drop(columns=['Unnamed: 0'])
    s_df = pd.read_csv("data/pop_export/starsim/net_s.csv").drop(columns=['Unnamed: 0'])
    w_df = pd.read_csv("data/pop_export/starsim/net_w.csv").drop(columns=['Unnamed: 0'])
    g_df = pd.read_csv("data/pop_export/starsim/net_g.csv").drop(columns=['Unnamed: 0'])
    print('View home network edgelist')
    w_df.head()
    return g_df, h_df, s_df, w_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now we'll calculate some basic network statistics in the cell below and print as a dataframe.
    """)
    return


@app.cell
def _(g_df, h_df, np, nx, pd, s_df, w_df):
    # Function to calculate network statistics
    def get_stats_df(nets, names):
        rows = []
        stat_names = ['Mean degree', 'Median degree', 'Min degree', 'Max degree', 'Number of nodes', 'Number of edges']
        for _net, name in zip(nets, names):
            G = nx.from_pandas_edgelist(_net, source='p1', target='p2', create_using=nx.Graph())
            degrees = np.array([d for _, d in G.degree()])
            values = [degrees.mean().round(2), np.median(degrees).round(2), degrees.min(), degrees.max(), G.number_of_nodes(), G.number_of_edges()]
            for stat, val in zip(stat_names, values):
                rows.append({'Network': name, 'Statistic': stat, 'GeoPops': val})
        df = pd.DataFrame(rows)
        return df.pivot(index='Statistic', columns='Network', values='GeoPops')
    stats_df = get_stats_df(nets=[h_df, s_df, w_df, g_df], names=['Household', 'School', 'Workplace', 'Group Quarters'])
    stats_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Section 1.7.2 of `1_run_geopops.ipynb` explains how each network is generated. For example, the school and workplace networks are generated using stochastic block modeling with mean degrees of 12 and 8, respectively. You can adjust the parameters used in network generation with `WriteConfig()` before running `SynthPop()`. This would change the structure of your networks and, in turn, the progression of disease in the simulation. The following code illustrates the density of the home network by age (takes a while to run). You can see genrations of children, parents, and grandparents.
    """)
    return


@app.cell
def _(pd, sns):
    _people_all = pd.read_csv('data/pop_export/people_all.csv')
    _p1 = _people_all[['uid', 'age']].rename(columns={'uid': 'p1', 'age': 'p1_age'})
    _p2 = _people_all[['uid', 'age']].rename(columns={'uid': 'p2', 'age': 'p2_age'})
    _n = 'h'
    _net = pd.read_csv('data/pop_export/starsim/net_{n}.csv')  # swap this out with 's', 'w', or 'g' to see the other networks
    _net = _net.merge(_p1, how='left', on='p1')
    _net = _net.merge(_p2, how='left', on='p2')
    sns.kdeplot(data=_net, x='p1_age', y='p2_age', fill=True, thresh=0, levels=100, cmap='mako')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The following cell counts how many agents there are in each network by age group. Just replace `n` with the network key: h=home, s=school, w=work, g=group quarters.
    """)
    return


@app.cell
def _(pd):
    _people_all = pd.read_csv('data/pop_export/people_all.csv')
    agegroup = _people_all[['uid', 'agegroup']]
    age = _people_all[['uid', 'age']]
    _n = 's'
    _net = pd.read_csv('data/pop_export/starsim/net_{n}.csv')
    _p1 = _net['p1']
    _p2 = _net['p2']
    net_uids = pd.concat([_p1, _p2])
    net_uids = net_uids.drop_duplicates().reset_index().drop(columns=['index']).rename(columns={0: 'uid'})
    net_uids = net_uids.merge(agegroup, on='uid', how='left')
    net_by_ag = net_uids.groupby('agegroup').size().reset_index().rename(columns={0: 'count'})
    net_by_ag
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3.2 Infections by network
    Now we'll create a Starsim simulation and track outcomes by network. First, let's think about how Starsim decides who gets infected each time step.

    For each time step, Starsim knows who the infectious agents are (sources) and who the susceptible agents are (targets). For every contact (i.e., network edge) between agents, there is a probability of transmission every time step if one of those agents is infectious. To calculate that probability, starsim scales the base infectiousness of the disease by:
    * The source’s relative transmissability (rel_sus) <-- will look at rel_sus and rel_trans in next notebook
    * The target’s relative susceptibility (rel_trans)
    * The effective transmission strength of that edge (i.e., edge_weight)

    To decide if a target agent gets infected, Starsim draws a uniform random number between 0 and 1 for that contact.
    If the probability is larger than the random draw, that contact is marked as having caused an infection. So, for each time step we have a list of infectious sources and a list of targets who got infected.

    When an agent enters a state (e.g., I), Starsim samples and stores the future time that agent will transmission out of that state (e.g., from dur_inf). When that stored time is reached, the agent transitions to the next state (e.g., R).

    Now, load the people and define the networks.
    """)
    return


@app.cell
def _(geopops, ss):
    # load the people object
    ppl = ss.load('data/pop_export/starsim/ppl.pkl')

    # define the networks. This also rewrites the edge lists in pop_export/starsim/net_*.csv
    h = geopops.ForStarsim.GPNetwork(name='homenet', edge_weight=1.0)
    s = geopops.ForStarsim.GPNetwork(name='schoolnet', edge_weight=1.0)
    w = geopops.ForStarsim.GPNetwork(name='worknet', edge_weight=1.0)
    g = geopops.ForStarsim.GPNetwork(name='gqnet', edge_weight=1.0)
    return g, h, ppl, s, w


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The next cell modifies Starsim's pre-built SIR class so that we can track new infections by network. This will be passed into the sim later.
    """)
    return


@app.cell
def _(np, ss):
    # Adapted SIR class that tracks new infections by network
    class SIRByNetwork(ss.SIR):
        """
        Same as Starsim's default SIR, but adds a result:
          - new_infections_by_network[t, net_idx]
        counting how many new infections occurred on each network at each timestep.
        """

        def __init__(self, pars=None, **kwargs):
            super().__init__(pars=pars, **kwargs)
            self.name = 'sir'  # stable module name for results (optional)

        def init_results(self):
            # Keep all default SIR/Infection results (n_infected, new_infections, etc.)
            super().init_results()

            n_steps = self.sim.t.npts
            n_nets = len(self.sim.networks)

            self.define_results(
                ss.Result(
                    name='new_infections_by_network',
                    module=self.name,
                    dtype=int,
                    shape=(n_steps, n_nets),   # time x network index
                    scale=True,
                    auto_plot=False,
                    label='New infections by network',
                )
            )
            return

        def step(self):
            # Use the same infection mechanism as Infection.step(), but keep 'networks' output
            new_cases, sources, networks = self.infect()

            # Convert per-case network indices into counts per network
            n_nets = len(self.sim.networks)
            if len(new_cases):
                net_arrays = []
                for arr in networks:
                    arr = np.atleast_1d(arr)
                    if arr.size:
                        net_arrays.append(arr)
                if net_arrays:
                    all_nets = np.concatenate(net_arrays)
                    counts = np.bincount(all_nets, minlength=n_nets)
                else:
                    counts = np.zeros(n_nets, dtype=int)
            else:
                counts = np.zeros(n_nets, dtype=int)

            self.results.new_infections_by_network[self.ti, :] = counts

            # Keep default behavior: set outcomes for new cases
            if len(new_cases):
                self.set_outcomes(new_cases, sources)

            return new_cases, sources, networks

    return (SIRByNetwork,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now define the Starsim sim object, run it, and plot infections by network.
    """)
    return


@app.cell
def _(SIRByNetwork, g, h, np, plt, ppl, s, sc, ss, w):
    # Define your SIR disease model and parameters
    sir_adapted = SIRByNetwork(sc.objdict(init_prev=ss.bernoulli(p=0.001), # Initial prevalence, drawn from Bernoulli distribution
                                          beta=0.05, # Disease's base infectiousness 
                                                     # scaled by rel_trans, rel_sus, and edge_weight <--will see these later
                                          dur_inf=ss.normal(7.0), # Duration of infectiousness, drawn from normal distribution
                                          p_death=ss.bernoulli(p=0.001) # Probability of death, drawn from Bernoulli distribution
                                          ))
                        
    # Run the model
    np.random.seed(123)
    sim1 = ss.Sim(
        pars    = sc.objdict(start = 0, stop  = 50, dt = 1.0),
        people  = ppl, # people object
        networks = [h,s,w,g], # networks
        diseases = [sir_adapted],  # disease model
    ).run()

    #Store the results
    res1 = sim1.results

    # Function for plotting new infections by network
    def plot_infections_by_network(res):
        infs_h = res['sir'].new_infections_by_network[:, 0]
        infs_s = res['sir'].new_infections_by_network[:, 1]
        infs_w = res['sir'].new_infections_by_network[:, 2]
        infs_g = res['sir'].new_infections_by_network[:, 3]

        plt.figure(figsize=(10, 6))
        plt.title('Infections by Network')
        plt.plot(res.timevec, infs_h, label='Home')
        plt.plot(res.timevec, infs_s, label='School')
        plt.plot(res.timevec, infs_w, label='Work')
        plt.plot(res.timevec, infs_g, label='Group Quarters')
        plt.legend()
        plt.show()

    # Plot infections by network
    plot_infections_by_network(res1)
    return plot_infections_by_network, sir_adapted


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now try redefining your edge weights in the networks and see how it affects the simulation in the cell below.
    """)
    return


@app.cell
def _(geopops, plot_infections_by_network, ppl, sc, sir_adapted, ss):
    # Redefine the networks with new edge weights
    h2 = geopops.ForStarsim.GPNetwork(name='homenet', edge_weight=2.0)
    s2 = geopops.ForStarsim.GPNetwork(name='schoolnet', edge_weight=1.0)
    w2 = geopops.ForStarsim.GPNetwork(name='worknet', edge_weight=0.5)
    g2 = geopops.ForStarsim.GPNetwork(name='gqnet', edge_weight=2.0)

    # Run the model
    sim2 = ss.Sim(
        pars    = sc.objdict(start = 0, stop  = 50, dt = 1.0),
        people  = ppl,
        networks = [h2,s2,w2,g2],
        diseases = [sir_adapted],  # use custom Measles model
    ).run()

    #Store the results
    res2 = sim2.results

    # Plot infections by network
    plot_infections_by_network(res2)
    return


if __name__ == "__main__":
    app.run()
