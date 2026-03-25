import marimo

__generated_with = "0.21.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import sys, platform, pathlib
    sys.path.insert(0, str(pathlib.Path().resolve().parent.parent / "src"))
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
    * 3.2 Look at contacts by age by network
    * 3.3 Run a simulation and look at infections by network
    * 3.4 Compare simulations when changing edge weight on a network

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
    h_df.head()
    return g_df, h_df, s_df, w_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The cell below calculates some basic network statistics and prints them in dataframe.
    """)
    return


@app.cell
def _(g_df, h_df, np, nx, pd, s_df, w_df):
    # Function to calculate network statistics
    def get_stats_df(nets, names):
        rows = []
        stat_names = [
            "Mean degree",
            "Median degree",
            "Min degree",
            "Max degree",
            "Number of nodes",
            "Number of edges",
        ]

        for net, name in zip(nets, names):
            G = nx.from_pandas_edgelist(net, source="p1", target="p2", create_using=nx.Graph())
            degrees = np.array([d for _, d in G.degree()])
            values = [
                degrees.mean().round(2),
                np.median(degrees).round(2),
                degrees.min(),
                degrees.max(),
                G.number_of_nodes(),
                G.number_of_edges(),
            ]
            for stat, val in zip(stat_names, values):
                rows.append({"Network": name, "Statistic": stat, "GeoPops": val})

        df = pd.DataFrame(rows)
        return df.pivot(index="Statistic", columns="Network", values="GeoPops")

    stats_df = get_stats_df(
        nets=[h_df, s_df, w_df, g_df],
        names=["Household", "School", "Workplace", "Group Quarters"],
    )
    stats_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Section 1.7.2 of `1_run_geopops.ipynb` explains how each network is generated. For example, the school and workplace networks are generated using Stochastic Block Modeling with mean degrees of 12 and 8, respectively. You can adjust the network algorithm parameters with `WriteConfig()` before running `SynthPop()`. This would change the structure of your networks and, in turn, the progression of disease in the simulation.

    ## 3.2 Contacts by age
    The next cell calculates the average number of contacts by age group for each network. This is comparible to the plots made by [Epistorm-Mix](https://www.epistorm.org/data/epistorm-mix) in Figure 2A of their [manuscript](https://www.medrxiv.org/content/10.1101/2025.11.20.25340662v1.full.pdf) (See page 8). The Epistorm-Mix study created contact matrices by age groups for different locations including home, school, and work from survey data. The Epistorm-Mix contact matrices are meant to be nationally representative of the US. The same plot created with synthesized GeoPops data of Spartanburg are not quite the same, but not far off!
    """)
    return


@app.cell
def _(pd, plt):
    _people_all = pd.read_csv('data/pop_export/people_all.csv')
    _people_all['uid'] = _people_all.index
    _people_all['agegroup'] = _people_all['age'].apply(lambda x: '00-04' if x < 5 else '05-09' if x < 10 else '10-14' if x < 15 else '15-19' if x < 20 else '20-24' if x < 25 else '25-29' if x < 30 else '30-34' if x < 35 else '35-39' if x < 40 else '40-44' if x < 45 else '45-49' if x < 50 else '50-54' if x < 55 else '55-59' if x < 60 else '60-64' if x < 65 else '65-69' if x < 70 else '70-74' if x < 75 else '75-79' if x < 80 else '80+')
    _people_all = _people_all.loc[~_people_all['age'].isnull()]
    uid_agegroup = _people_all[['uid', 'agegroup']].drop_duplicates()
    uid_agegroup = uid_agegroup.loc[~uid_agegroup['agegroup'].isnull()]
    uid_agegroup['n_people'] = 1
    n_agegroup = uid_agegroup.groupby('agegroup')['n_people'].sum().reset_index()

    def avg_contacts(n):
        net = pd.read_csv(f'data/pop_export/starsim/net_{n}.csv').drop(columns=['Unnamed: 0'])
        net = net.drop(columns=['edge_weight'])
        all_people = pd.concat([net['p1'], net['p2']])
        contact_counts = all_people.value_counts().rename('n_contacts').reset_index().rename(columns={'index': 'uid'})
        contact_counts = contact_counts.merge(uid_agegroup, how='left', on='uid')
        contact_counts = contact_counts.loc[~contact_counts['agegroup'].isnull()]  # edgelist has columns 'p1' and 'p2'
        avg_contacts = contact_counts.groupby('agegroup')['n_contacts'].sum().reset_index()
        avg_contacts = avg_contacts.merge(n_agegroup, how='left', on='agegroup')
        avg_contacts['avg_contacts'] = avg_contacts['n_contacts'] / avg_contacts['n_people']
        return avg_contacts
    home = avg_contacts('h')  # counts appearances as p1 or p2
    school = avg_contacts('s')  # name the count column
    work = avg_contacts('w')
    gq = avg_contacts('g')
    plt.figure(figsize=(10, 6))
    plt.title('Average Contacts by Age Group')
    plt.plot(home['agegroup'], home['avg_contacts'], label='Home')
    plt.plot(school['agegroup'], school['avg_contacts'], label='School')
    plt.plot(work['agegroup'], work['avg_contacts'], label='Work')
    plt.plot(gq['agegroup'], gq['avg_contacts'], label='Group Quarters')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The next cell prints the density of contacts by age for each GeoPops network (it takes 8 min time to run). The density plots below are comparible to the [Epistorm-Mix](https://www.epistorm.org/data/epistorm-mix) GAM plots (see Figure 2B-D on page 8 of the Epistorm-Mix [manuscript](https://www.medrxiv.org/content/10.1101/2025.11.20.25340662v1.full.pdf)).
    """)
    return


@app.cell
def _(pd, plt, sns):
    networks = [('h', 'Home'), ('s', 'School'), ('w', 'Work'), ('g', 'Group Quarters')]
    _people_all = pd.read_csv('data/pop_export/people_all.csv')

    def make_kde_plot():
        p1 = _people_all[['uid', 'age']].rename(columns={'uid': 'p1', 'age': 'p1_age'})
        p2 = _people_all[['uid', 'age']].rename(columns={'uid': 'p2', 'age': 'p2_age'})
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        for ax, (n, title) in zip(axes.flat, networks):
            net = pd.read_csv(f'data/pop_export/starsim/net_{n}.csv')
            if 'Unnamed: 0' in net.columns:
                net = net.drop(columns=['Unnamed: 0'])
            net = net.merge(p1, how='left', on='p1')
            net = net.merge(p2, how='left', on='p2')
            sns.kdeplot(data=net, x='p1_age', y='p2_age', fill=True, thresh=0, levels=30, cmap='mako', bw_adjust=2.0, gridsize=200, warn_singular=False, ax=ax)
            ax.set_title(title)
        fig.tight_layout()
        return fig
    fig = make_kde_plot()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Older age groups are represented in the home, school, and work networks but because the contacts are so sparse, the kde plot doesn't show them. The group quarters network looks the way it does because GeoPops assigns all agents within incarceration facilities to be 30 years old and all agents in nursing homes to be 75 years old.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3.3 Infections by network
    Now we'll create a Starsim simulation and track outcomes by network. First, let's think about how Starsim decides who gets infected each time step.

    For each time step, Starsim knows who the infectious agents are (sources) and who the susceptible agents are (targets). For every contact (i.e., network edge) between agents, there is a probability of transmission every time step. To calculate that probability, Starsim scales the base infectiousness of the disease by:
    * The source’s relative transmissability (rel_sus) <-- will look at rel_sus and rel_trans in next notebook
    * The target’s relative susceptibility (rel_trans)
    * The effective transmission strength of that edge (i.e., edge_weight)

    To decide if a target agent gets infected, Starsim draws a uniform random number between 0 and 1 for that contact. If the probability is larger than the random draw, that contact is marked as having caused an infection. So, for each time step we have a list of infectious sources and a list of targets who got infected.

    Note: By default, Starsim does not clamp the probability of transmission to be less than one, but the random draw is betwen 0 and 1. This means that a transmission will definitely happen if the product of beta, rel_sus, rel_trans, and edge_weight is over 1. This is more likely with a disease with very high base infectiousness.

    When an agent enters a state (e.g., I), Starsim stores the future time that agent will transition out of that state (e.g., from the parameter for infection duration). When that stored time is reached, the agent transitions to the next state (e.g., R).

    Now, load the people and define the networks.
    """)
    return


@app.cell
def _(geopops, ss):
    # Load the Starsim people object
    ppl = ss.load('data/pop_export/starsim/ppl.pkl')

    # Define the networks. This also rewrites the edge lists in pop_export/starsim/net_*.csv
    h = geopops.ForStarsim.GPNetwork(name='homenet', edge_weight=1.0)
    s = geopops.ForStarsim.GPNetwork(name='schoolnet', edge_weight=1.0)
    w = geopops.ForStarsim.GPNetwork(name='worknet', edge_weight=1.0)
    g = geopops.ForStarsim.GPNetwork(name='gqnet', edge_weight=1.0)
    return g, h, ppl, s, w


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The next cell modifies Starsim's pre-built SIR class (defined in the Starsim package file `diseases.py`) so that we can track new infections by network. This will be passed into the sim later.
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **What other kinds of interactions would be important to include in a measles simulation? How could we turn them into networks?**
    """)
    return


if __name__ == "__main__":
    app.run()
