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
    import numpy as np
    import seaborn as sns
    import scipy.stats as stats
    import starsim as ss
    import sciris as sc
    import geopandas as gpd
    from geopandas import GeoDataFrame
    from shapely.geometry import Point
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from IPython.display import display, Image
    from measles_geopops import Measles, custom_seeding, plot_measles

    return (
        FuncAnimation,
        Image,
        Measles,
        Point,
        custom_seeding,
        display,
        geopops,
        gpd,
        pd,
        plot_measles,
        plt,
        sc,
        ss,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 4.0 Measles and seeding infections
    In this notebook, we'll explore the custom measles class and the impact of seeding infections to a selected school.

    ## 4.1 Custom Measles class
    Here's a refresher on how transmission works from previous notebooks:

    For each time step, Starsim knows who the infectious agents are (sources) and who the susceptible agents are (targets). For every contact (i.e., edge) between agents, there is a probability of infection every time step if one of those agents is infectious. To calculate that probability, Starsim scales the base infectiousness of the disease by:
    * The source’s relative transmissability
    * The target’s relative susceptibility
    * The effective transmission strength of that contact (edge_weight)

    To decide if a target agent gets infected, Starsim draws a uniform random number between 0 and 1 for that contact.
    If the probability is larger than the random draw, that contact is marked as having caused an infection. So, for each time step we have a list of infectious sources and a list of targets who got infected.

    When an agent enters a disease state (e.g., Infectious), Starsim samples and stores the future time that agent will transmission out of that state (e.g., using the duration of infection). When that stored time is reached, the agent transitions to the next state (e.g., Recovered).

    The custom measles class is called `Measles()` and is stored in `mealses_geopops.py` builds on Starsim's base SIR class stored in the Starsim file `diseases.py`. You can customize diseases in many ways (adding disease compartments or subgroup-specific parameters). Our custom measles class includes:
    * A state for Exposed individuals
    * Age-dependent relative transmissability and relative susceptability
    * Reduced susceptibility for vaccinated individuals
    * A custom function to seed infections by age and school

    Here are the main disease parameters used in `Measles()`:

    | Parameter | Definition | Default value |
    | -------- | -------- | -------- |
    | beta | Disease's base infectiousness, scaled by rel_trans, rel_sus, and edge_weight to determine probability of infection | 0.9 |
    | dur_exp | Duration of Exposed state (not contagious), drawn from normal distribution | 10.0 |
    | dur_inf | Duration of Infecitous state (contagious), drawn from normal distribution | 8.0 |
    | vax_eff | Effectiveness of vaccine at reducing susceptibility, scales rel_sus by 1-vax_eff | 0.97 |
    | rel_sus_babies | Relative susceptability of ages <2 | 1.0 |
    | rel_sus_young | Relative susceptability of ages 2-4 | 1.0 |
    | rel_sus_school | Relative susceptability of ages 5-18 | 1.0 |
    | rel_sus_adults | Relative susceptability of ages >18| 0.0 |
    | rel_trans_babies | Relative transmissibility of ages <2 | 1.0 |
    | rel_trans_young | Relative transmissibility of ages 2-4 | 1.0 |
    | rel_trans_school | Relative transmissibility of ages 5-18 | 1.0 |
    | rel_trans_adults | Relative transmissibility of ages >18| 0.0 |

    [Hopkins Medicine](https://www.hopkinsmedicine.org/health/conditions-and-diseases/measles-what-you-should-know#:~:text=Nine%20out%20of%2010%20unimmunized,room%20if%20they%20are%20unimmunized.) reports 9 out of 10 unimmunized children will contract the virus if they come into contact with an infected individual. So we set the base infectiousness, beta, to be 0.9.

    The [CDC](https://www.cdc.gov/measles/hcp/communication-resources/clinical-diagnosis-fact-sheet.html#:~:text=Measles%20is%20a%20highly%20contagious%20respiratory%20virus,more%20than%20104%C2%B0%20F%20when%20rash%20appears) describes the following progression of illness, which informs dur_exp and dur_inf in the model.

    | Measles progression | Days | Model compartment |
    | -------- | -------- | -------- |
    | Incubation, no symptoms | 1-10 | Exposed |
    | Prodromal symptoms, cough/fever | 11-14 | Infectious |
    | Rash | 15-18 | Infectious |
    | Rash clears |19-21 | Recovered |

    To simplify the Spartanburg example, we also assume
    * Only children under 18 can get infected, so we set rel_sus_adults=0 and rel_trans_adults=0
    * Infections only occur in the home and at school so we don't need to include the work and group quarters networks in the sim
    * No one dies from measles, probability of death is 0

    Initial prevalence is defined with a custom seeding function stored in `measles_geopops.py`. We'll look at adjusting this in following sections.

    First, try changing the parameter values in the cell below to see how they change infection curves.

    In the plots, "Infectious" refers to the number of agents currently in the infectious state (I), and "cumulative infections" means the cumulative sum of new infections.
    """)
    return


@app.cell
def _(Measles, geopops, plot_measles, sc, ss):
    # Load the people object
    ppl = ss.load('data/pop_export/starsim/ppl_vax.pkl')
    h = geopops.ForStarsim.GPNetwork(name='homenet', edge_weight=1.0)
    # Define the networks with edge weights. 
    # Assuming same relative edge weights but can change if you want to
    s = geopops.ForStarsim.GPNetwork(name='schoolnet', edge_weight=1.0)
    _measles_pars = sc.objdict(beta=0.6, dur_exp=ss.normal(10.0), dur_inf=ss.normal(8.0), p_death=ss.bernoulli(p=0.0), vax_eff=0.97, rel_sus_babies=1.0, rel_sus_young=1.0, rel_sus_school=1.0, rel_sus_adults=0.0, rel_trans_babies=1.0, rel_trans_young=1.0, rel_trans_school=1.0, rel_trans_adults=0.0)
    sim1 = ss.Sim(pars=sc.objdict(start=0, stop=300, dt=1.0), people=ppl, networks=[h, s], diseases=[Measles(_measles_pars)]).run()
    res1 = sim1.results  # disease's base infectiousness (scaled by rel_trans, rel_sus, and edge_weight)
    # Run sim with no quarantine
    # Store results
    # Plot results
    plot_measles(sim1, res1)  # duration of exposed state, in this model not infectious, drawn from normal distribution  # duration of infectious state, drawn from normal distribution  # probability of death, drawn from Bernoulli distribution  # effectiveness of vaccine, scales rel_sus by 1-vax_eff  # relative susceptability of ages <2  # relative susceptability of ages 2-4  # relative susceptability of ages 5-17  # relative susceptability of ages >18  # relative transmissability of ages <2  # relative transmissability of ages 2-4  # relative transmissability of ages 5-17  # relative transmissability of ages >18  # use custom Measles model
    return h, ppl, res1, s, sim1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The plot above is created with the `plot_measles()` function, which is defined in `measles_geopops.py`. You can also use Starsim's plotting function or plot one outcome at a time.
    """)
    return


@app.cell
def _(plt, res1, sim1):
    # Starsim's plotting function
    sim1.plot()

    # Print available outcomes to plot with this disease
    # 'n_susceptible', 'n_infected', 'n_recovered', 'n_exposed', 'prevalence', 'new_infections', 'cum_infections', 'new_infections_by_network'
    print(res1['measles'].keys()) 

    # Plot one outcome at a time
    plt.plot(res1['measles'].prevalence)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4.2 Seed infections
    The first measles infections in Spartanburg were [reported at two schools](https://dph.sc.gov/news/measles-update-dph-confirms-locations-spartanburg-county-outbreak-media-briefing-take-place): Fairforest Elementary (public) and Global Academy of South Carolina (private). GeoPops does not currently include private schools or daycares but will in the future. In this section, we find the school code and CBG of Fairforest Elementary. Then we seed infections to agents attending this chool, run a sim, and plot infected by CBG over time.

    ### 4.2.1 Find target school
    To target a certain school in our seeding function we need its school code, which corresponds to the `sch_code` variable in our people object. We can use a file that was downloaded during `geopops.DownloadData()` in `1_run_geopops.ipynb`. `EDGE_GEOCODE_PUBLICSCH_1920.xlsx` comes from NCES data and has every public school with its name and NCES school code. The variable NCESSCH in this file corresponds to the variable sch_code in our Starsim people object and in `pop_export/sch_students.csv`, a GeoPops file listing students by school.
    """)
    return


@app.cell
def _(pd):
    # Read in list of public schools and filter for Spartanburg County 
    public_schools = pd.read_excel('data/school/EDGE_GEOCODE_PUBLICSCH_1920.xlsx') # big file, takes a while
    public_schools = public_schools.loc[(public_schools['CNTY'] == 45083)] # filter for Spartanburg County
    public_schools['NAME'].unique() # Print out school names
    return (public_schools,)


@app.cell
def _(public_schools):
    # Now filter EDGE_GEOCODE_PUBLICSCH_1920.xlsx for Fairforest Elementary and get it's NCESSCH number which corresponds to sch_code
    school_name = 'Fairforest Elementary'
    school_info = public_schools.loc[public_schools['NAME'] == school_name]
    school_info # Print info for selected school
    return school_info, school_name


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now let's figure out what CBG that school is located in using a shapefile already downloaded when you made the GeoPops population. We use this for plotting later. We can cross reference the LAT and LON in the school_info dataframe with the geography in the SC CBG shapefile that was downloaded in `1_run_geopops.ipynb`. Copy and paste the code under `NCESSCH` above into `sch_code` below.
    """)
    return


@app.cell
def _(Point, gpd, school_info):
    _geo_shp = gpd.read_file('data/geo/tl_2019_45_bg.shp')  # This file is already downloaded into geo folder
    _geo_shp['cbg_geocode'] = _geo_shp['GEOID'].astype(int)
    sch_code = 450363001025
    sch_geo = gpd.sjoin(gpd.GeoDataFrame(school_info.loc[school_info['NCESSCH'] == sch_code], geometry=school_info.loc[school_info['NCESSCH'] == sch_code].apply(lambda r: Point(r['LON'], r['LAT']), axis=1), crs='EPSG:4326').to_crs(_geo_shp.crs), _geo_shp[['cbg_geocode', 'geometry']], how='left', predicate='within')['cbg_geocode'].iloc[0]
    print('CBG of selected school:', sch_geo)
    return (sch_geo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4.2.2 Custom seeding function
    Now we know that the sch_code for Fairforest Elementary is 450363001025. The `custom_seeding()` function is stored in `measles_geopops.py`. Here's the logic: identify agents under `max_age` at school `sch_code` and assign `n_seeds` of them to start the simulation exposed. We limit the age to avoid seeding teachers with infections. The cell below shows how you can adjust the function to seed infections to different schools. It starts with the default values for max_age, sch_codes, and n_seeds. Trying adjusting these and see how it changes the results. You can go back to section 4.2.1 to target a different school.
    """)
    return


@app.cell
def _(Measles, custom_seeding, h, plot_measles, ppl, s, sc, school_name, ss):
    sch_code_1 = 450363001025
    _seeding_fn = custom_seeding(max_age=18, sch_codes=[sch_code_1], n_seeds=30)
    _measles_pars = sc.objdict(init_prev=ss.bernoulli(p=_seeding_fn))
    sim2 = ss.Sim(pars=sc.objdict(start=0, stop=300, dt=1.0), people=ppl, networks=[h, s], diseases=[Measles(_measles_pars)]).run()
    res2 = sim2.results
    plot_measles(sim2, res2, label=f'Seeded infections in {school_name}')
    return (sch_code_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4.3 Spatial Spread
    We also want to see how seeding infections at different schools impacts spatial spread of the disease. Because we know where each agent lives, we can track infections by geography (down to CBG). The GeoPops class `SubgroupTracking()` allows you to track infections by any of your agent attributes as long as they are a state in the people object. It takes the input arguments `subgroup`, `outcome`, and `name`. The outcome should be one of the disease states, so for measles, it could be susceptible, exposed, infected, or recovered. Then you can pass it into a sim as an ss.Analyzer() using its name.

    In the next cell we combine the custom seeding function and the geo_tracking analyzer into a sim and run it. Then in the subsequent cell, we plot infected by CBG over time. After running both cells once, try picking a different school in section 4.2.1. Then try running the cells below again and to how seeding a different school changes spatial spread.
    """)
    return


@app.cell
def _(
    Measles,
    custom_seeding,
    geopops,
    h,
    plot_measles,
    ppl,
    s,
    sc,
    sch_code_1,
    ss,
):
    geo_tracking = geopops.ForStarsim.SubgroupTracking(subgroup='cbg_geocode', outcome='infected', name='geo_tracking')
    _seeding_fn = custom_seeding(max_age=18, sch_codes=[sch_code_1], n_seeds=30)
    _measles_pars = sc.objdict(init_prev=ss.bernoulli(p=_seeding_fn))
    sim3 = ss.Sim(pars=sc.objdict(start=0, stop=300, dt=1.0), people=ppl, networks=[h, s], diseases=[Measles(_measles_pars)], analyzers=[geo_tracking]).run()
    res3 = sim3.results
    plot_measles(sim3, res3)
    return (sim3,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Run the next cell to plot spatial spread.
    """)
    return


@app.cell
def _(FuncAnimation, Image, display, gpd, plt, sch_geo, sim3):
    # Get population per CBG
    ppl_df = sim3.people.to_df()
    ppl_df = ppl_df.groupby('cbg_geocode').size().reset_index()
    ppl_df = ppl_df.loc[ppl_df['cbg_geocode'] != 0]  # drop dummy agents who live outside the geo area and have cbg_geocode = 0
    ppl_df = ppl_df.rename(columns={0: 'pop'})
    geo_df = sim3.analyzers[0].get_subgroup_data()
    # Get infected by tract with analyzer
    geo_df = geo_df.loc[geo_df['cbg_geocode'] != 0].set_index('cbg_geocode')  # Returns second analyzer (cbg_tracking) as dataframe
    geo_df = geo_df.merge(ppl_df, on='cbg_geocode', how='left').set_index('cbg_geocode')  # drop dummy agents who live outside the geo area and have cbg_geocode = 0
    _geo_shp = gpd.read_file('data/geo/tl_2019_45_bg.shp')
    _geo_shp['cbg_geocode'] = _geo_shp['GEOID'].astype(int)
    # Calculate infected per 1000 population
    # for col in geo_df.columns[:-1]:
    #     geo_df[col] = geo_df[col] / geo_df['pop'] * 1000
    # geo_df.drop(columns=['pop'], inplace=True)
    _geo_shp = _geo_shp.merge(geo_df, on='cbg_geocode', how='left')
    # Get shapefile for CBGs in SC
    # This file is already downloaded in `1_run_geopops.ipynb` into geo folder
    target_geom = _geo_shp.loc[_geo_shp['cbg_geocode'] == sch_geo, 'geometry'].iloc[0]  # This file is already downloaded in `1_run_geopops.ipynb` into geo folder
    cx, cy = (target_geom.centroid.x, target_geom.centroid.y)
    cols = list(geo_df.columns[:-1])
    # Merge with analyzer results
    vmin = _geo_shp[cols].min().min()
    vmax = _geo_shp[cols].max().max()
    # Get centroid of target tract
    fig, ax = plt.subplots(figsize=(10, 6))

    def update(frame):
    # Columns you want to animate
        ax.clear()
        col = cols[frame]
    # Keep color scale fixed across frames so colors are comparable
        _geo_shp.plot(column=col, cmap='Blues', legend=False, ax=ax, edgecolor='black', linewidth=0.5, vmin=vmin, vmax=vmax)
        ax.set_title(f'Infected by CBG\n{col}', fontsize=20, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.annotate(text='CBG of seeded school', xy=(cx, cy), xytext=(-0.12, 0.5), textcoords='axes fraction', arrowprops=dict(arrowstyle='->', color='black', linewidth=1.5), fontsize=10, ha='right', va='center', annotation_clip=False)
    _geo_shp.plot(column=cols[0], cmap='Blues', legend=True, ax=ax, edgecolor='black', linewidth=0.5, vmin=vmin, vmax=vmax)
    ax.set_title(f'Infected CBG\n{cols[0]}', fontsize=20, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.annotate(text=str(sch_geo), xy=(cx, cy), xytext=(-0.12, 0.5), textcoords='axes fraction', arrowprops=dict(arrowstyle='->', color='black', linewidth=1.5), fontsize=10, ha='right', va='center', annotation_clip=False)  # Plot current frame
    cbar = fig.get_axes()[1]
    cbar.set_label('Infected')
    plt.tight_layout()
    plt.close(fig)  # avoid stacking a new legend/colorbar every frame
    ani = FuncAnimation(fig, update, frames=len(cols), interval=100, repeat=True)
    ani.save('map_animation.gif', writer='pillow', fps=8)
    # Draw first frame once so we can make one shared colorbar
    # Grab the colorbar axis created by the first plot
    display(Image('map_animation.gif'))  # Add arrow annotation  # outside left side of axes
    return


if __name__ == "__main__":
    app.run()
