# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "geopandas==1.1.3",
#     "ipython==9.11.0",
#     "marimo>=0.21.1",
#     "matplotlib==3.10.8",
#     "numpy==2.4.3",
#     "pandas==3.0.1",
#     "scipy==1.17.1",
#     "seaborn==0.13.2",
#     "shapely==2.1.2",
# ]
# ///

import marimo

__generated_with = "0.21.1"
app = marimo.App(
    css_file="/usr/local/_marimo/custom.css",
    auto_download=["html"],
)


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
    Here's a refresher of how Starsim transmission works from previous notebooks:

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

    [Hopkins Medicine](https://www.hopkinsmedicine.org/health/conditions-and-diseases/measles-what-you-should-know#:~:text=Nine%20out%20of%2010%20unimmunized,room%20if%20they%20are%20unimmunized.) reports 9 out of 10 unimmunized children will contract the virus if they come into contact with an infected individual. So we set the base infectiousness, beta, to be 0.9. The probability of infection for an exposed *unimunized* 5 year old in our model is therefore 0.9. And the probability of infection for an exposed *imunized* 5 year old is 0.027.
    * Unimunized: beta * rel_sus_school * rel_trans_school * edge_weight equals 0 (0.9 * 1 * 1 * 1 = 0.9)
    * Imunized: beta * rel_sus_school * (1 - vax_eff) * rel_trans_school * edge_weight equals 0.027 (0.9 * 1 * (1 - 0.97) * 1 * 1 = 0.027)

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

    After running the cell below, try changing the parameter values and re-running to see how they change infection curves.

    Note: In the plots, "Infectious" refers to the number of agents currently in the infectious state (I), and "cumulative infections" means the cumulative sum of new infections.
    """)
    return


@app.cell
def _(Measles, geopops, plot_measles, sc, ss):
    # Load the people object
    ppl = ss.load("data/pop_export/starsim/ppl_vax.pkl")

    # Define the networks with edge weights
    # Assuming same relative edge weights but can change if you want to
    h = geopops.ForStarsim.GPNetwork(name="homenet",
                                     edge_weight=1.0) # default=1.0
    s = geopops.ForStarsim.GPNetwork(name="schoolnet",
                                     edge_weight=1.0) # default=1.0

    # Define disease parameters
    measles_pars = sc.objdict(
        beta=0.9, # disease's base infectiousness 
        dur_exp=ss.normal(10.0), # duration exposed, not contatious, default=10
        dur_inf=ss.normal(8.0), # duration infectious, contagious, default=8
        vax_eff=0.97, # vaccine efficacy,
                      # scales beta by 1-vax_eff, default=0.97
        rel_sus_babies=1.0, # default=1.0
        rel_sus_young=1.0, # default=1.0
        rel_sus_school=1.0, # default=1.0
        rel_sus_adults=0.0, # default=0.0
        rel_trans_babies=1.0, # default=1.0
        rel_trans_young=1.0, # default=1.0
        rel_trans_school=1.0, # default=1.0
        rel_trans_adults=0.0, # default=0.0
        )

    # Run sim
    sim1 = ss.Sim(
        pars=sc.objdict(start=0, stop=300, dt=1.0),
        people=ppl,
        networks=[h, s],
        diseases=[Measles(measles_pars)],
        ).run()

    # Store results
    res1 = sim1.results  

    # Plot results
    plot_measles(sim1, res1)  
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
    To target a certain school in our seeding function we need its school code, which corresponds to the `sch_code` variable in our people object. We can use the file `EDGE_GEOCODE_PUBLICSCH_1920.xlsx` that was downloaded with `geopops.DownloadData()` in `1_run_geopops.ipynb`. This file comes from NCES data and has every public school with its name and NCES school code. The variable `NCESSCH` in this file corresponds to the variable `sch_code` in our Starsim people object and in `pop_export/sch_students.csv`, a GeoPops file listing students by school.
    """)
    return


@app.cell
def _(pd):
    # Read in list of public schools (big file, takes a while)
    public_schools = pd.read_excel('data/school/EDGE_GEOCODE_PUBLICSCH_1920.xlsx') 

    # Filter for Spartanburg County
    public_schools = public_schools.loc[(public_schools['CNTY'] == 45083)] 

    # Print out school names
    public_schools['NAME'].unique() 
    return (public_schools,)


@app.cell
def _(public_schools):
    # Now filter for Fairforest Elementary
    # Get its NCESSCH number which corresponds to sch_code
    school_name = 'Boiling Springs High'
    school_info = public_schools.loc[public_schools['NAME'] == school_name]

    # Print info for selected school
    school_info 
    return school_info, school_name


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The `sch_code` for Fairforest Elementary is 450363001025. Now let's figure out what CBG that school is located in using the shapefile `tl_2019_45_bg.shp`, which was already downloaded into the geo folder when you made the GeoPops population in `1_run_geopops.ipynb`. The CBG code is used for plotting later. To find it, we can cross reference the LAT and LON in the `school_info` dataframe with the `geography` column in the shapefile. Copy and paste the school code under `NCESSCH` from the dataframe in the previous cell into `sch_code` below. If using Marimo, you need to delete the commas.
    """)
    return


@app.cell
def _(Point, gpd, school_info, school_name):
    sch_code = 450351001000

    # This file is already downloaded into geo folder
    geo_shp = gpd.read_file('data/geo/tl_2019_45_bg.shp') # This file is already downloaded into geo folder
    geo_shp['cbg_geocode'] = geo_shp['GEOID'].astype(int)

    sch_geo = (
        gpd.sjoin(
            gpd.GeoDataFrame(
                school_info.loc[school_info["NCESSCH"] == sch_code],
                geometry=school_info.loc[school_info["NCESSCH"] == sch_code]
                        .apply(lambda r: Point(r["LON"], r["LAT"]), axis=1),
                crs="EPSG:4326",
            ).to_crs(geo_shp.crs),
            geo_shp[["cbg_geocode", "geometry"]],
            how="left",
            predicate="within",
        )["cbg_geocode"].iloc[0]
    )

    print(f'School code of {school_name}:', sch_code)
    print(f'CBG of {school_name}:', sch_geo)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4.2.2 Custom seeding function
    Now we know that the sch_code for Fairforest Elementary is 450363001025 and its CBG is 450830228022. The `custom_seeding()` function is stored in `measles_geopops.py`. Here's the logic: identify agents under `max_age` at school `sch_code` and assign `n_seeds` of them to start the simulation exposed. We limit the age to avoid seeding teachers with infections because in our toy example, adults can't get infected.

    The cell below shows how you can adjust the function to seed infections to different schools. It starts with the default values: max_age=18, sch_codes=[450363001025], and n_seeds=[30]. Try adjusting `n_seeds` below and compare results to your previous plots. Then, go back to section 4.2.1 and try targeting a different school.
    """)
    return


@app.cell
def _(Measles, custom_seeding, h, plot_measles, ppl, s, sc, school_name, ss):
    _sch_code = 450348000060, # default=450363001025

    _seeding_fn = custom_seeding(max_age=18, # default=18
                                sch_codes=[_sch_code], # defined above
                                n_seeds=30) # default=30

    _measles_pars = sc.objdict(init_prev=ss.bernoulli(p=_seeding_fn))

    sim2 = ss.Sim(
        pars=sc.objdict(start=0, stop=300, dt=1.0),
        people=ppl,
        networks=[h, s],
        diseases=[Measles(_measles_pars)],
        ).run()

    res2 = sim2.results

    plot_measles(sim2, res2, label=f"Seeded infections at {school_name}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4.3 Spatial Spread
    We also want to see how seeding infections at different schools impacts spatial spread of the disease. Because we know where each agent lives, we can track infections by geography (down to CBG). The GeoPops class `SubgroupTracking()` allows you to track infections by any of your agent attributes (`agegroup`, `commuter_income_category`, `race_ethnicity`) as long as it is a state in the People object. `SubgroupTracking()` takes the input arguments `subgroup`, `outcome`, and `name`. The `outcome` should be one of the disease compartments, so for measles, it could be `susceptible`, `exposed`, `infected`, or `recovered`. Then you can pass it into a sim as an ss.Analyzer() using the `name`.

    In the next cell we combine the custom seeding function and a `geo_tracking` analyzer into a sim and run it. Then there is a bunch of code to plot an animation of infected by CBG over time. Run the cell making sure that `_sch_geo` (the school's CBG) corresponds to the `sch_code` for your target school. Check out the animation, then try targeting a different school. You can select a school in 4.2.1 or copy and paste from one of these combos.

    | school_name | sch_code | sch_geo |
    | -------- | -------- | -------- |
    | Fairforest Elementary | 450363001025 | 450830228022 |
    | Landrum Middle | 450348000060 | 450830226002 |
    | Boiling Springs High | 450351001000 | 450830224051 |
    """)
    return


@app.cell
def _(
    FuncAnimation,
    Image,
    Measles,
    custom_seeding,
    geopops,
    gpd,
    h,
    plot_measles,
    plt,
    ppl,
    s,
    sc,
    ss,
):
    # Make sure these correspond to the same school!
    _school_name = "Landrum Middle"
    _sch_code = 450348000060 
    _sch_geo = 450830226002 

    # Define subgroup tracking Analyzer
    geo_tracking = geopops.ForStarsim.SubgroupTracking(
        subgroup="cbg_geocode", 
        outcome="infected", 
        name="geo_tracking"
        )

    # Define seeding function parameters
    _seeding_fn = custom_seeding(max_age=18, # default=18
                                 sch_codes=[_sch_code], # defined above
                                 n_seeds=30) # default=30

    # Add seeding function to measles parameters
    _measles_pars = sc.objdict(init_prev=ss.bernoulli(p=_seeding_fn))

    # Run sim
    sim3 = ss.Sim(
        pars=sc.objdict(start=0, stop=300, dt=1.0),
        people=ppl,
        networks=[h, s],
        diseases=[Measles(_measles_pars)],
        analyzers=[geo_tracking],
        ).run()

    # Store results
    res3 = sim3.results

    # Plot results
    plot_measles(sim3, res3)

    ### Animation plotting ###

    # Get population per CBG
    # if you want to compute per pop infection counts
    ppl_df = sim3.people.to_df()
    ppl_df = ppl_df.groupby("cbg_geocode").size().reset_index(name="pop")
    ppl_df = ppl_df.loc[
        ppl_df["cbg_geocode"] != 0
    ]  # drop dummy agents who live outside the geo area

    # Get infected by CBG with analyzer
    geo_df = sim3.analyzers[0].get_subgroup_data()
    geo_df = geo_df.loc[
        geo_df["cbg_geocode"] != 0
    ].copy()  # drop dummy agents who live outside the geo area

    # Merge infected and pop by CBG 
    geo_df = geo_df.merge(ppl_df, on="cbg_geocode", how="left").set_index("cbg_geocode")

    # Merge with shapefil
    _geo_shp = gpd.read_file("data/geo/tl_2019_45_bg.shp")
    _geo_shp["cbg_geocode"] = _geo_shp["GEOID"].astype(int)
    _geo_shp = _geo_shp.merge(geo_df, on="cbg_geocode", how="left")

    # Get geometry for the seeded school CBG
    target_geom = _geo_shp.loc[
        _geo_shp["cbg_geocode"] == _sch_geo, "geometry"
    ].iloc[0]

    cx, cy = target_geom.centroid.x, target_geom.centroid.y

    # Columns to animate
    cols = list(geo_df.columns[:-1])  # exclude 'pop'
    frame_cols = cols[::2]  # every other frame to keep the GIF smaller

    # Fixed color scale across frames
    vmin = _geo_shp[frame_cols].min().min()
    vmax = _geo_shp[frame_cols].max().max()

    fig, ax = plt.subplots(figsize=(7, 4))

    # Draw first frame once so the colorbar is created
    _geo_shp.plot(
        column=frame_cols[0],
        cmap="Blues",
        legend=True,
        ax=ax,
        edgecolor="black",
        linewidth=0.5,
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_title(f"Infected by CBG\n{frame_cols[0]}", fontsize=16, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.annotate(
        text=f"CBG of {_school_name}",
        xy=(cx, cy),
        xytext=(-0.12, 0.5),
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", color="black", linewidth=1.5),
        fontsize=10,
        ha="right",
        va="center",
        annotation_clip=False,
    )

    # Label the colorbar
    cbar = fig.get_axes()[1]
    cbar.set_label("Infected")

    def update(frame):
        ax.clear()
        col = frame_cols[frame]

        _geo_shp.plot(
            column=col,
            cmap="Blues",
            legend=False,
            ax=ax,
            edgecolor="black",
            linewidth=0.5,
            vmin=vmin,
            vmax=vmax,
        )

        ax.set_title(f"Infected by CBG\n{col}", fontsize=16, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.annotate(
            text=f"CBG of {_school_name}",
            xy=(cx, cy),
            xytext=(-0.12, 0.5),
            textcoords="axes fraction",
            arrowprops=dict(arrowstyle="->", color="black", linewidth=1.5),
            fontsize=10,
            ha="right",
            va="center",
            annotation_clip=False,
        )

    plt.tight_layout()

    ani = FuncAnimation(
        fig,
        update,
        frames=len(frame_cols),
        interval=200,
        repeat=True,
    )

    ani.save("map_animation.gif", writer="pillow", fps=4)
    plt.close(fig)

    Image("map_animation.gif", width=700)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
