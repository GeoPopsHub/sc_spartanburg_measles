# GeoPops Measles Tutorial

This tutorial demonstrates how to simulate a measles outbreak in Spartanburg County, South Carolina, USA using the synthetic population generator, [GeoPops](https://github.com/GeoPopsHub/geopops), and the agent-based modeling software, [Starsim](https://starsim.org/). The tutorial includes five interactive notebooks in which users: 
1. Make a population with GeoPops
2. Compare the population to real Census data
3. Explore home, school, and workplace networks
4. Run an Starsim ABM with infections seeded to a specific school and observe spatial spread, and
5. Test different quarantine strategies and track outcomes by age group. 

Think of this tutorial and measles example as a starting point of an iterative community-building modeling exercise. GeoPops and Starsim provide a customizable framework for a detailed, context-specific scenario model, but this version is the first draft! There is still a lot that can be added in collaboration with public health researchers and officials on the ground to make the population and model more realistic. The [GeoPops_Measles.pdf](https://github.com/GeoPopsHub/sc_spartanburg_measles/blob/main/GeoPops_Measles.pdf) provides an overview of the GeoPops with Starsim framework as well as the measles outbreak in South Carolina.

## GeoPops with Starsim
![GeoPops Starsim](GeoPops_Measles.jpeg)

The **GeoPops with Starsim** framework enables detailed scenario modeling because for every agent, we know:
* Several demographic characteristics (age, gender, race/ethnicity, income, and more)
* Where they live and where they go to school or work
* Who infected them, when, and where (home, school, work, GQ)
* When they transition disease compartments (e.g., S, I, R)

Think of the framework like a puzzle. Each puzzle piece is a different model component (e.g., people, networks, disease). By customizing the logic and parameters of each component, you determine the how the puzzle pieces fit together and what picture they make.

All GeoPops and Starsim code is open source, enabling users to customize populations, diseases, and interventions for their own research questions. Our aim is to build capacity of modelers and increase decision-making relevance with context-specific simulations using open-source, user-friendly tools.

You can download the repo and go through the notebooks in order, or try out the interactive Marimo notebooks online. To use Marimo, simply click on a link below and fork the notebook using a Gmail or GitHub account. Then you can make your own changes without altering the source file. The first notebook explains how GeoPops works and how to make a population. It needs to be run locally, but you can still go through the other notebooks and Marimo links without doing so.  

| Notebook | Description | Marimo |
| -------- | -------- | -------- | 
| 1_run_geopops.ipynb | Make a GeoPops population of Spartanburg, SC | N/A |
| 2_explore_people.ipynb | Explore Starsim People object and compare GeoPops population to real Census data | [![Open in molab](https://molab.marimo.io/molab-shield.svg)](https://molab.marimo.io/notebooks/nb_Z9nu2t6jQpk8kY1FF5YTVw) |
| 3_explore_networks.ipynb | Run a simple SIR model and see what happends when you change network edge weights | [![Open in molab](https://molab.marimo.io/molab-shield.svg)](https://molab.marimo.io/notebooks/nb_jSd7sQou61oQUNSFZNYC1H) |
| 4_measles_seeding.ipynb | Explore the custom measles model, seed infections to a specific school, and observe spatial spread | [![Open in molab](https://molab.marimo.io/molab-shield.svg)](https://molab.marimo.io/notebooks/nb_7bWLdVBk4GGm9sRE9wmS1S) |
| 5_measles_quarantine.ipynb | Eplore the custom measles model and test four quarantine strategies:<br>- Infectious individual only<br>- Infectious individual and siblings<br>- Infectious individual and contacts<br>- Entire school of infectious individual | [![Open in molab](https://molab.marimo.io/molab-shield.svg)](https://molab.marimo.io/notebooks/nb_pMKcSuZQ7nRSVsih3EyB6u)|



