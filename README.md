# GeoPops Measles Tutorial

In this tutorial, we'll simulale a measles outbreak in Spartanburg County, South Carolina, USA using the synthetic population generator, GeoPops, and the agent-based modeling software, Starsim. 

First, take a look at the [GeoPops_Measles.pdf](https://github.com/GeoPopsHub/sc_spartanburg_measles/blob/main/GeoPops_Measles.pdf) for an overview of the GeoPops with Starsim framework as well as the measles outbreak in South Carolina.

Then, you can download the repo and go through the notebooks in order, or try out the interactive Marimo notebooks online. To use Marimo, you simply fork the notebook using a Gmail or GitHub account, and then you can make your own changes in the notebook without affecting the source files.

| Notebook | Description | Marimo link |
| -------- | -------- | -------- | 
| 1_run_geopops.ipynb | Make a GeoPops population of Spartanburg, SC| [![Open in molab](https://molab.marimo.io/molab-shield.svg)](https://molab.marimo.io/notebooks/nb_Z9nu2t6jQpk8kY1FF5YTVw) |
| 2_explore_people.ipynb | Explore Starsim People object and compare GeoPops population to real Census data| |
| 3_explore_networks.ipynb | Run a simple SIR model and see what happends when you change network edge weights | |
| 4_measles_seeding.ipynb | Seed infections to a specific school and observe spatial spread | |
| 5_run_geopops.ipynb | Test four quarantine strategies:<br>- Infected individual only<br>- Infected individual and siblings<br>- Infected individual and contacts<br>- Entire school | |