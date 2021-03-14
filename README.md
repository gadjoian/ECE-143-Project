# ECE 143 Project Team 24

## Introduction
This repo contains the code used to create the presentation for Team 24's presentation, which focused on the impact COVID-19 had on global air traffic. 

## Repository Structure and Contents
This repository consists of two directories, several notebooks, a requirements text file, and our final presentation pdf. The directories are the Data and src directories. The Data directory contains all of the data used for this project, except for the main data set which contained 53 million rows and required upwards of 5 gigabytes in zipped form. Due to this we included a method in CountryAndGlobalAnalysis.ipynb, which is the only notebook that needs the dataset in full, to download it all. <br />

**Note that this particular notebook will take several hours to reproduce, as there are many operations that are needed to be undergone to fully filter the data and generate the needed graphs. We have included a prerun notebook for convenience.** <br />

The src directory contains the .py versions of the notebooks. Finally the root directory of the repository contains all of the notebooks and our presentation. These notebooks should be fully contained, and can simply be run. **CountryAndGlobalAnalysis.ipynb will need the download_data() cell uncommented to download the data, this notebook will take several hours to run.** The dataset required for executing notebooks **Company_Wise_Plots.ipynb** & **Country_Wise_Plots.ipynb** is pre downloaded into the Data directory (these two notebooks execute relatively quickly).

Notebooks:<br />
**1. CountryAndGlobalAnalysis.ipynb : Contains plots related to global flight data analysis**<br />
**2. Country_Wise_Plots.ipynb       : Contains plots related to country wise flight data analysis**<br />
**3. Company_Wise_Plots.ipynb       : Contains plots related to company wise flight data analysis**<br />

The pre-executed pdf version of these notebooks are also included in the root directory for your reference.  

## Running the Code
This code should all be self contained. The three notebooks should be runnable after pip installing the necessary packages, most likely just plotly. We had pre run both notebooks for your convenience. The CountryAndGlobalAnalysis takes a **very** long time to run due to processing and filtering the 53 million row dataset.

## Third Party Modules
1. numpy 1.19.5
2. pandas 1.2.1
3. matplotlib 3.3.4
4. plotly 4.14.3
5. scipy 1.2.2
6. countryinfo 0.1.2

To download all of these dependencies in a convenient way, you can run pip install requirements.txt.



