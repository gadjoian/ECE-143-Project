# ECE 143 Project Team 24

## Introduction
This repo contains the code used to create the presentation for Team 24's presentation, which focused on the impact COVID-19 had on global air traffic. 

## Repository Structure and Contents
This repository consists of two directories, several notebooks, and our final presentation pdf. The directories are the Data and src directories. The Data directory contains all of the data used for this project, except for the main data set which contained 53 million rows and required upwards of 5 gigabytes in zipped form. Due to this we included a method in CountryAndGlobalAnalysis.ipynb, which is the only notebook that needs the dataset in full, to download it all. 
**Note that this particular notebook will take several hours to reproduce, as there are many operations that are needed to be undergone to fully filter the data and generate the needed graphs. We have included a prerun notebook for convenience.** 
The src directory contains the .py versions of the notebooks. Finally the root directory of the repository contains all of the notebooks and our presentation. These notebooks should be fully contained, and can simply be run. **CountryAndGlobalAnalysis.ipynb will need the download_data() cell uncommented to download the data, this notebook will also take several hours to run**

## Running the Code
This code should all be self contained. The two notebooks should be runnable after pip installing the necessary packages, most likely just plotly. We had prerun both notebooks for your convenience. The CountryAndGlobalAnalysis takes a **very** long time to run due to processing and filtering the 53 million row dataset.

## Third Party Modules
1. Plotly (to generate the nicely formated graphs)
2. CountryInfo (to gather population information for all countries)
