# Welcome to Greeneye

Greeneye is a satellite images classifier aiming to tackle deforestation in the Amazon bassin. This app is based on the Kaggle dataset "[Understanding the Amazon from Space](https://www.kaggle.com/c/planet-understanding-the-amazon-from-space)" 
and has initially been developed as a student project at the end of [Le Wagon](https://www.lewagon.com/)'s bootcamp.

Credits to Tomás Di Gennaro, Vinicius Moura and Guillaume Vanderwinden who built it from scratch in 8 days (December 2020).

![](https://github.com/GuiVdwinden/greeneye/readme_banner.png?raw=true)

The demo, hosted on Heroku, can be found here.

# Application purpose

In a context of **increasing deforestation** in the Amazon bassin, the objective of Greeneye is to label satellite images with their respective category for land cover, land use and atmospheric condition (plain forest, agriculture, pasture, artisanal mining...). Applied on the same region for a certain amount of time, this tool will enable governmental and non-governmental actors to monitor any change in the landscape and react to illegal actions as well as prevent damages from worrisome patterns.


# Data

The dataset is composed of 40,000 different labelled (17 labels) pictures (jpeg - 256x256). Detailed infomation can be found [here](https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/data).

# Models

This folder contains the different models that have been trained along the project. The initial model used for the app is based on a ResNet50.

# Executive summary

*Describe the methodology used (preprocessing to performance metrics)
