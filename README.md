# Welcome to Greeneye

Greeneye is a satellite images classifier aiming to tackle deforestation in the Amazon basin. 

This app is based on the dataset "[Understanding the Amazon from Space](https://www.kaggle.com/c/planet-understanding-the-amazon-from-space)" from Kaggle
and has originally been developed as a student project at the end of [Le Wagon](https://www.lewagon.com/)'s bootcamp. Credits to Tomás Di Gennaro, Vinicius Moura and Guillaume Vanderwinden who built it from scratch in 8 days (December 2020). The demo, hosted on Heroku, can be found here.

*Caution: this tool is not ready to be scaled yet. Any contributor is welcome to join the adventure and support Greeneye's development for a greener future!*

![alt text](https://github.com/GuiVdwinden/greeneye/blob/master/readme_banner.png?raw=true)

# Application purposes

In a context of **increasing deforestation** in the Amazon basin, the objective of Greeneye is to label satellite images with their respective category for land cover, land use (plain forest, agriculture, pasture, artisanal mining...) and atmospheric condition. 
Applied on the same area for a certain time interval, Greeneye has the potential to help governmental and non-governmental actors monitor any change in the landscape and react to illegal actions as well as prevent damages from worrisome patterns.

# Data

The dataset is composed of 40,000 different labelled (17 labels) satellite pictures (jpeg - 256x256) from the Amazon Forest. "The data comes from Planet's Flock 2 satellites in both sun-synchronous and ISS orbits and was collected between January 1, 2016 and February 1, 2017." Detailed infomation can be found [here](https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/data).

# Models

The first model used by Greeneye is a ResNet50. Further models are being trained, tested and stored in this folder.

# Executive summary

*Describe the methodology used (preprocessing to performance metrics)*
