# Introduction
This repository contains the code for my master thesis: 
**Connectome analysis of the Drosophila direction-selective cells.**
I investigate the anatomical basis of the 6 types of optic flow encoded by the fly based on Henning et al., 2022.
Here, I present the code for all Figures that are presented in my thesis: 
The code has been based on analysis already made by @sebasto7

# Content
- data folder
- script folder
- figure folder

# connectome-analysis
Retrieval and analysis of connectomics data from the Flywire dataset.


USER PRE-REQUISITES:
--------------------

1. To have a google (e.g., gmail) account

COMPUTER INSTALLATIONS:
----------------------
ENVIRONMENT MANAGER
- Download and install anaconda (or miniconda) from: https://docs.anaconda.com/anaconda/install/

GIT
- Download and install git from: https://git-scm.com/


ENVIRONMENT REQUISITES:
----------------------

INITIALIZE ENVIRONMENT
The anaconda prompt or the git bash terminals can be used.
In the git bash though, anaconda environments need to be set as source, running:

source C:/Users/[USERNAME]/anaconda3/Scripts/activate

In any of those terminals, follow the commands:

- conda create --name <env_name> python=3.9
- activate <env_name> (-- FOR ACCESSIND DATA VIA NEUPRINT --)

INSTALL PACKAGES FOR DEALING WITH EM DATA SETS
- pip install neuprint-python
- pip install navis
- pip install fafbseg

CHECK IF THE FOLLOWING PACKAGES ARE INSTALLED, IF NOT, INTALL THEM
- pip install numpy
- pip install pandas
- pip install matplotlib
- pip install seaborn
- (other packages might be needed)

OPTIONAL RECOMENDED PACKAGES
- pip install notebook (for Jupyter Notebooks)

IMPORTANT CONSIDERATIONS
- Since NeuPrint is working with a google account (Gmail or similar). If you are
any jupyter notebook in the browser, it is better to use google chrome.
For that, google chrome needs to be the default browser option in the computer.

OPEN A JUPYTER NOTEBOOK
- Typer "jupyter notebook" in the <env_name> environment prompt
- Look for the notebook file (.ipynb) and open it

OPTIONALLY, OTHER TEXT EDITORS CAN BE USED THAT HANDLE COMPLEX FILE TYPES:
- Atom: https://atom.io/
- VSCode: https://code.visualstudio.com/

OR SIMPLER TEXT EDITORS, SUCH US:
- Vim (no installation needed)
- Nano (no installation needed)
