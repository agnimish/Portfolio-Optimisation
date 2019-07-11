
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import sys
from sklearn import linear_model
from sklearn import metrics as sm

##################################################################################
##################################################################################
## Template file for problem 1.                                                 ##
## Step1: Solve the problem in the jupyter notebook worksheet
## Step2: Copy function getSymbolsToTrade() from jupyter notebook worksheet below
## Step3: Copy function getWeights(c,g, Qt) from the worksheet below
## Step4: Copy optional function getPrediction(n) from the worksheet below
## Step5: Copy any other helper functions that you have created in the class CustomFeatures() below
## Step5: Submit the template file back on the platform
## Don't change any other function
##################################################################################

########################################################################################
########################################################################################
#### VERY IMPORTANT: ONLY COPY THE BODY OF THE FUNCTION below                        ###
###  do not change the function signature or inputs else your file will not evaluate ###
########################################################################################
########################################################################################


class MyTradingFunctions():

    def __init__(self):  #Put any global variables here
        self.lookback = 36  ## max number of historical datapoints you want at any given time
        self.params = {}

        # for example you can import and store an ML model from scikit learn in this dict
        self.model = {}

        # and set a frequency at which you want to update the model

        self.updateFrequency = 6
        self.CustomFeatures = CustomFeatures()


    ###########################################
    ## ONLY FILL THE FOUR FUNCTIONS BELOW    ##
    ###########################################

    ###############################################################################
    ### TODO 1: FILL THIS FUNCTION TO the asset group you want to model for     ###
    ### This can be 'G1' or 'G2'                                                ###
    ### VERY IMPORTANT: do not change the function signature or inputs else your file will not evaluate
    ###############################################################################

    def getSymbolsToTrade(self):
        return 'G1'

    ###############################################################################
    ### TODO 2: FILL THE LOGIC IN THIS FUNCTION to generate weights
    ## This function takes in the following inputs:
    ## identifiers: asset identifiers
    ## reward: reward at time t (based on w(t-1))
    ## wi: weights to initialize from, if you want to use
    ## Dt: value of column 'd' per asset
    ## St: value of column 'S' per asset
    ## Qt: value of column 'q' per asset
    ## g: value of constant gamma, read problem descrption for details
    ## U: value of constant U, read problem descrption for details
    ## t: value of constant t, read problem descrption for details
    ## T: value of constant T, read problem descrption for details
    ## P: value of constant P, read problem descrption for details
    ## delta: value of constant delta, read problem descrption for details
    ## chi: value of constant chi, read problem descrption for details
    ## eta: value of constant eta, read problem descrption for details
    ## **kwargs: any additional params you want to add can be specified here. kwargs is a dictionary

    ### VERY IMPORTANT: do not change the function signature or inputs else your file will not evaluate
    ###############################################################################
    ###############################################################################

    def getWeights(self, identifiers, reward, wi, Dt, St, Qt, g, U, t, T, P, delta, chi, eta, df, trr, **kwargs):
        ################################################
        ####   COPY FROM BELOW INTO TEMPLATE FILE   ####
        ################################################
        weights = pd.Series(np.random.random(len(identifiers)), index=identifiers)
        
        weights[Qt==0] = 0
        
        ## to use kwargs
        var = kwargs['var']
        test = kwargs['func_test'](2)
        weights = weights/weights.sum()
        if (weights>g).any():
            weights[weights>g] = g
            weights = weights/weights.sum()
        
        ## You can call new features like below:
        CustomFeaturesCls = CustomFeatures()
        test = CustomFeaturesCls.newFeature1()
        return weights

    ## Step 2b: Fill extra arguments below. See sample below

    def getKwargs(self):   
        return {'var': 2, 'func_test': lambda x: 2*x}

    ###############################################################################
    ### TODO 3: OPTIONAL FILL THE LOGIC IN THIS FUNCTION to generae predictions for returns
    ## This function takes in the following inputs:
    ## identifiers: asset identifiers
    ## reward: reward at time t (based on w(t-1))
    ## wi: weights to initialize from, if you want to use
    ## Dt: value of column 'd' per asset
    ## St: value of column 'S' per asset
    ## Qt: value of column 'q' per asset
    ## g: value of constant gamma, read problem descrption for details
    ## U: value of constant U, read problem descrption for details
    ## t: value of constant t, read problem descrption for details
    ## T: value of constant T, read problem descrption for details
    ## P: value of constant P, read problem descrption for details
    ## delta: value of constant delta, read problem descrption for details
    ## chi: value of constant chi, read problem descrption for details
    ## eta: value of constant eta, read problem descrption for details
    ## **kwargs: any additional params you want to add can be specified here. kwargs is a dictionary

    ### VERY IMPORTANT: do not change the function signature or inputs else your file will not evaluate
    ###############################################################################
    ###############################################################################

    def getPrediction(self,identifiers, wi, Dt, St, Qt, g, U, t, T, P, delta, chi, eta, **kwargs):
        return np.zeros(len(identifiers))


####################################################
##   YOU CAN DEFINE ANY CUSTOM FEATURES HERE      ##
####################################################
class CustomFeatures():

    def newFeature1(self, **kwargs):
        return None
