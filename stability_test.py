#This is basically Jonah Pemstein's code:
#https://github.com/japem/stabilization/blob/master/main_code.py

#Cronbach's alpha to measure reliability of different stats

#Import modules
import numpy as np
import pandas as pd
import math
import random
import timeit
import os
print("Modules all imported")

#Start time
stime = timeit.default_timer() #start time

#Define functions
def time(start, msg="   "):
    curtime = timeit.default_timer()
    tottime = curtime-start
    hours = math.floor(tottime/3600)
    minutes = math.floor(tottime/60)-hours*60
    seconds = tottime % 60
    if seconds < 10:
        seconds = "0"+str(round(seconds,1))
    else:
        seconds = str(round(seconds,1))
    if minutes < 10:
        minutes = "0"+str(minutes)
    if hours < 10:
        hours = "0"+str(hours)
    print(msg, "Time elapsed: "+str(hours)+":"+str(minutes)+":"+str(seconds))

def alpha(prepped):
    #Calculate Cronbach's alpha       
    stdx = np.std(prepped.sum())
    varx = stdx*stdx #The variance of all total scores
    fpv = prepped.transpose()
    stdy = fpv.std()
    vary = stdy*stdy #The variances of every player-year's scores
    k = prepped.shape[0] #Number of "test items", in this case balls in play 
    kterm = k/(k-1)
    sum_vary = np.sum(vary) #The sum of all variances of total scores
    varterm = 1-(sum_vary/varx)
    return(kterm * varterm)

def calculate(statlist, # list; string of each stat
              data, # dataframe
              playeridtype, # string; name of ID column
              yearcolumn, # string; name of year column
              denom_name, # string; denominator name
              yearrange, # list; selection of years
              playertype, # string; name for player type (pitchers/hitters/etc)
              path, # string;' os path to store output
              maxdenom, # integer;  value for largest sample size
              increment, # integer of increment
              extradenom=[] # list; additional sample sizes, in addition to the others run by maxdenom & increment combo
             ):
    #Create dictionary with every increment of denominator desired
    statnum=[]
    for i in range(1, int(maxdenom/increment)):
        statnum.append(i*increment)
    statnum.extend(extradenom)
    statnum.sort()
    stat_dict = {denom_name:statnum}

    alpha_df, mean_df, sd_df, count_df = pd.DataFrame(stat_dict),pd.DataFrame(stat_dict),pd.DataFrame(stat_dict),pd.DataFrame(stat_dict)  #create dataframes with every increment of denominator desired  

    playerlist = list(data[playeridtype].unique()) #make a list of all player IDs

    #Iterate through different statistics
    for stat in statlist:
        alpha_list, mean_list, sd_list, count_list = [],[],[],[] #clear list of alphas, means, standard deviations, and sample sizes
        nums_dict = {} #Create empty dictionary
        for i in playerlist: #Populate dictionary with batter numbers for the given statistic
            for y in yearrange:
                nums = pd.Series.tolist(data[(data[playeridtype] == i) & (data[yearcolumn] == y)][stat])
                nums_dict[str(i)+str(y)] = nums

      #Iterate through different numbers of events, each time creating a dataframe that alpha can be calculated from
        for samplesize in stat_dict[denom_name]:
            #Create empty dataframe
            x = {}
            prepped = pd.DataFrame(x)
            #Fill that dataframe with a random sample of events
            for i in nums_dict:
                if len(nums_dict[i]) >= samplesize:
                    prepped[str(i)] = random.sample(nums_dict[i], samplesize) #Add the random sample to the prepped dataframe that will be used to calculate alpha 
         
            if prepped.shape[1] >= 5: #If there are at least five players with enough events, add alpha to the list of alphas for that stat (and mean, standard deviation, and count)
                a = alpha(prepped)
                alpha_list.append(a)
                m = np.mean(prepped.mean())
                mean_list.append(m)
                s = np.std(prepped.mean())
                sd_list.append(s)
                n = prepped.shape[1]
                count_list.append(n)
            else: #There aren't enough batters to calculate alpha
                break #stop calculating alpha for this stat and move on to the next stat
            
        #Add that list of alphas for that stat to the dataframe containing alpha for all stats
        alpha_df, mean_df, sd_df, count_df = alpha_df.loc[:len(alpha_list)-1], mean_df.loc[:len(mean_list)-1], sd_df.loc[:len(sd_list)-1], count_df.loc[:len(count_list)-1]
        alpha_df[stat], mean_df[stat], sd_df[stat], count_df[stat] = (alpha_list, mean_list, sd_list, count_list)
        time(stime,msg="Completed "+stat+" for "+path+".")
    alpha_df.to_csv(path+"_alpha.csv",index=False)
    mean_df.to_csv(path+"_mean.csv",index=False)
    sd_df.to_csv(path+"_sd.csv",index=False)
    count_df.to_csv(path+"_count.csv",index=False)
    print("Completed", path)

print("Functions all defined")
