# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 19:07:32 2018

@author: collinr
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class clustering(object):
    
    def __init__(self,outpath,clusters=None):
        self.outpath = outpath   
        self.clusters = clusters
    
    #gets the silhouette score and charts for different numbers of clusters    
    def analyse_cluster(self,data):
        for n in self.clusters:  
            fig,(ax1) = plt.subplots(1)
            fig.set_size_inches(18, 7)
    
            ax1.set_xlim([-0.1, 1])
    
            ax1.set_ylim([0, len(data) + (n + 1) * 10])
    
            clusterer = KMeans(n_clusters=n, random_state=10)
            cluster_labels = clusterer.fit_predict(data)
    
            silhouette_avg = silhouette_score(data, cluster_labels)
            print("For n_clusters =", n,
                  "The average silhouette_score is :", silhouette_avg)
        
            sample_silhouette_values = silhouette_samples(data, cluster_labels)
        
            y_lower = 10
            for i in range(n):
                ith_cluster_silhouette_values = \
                    sample_silhouette_values[cluster_labels == i]
        
                ith_cluster_silhouette_values.sort()
        
                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i
        
                color = cm.spectral(float(i) / n)
                ax1.fill_betweenx(np.arange(y_lower, y_upper),
                                  0, ith_cluster_silhouette_values,
                                  facecolor=color, edgecolor=color, alpha=0.7)
        
                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        
                y_lower = y_upper + 10
        
            ax1.set_title("The silhouette plot for the various clusters.")
            ax1.set_xlabel("The silhouette coefficient values")
            ax1.set_ylabel("Cluster label")
        
            ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
        
            ax1.set_yticks([]) 
            ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
        
            plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                          "with n_clusters = %d" % n),
                         fontsize=14, fontweight='bold')
        
            plt.show()

#specific data gathering and clustering for tennis players
class Tennis_cluster(clustering):
    
    def __init__(self,outpath,year,gender,clusters=None,surface=None,data=None):
        self.outpath = outpath   
        self.clusters = clusters        
        self.year = year
        self.gender = gender
        self.surface = surface
        self.data = data
    
    #download the data from Jeffs github    
    def getdata(self):
        data = pd.DataFrame()
        for x in self.year:
            url = 'https://raw.githubusercontent.com/JeffSackmann/tennis_'+self.gender+'/master/' +self.gender+ '_matches_'+str(x)+'.csv'
            df = pd.read_csv(url)
            data = data.append(df)
        self.data = data        
    
    #if you want to exclude any player who doesn't meet the criteria of having played 5 games
    def getexclusions(self):
        exclude = []
        playerd = self.data[['winner_name']]
        for player in playerd['winner_name']:    
            if len(playerd[playerd['winner_name']==player])<5:
                if player not in exclude:
                    exclude.append(player)
        return exclude
    
    #convert the data to be ready for clustering
    def sortdata(self):
        if self.data is None:
            self.getdata()
        dfs = self.data
        if self.surface is not None:
            dfs = dfs[dfs['surface']==self.surface]
        dfs['w_rtnptWon'] = dfs['l_svpt'] - dfs['l_1stWon'] - dfs['l_2ndWon']
        dfs['l_rtnptWon'] = dfs['w_svpt'] - dfs['w_1stWon'] - dfs['w_2ndWon']   
        winners = dfs[['winner_name', 'w_ace', 'w_df',	'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon', 'w_SvGms', 'w_bpSaved', 'w_bpFaced', 'w_rtnptWon', 'l_svpt']]
        winners = winners.rename(columns={'l_svpt': 'loppspvt'})
        winners = winners.groupby('winner_name').sum()
        winners = winners.reset_index()
        losers = dfs[['loser_name', 'l_ace', 'l_df', 'l_svpt', 'l_1stIn', 'l_1stWon', 'l_2ndWon', 'l_SvGms', 'l_bpSaved', 'l_bpFaced', 'l_rtnptWon', 'w_svpt']]
        losers = losers.rename(columns={'w_svpt': 'woppspvt'})
        losers = losers.groupby('loser_name').sum()
        losers = losers.reset_index()
        total = pd.merge(winners, losers, left_on=('winner_name'), right_on=('loser_name'), how='left')
        total['svpt'] = total['w_svpt'] + total['l_svpt']
        total['Ace'] = (total['w_ace'] + total['l_ace'])/total['svpt']
        total['df'] = (total['w_df'] + total['l_df'])/total['svpt']
        total['1stIn'] = total['w_1stIn'] + total['l_1stIn']
        total['1stIn%'] = total['1stIn']/total['svpt']
        total['1stwon'] = (total['w_1stWon'] + total['l_1stWon'])/total['1stIn']
        total['2ndsvpt'] = total['svpt'] - total['1stIn']
        total['2ndWon'] = (total['w_2ndWon'] + total['l_2ndWon'])/total['2ndsvpt']
        total['SvGms'] = total['w_SvGms'] + total['l_SvGms']
        total['bpSaved'] = total['w_bpSaved'] + total['l_bpSaved']
        total['bpFaced'] = total['w_bpFaced'] + total['l_bpFaced']
        total['bpFaced%'] = total['bpFaced']/total['svpt']
        total['broken'] = total['bpFaced']-total['bpSaved']
        total['broken%G'] = total['broken']/total['SvGms']
        total['broken%F'] = total['broken']/total['bpFaced']
        total['rtnptsWon'] = (total['l_rtnptWon']+total['w_rtnptWon'])/(total['loppspvt']+total['woppspvt'])
        df = total[['Ace', 'df', '1stIn%', '1stwon', '2ndWon', 'broken%G', 'broken%F', 'bpFaced%', 'rtnptsWon']]
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        return df, total
    
    #apply the kmeans algo, neeed to give the number of clusters
    def clusterdata(self,cluster):
        df, total = self.sortdata()
        kmeans = KMeans(n_clusters=cluster, random_state=0, n_init=10).fit(df)
        df['cluster'] = kmeans.labels_
        df1 = pd.concat([total['winner_name'], df], axis=1)
        df1 = df1[['winner_name', 'Ace', 'df', '1stIn%', '1stwon', '2ndWon', 'broken%G', 'broken%F', 'bpFaced%', 'rtnptsWon', 'cluster']]
        if self.surface is not None:
            df1.to_csv(outpath + self.gender + '_' + self.surface + '.csv')
        else:
            df1.to_csv(outpath + 'cluster.csv')

 
outpath = 'C:/Users/Rob/Documents/Tennis/'

#set the number of clusters to try for analysis
n_clusters = range(2,9)

#set the years you want to collect data
years = range(2013,2019)

#give surfaces to cluster for
surf = ['Hard', 'Clay', 'Grass']
gender = 'wta'

tennis = Tennis_cluster(outpath,years,gender)
tab1,tab2 = tennis.sortdata()
tennis.analyse_cluster(tab1)

for x in surf:
    tennis = Tennis_cluster(outpath,years,gender,surface=x)
    tennis.clusterdata(5)