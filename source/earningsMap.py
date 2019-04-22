import plotly
# plotly.tools.set_credentials_file(username='cle', api_key='FGWVL1Oa1la0wPprPRbL')
from PIL import Image, ImageDraw

import plotly.plotly as py
import plotly.figure_factory as ff
import numpy as np
import pandas as pd
import os
import time
from plotly.offline import iplot, init_notebook_mode
import plotly.graph_objs as go
import plotly.io as pio
from load import *

if not os.path.exists('earningMaps'):
        os.mkdir('earningMaps')


earn = parseEarning()

for csv_file in sorted(os.listdir("../data/yearlyEarningData")):
    year_csv = csv_file.split("_")[-1]
    year = year_csv.split(".")[0]
    csv_file= "../data/yearlyEarningData/" + csv_file
    df_sample = pd.read_csv(csv_file) # Read in your data

    fips = df_sample['fips'].tolist() # Read in FIPS Codes
    colorscale = ["#fff7fb","#eee8f3","#d7d6e8","#b5c4df","#8cb2d5","#5c9ec8","#2785bb","#066ca8","#035585","#023858"]
    features = list(df_sample)[2:]
    for feat in features:
        values =  df_sample[feat].tolist() 

        # For colorscale help: https://react-colorscales.getforge.io/
        ind_earnings = earn[feat]
        high = np.max((ind_earnings//1000)*1000)
        low = np.min((ind_earnings//1000)*1000)
        endpts = list(np.linspace(low, high, len(colorscale) - 1)) # Identify a suitable range for your data
        print(feat, year)
        print(endpts)
        title = feat + " Earning for "+ year
        try:
            os.mkdir('earningMaps/'+ feat)
        except:
            pass

        filename = "earningMaps/"+ feat+"/industry_earning_"+year+".png"
        fig = ff.create_choropleth(
            fips=fips, values=values, colorscale=colorscale, show_state_data=True, binning_endpoints=endpts, # If your values is a list of numbers, you can bin your values into half-open intervals
            county_outline={'color': 'rgb(255,255,255)', 'width': 0.5}, 
            legend_title='Earnings for '+ feat , title=title)
        
        pio.write_image(fig, filename)