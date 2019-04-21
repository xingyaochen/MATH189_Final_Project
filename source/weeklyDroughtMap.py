import plotly
plotly.tools.set_credentials_file(username='cle', api_key='FGWVL1Oa1la0wPprPRbL')

import plotly.plotly as py
import plotly.figure_factory as ff
import numpy as np
import pandas as pd
import os
import time

# Recurses through Drought Level Folder to read each csv file in folder
for csv_file in os.listdir("../data/Weekly Drought Level Graphs"):
    csv_file="../data/Weekly Drought Level Graphs/"+csv_file
    print(csv_file)
    df_sample = pd.read_csv(csv_file)
    values = df_sample['level_d'].tolist() # Read in values contained within your file
    fips = df_sample['fips'].tolist() # Read in FIPS Codes
    colorscale = ["#171c42","#1267b2","#8cb5c9","#dab2be","#c46852","#701b20"] # Create a colorscale

    # For colorscale help: https://react-colorscales.getforge.io/
    time.sleep(50)

    endpts = list(np.linspace(0, 4, len(colorscale) - 1)) # Identify a suitable range for your data

    fig = ff.create_choropleth(
        fips=fips, values=values, colorscale=colorscale, show_state_data=True, # If your values is a list of numbers, you can bin your values into half-open intervals
        county_outline={'color': 'rgb(255,255,255)', 'width': 0.5}, 
        legend_title='drought level', title='Drought level for week of'
    )
    py.iplot(fig, filename = 'Weekly Drought Level for week of')


# def createLevels(final_drought_dt = df_sample):
#     levels = ['d0', 'd1', 'd2', 'd3', 'd4']
#     final_drought_dt['level_d'] = np.zeros(final_drought_dt.shape[0])
#     for i, level in enumerate(levels):
#         per10Ind = final_drought_dt[level]>10.0
#         final_drought_dt['level_d'].loc[per10Ind] = i*np.ones(sum(per10Ind))
#     return final_drought_dt