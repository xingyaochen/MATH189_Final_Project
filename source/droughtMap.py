import plotly
plotly.tools.set_credentials_file(username='cle', api_key='FGWVL1Oa1la0wPprPRbL')

import plotly.plotly as py
import plotly.figure_factory as ff
import numpy as np
import pandas as pd


csv_file= "../data/final_drought2010.csv"
df_sample = pd.read_csv(csv_file) # Read in your data

def createLevels(final_drought_dt = df_sample):
    levels = ['d0', 'd1', 'd2', 'd3', 'd4']
    final_drought_dt['level_d'] = np.zeros(final_drought_dt.shape[0])
    for i, level in enumerate(levels):
        per10Ind = final_drought_dt[level]>10.0
        final_drought_dt['level_d'].loc[per10Ind] = i*np.ones(sum(per10Ind))
    return final_drought_dt

df_sample = createLevels(df_sample)

values = df_sample['level_d'].tolist() # Read in the values contained within your file
fips = df_sample['fips'].tolist() # Read in FIPS Codes

colorscale = ["#171c42","#4590c4","#dab2be","#a63329","#3c0911"] # Create a colorscale

# For colorscale help: https://react-colorscales.getforge.io/

endpts = list(np.linspace(0, 4, len(colorscale) - 1)) # Identify a suitable range for your data

fig = ff.create_choropleth(
    fips=fips, values=values, colorscale=colorscale, show_state_data=True, binning_endpoints=endpts, # If your values is a list of numbers, you can bin your values into half-open intervals
    county_outline={'color': 'rgb(255,255,255)', 'width': 0.5}, 
    legend_title='% change', title='% Domestic Water Usage in 2010'
)
py.plot(fig, filename='Drought Levels 2010')