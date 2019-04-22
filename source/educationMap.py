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


csv_file= "../data/education2012.csv"
df_sample = pd.read_csv(csv_file) # Read in your data

edu_levels = ['pct_less_than_hs', 'pct_hs_diploma', 'pct_college_or_associates', 'pct_college_bachelors_or_higher']
values = df_sample[edu_levels[2:]].sum(axis = 1).tolist() # Read in the values contained within your file
fips = df_sample['fips'].tolist() # Read in FIPS Codes

colorscale = ["#f7fcfd","#e7f6f9","#d1eeec","#aadfd3","#7cccb5","#54b991","#37a266","#1d843f","#00682a","#00441b"]

# For colorscale help: https://react-colorscales.getforge.io/

endpts = list(np.linspace(0, 100, len(colorscale) - 1)) # Identify a suitable range for your data

fig = ff.create_choropleth(
    fips=fips, values=values, colorscale=colorscale, show_state_data=True, binning_endpoints=endpts, # If your values is a list of numbers, you can bin your values into half-open intervals
    county_outline={'color': 'rgb(255,255,255)', 'width': 0.5}, 
    legend_title='Percent college associate or higher', title='Education Attainment 2012-2016'
)
pio.write_image(fig, 'educationMap2012_college.png')