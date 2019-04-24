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

# Recurses through Drought Level Folder to read each csv file in folder

def makeDroughtPNG():
    if not os.path.exists('maps'):
        os.mkdir('maps')

    for csv_file in sorted(os.listdir("../data/Weekly Drought Level Graphs")):
        print(csv_file)
        date = csv_file.split("_")[-1]
        date = date.split(".")[0]
        csv_file="../data/Weekly Drought Level Graphs/"+csv_file
        # date = date.replace("-", " ")
        df_sample = pd.read_csv(csv_file)
        values = df_sample['level_d'].tolist() # Read in values contained within your file
        fips = df_sample['fips'].tolist() # Read in FIPS Codes
        colorscale = ["#ffffe5", "#feeba2","#febc46","#f07818","#b74203","#662506"] # Create a colorscale
        # "#ffffe5"
        # For colorscale help: https://react-colorscales.getforge.io/

        # endpts = list(np.linspace(0, 4, len(colorscale) - 1)) # Identify a suitable range for your data

        fig = ff.create_choropleth(
            fips=fips, values=values, colorscale=colorscale, show_state_data=True, # If your values is a list of numbers, you can bin your values into half-open intervals
            county_outline={'color': 'rgb(255,255,255)', 'width': 0.5}, 
            legend_title='drought level', title='Drought level for week of '+ str(date)
        )
        curr_name = 'DroughtLevel_wk_'+ str(date)
        
        # py.plot(fig, filename = curr_name)
        pio.write_image(fig, 'maps/'+curr_name+".png")

        # time.sleep(0.5)
        print('Done with week ' + date)




def makeOneDroughtPNG(csv_file, png_filename, value, title, save = False):
    if not os.path.exists('maps'):
        os.mkdir('maps')
    print(csv_file)
    # date = csv_file.split("_")[-2]
    # date = date.split(".")[0]
    # csv_file="../data/Weekly Drought Level Graphs/"+csv_file
    # date = date.replace("-", " ")
    df_sample = pd.read_csv(csv_file)
    values = df_sample[value].tolist() # Read in values contained within your file
    fips = df_sample['fips'].tolist() # Read in FIPS Codes
    colorscale = ["#ffffe5", "#feeba2","#febc46","#f07818","#b74203","#662506"] # Create a colorscale
    # "#ffffe5"
    # For colorscale help: https://react-colorscales.getforge.io/

    # endpts = list(np.linspace(0, 4, len(colorscale) - 1)) # Identify a suitable range for your data

    fig = ff.create_choropleth(
        fips=fips, values=values, colorscale=colorscale, show_state_data=True, # If your values is a list of numbers, you can bin your values into half-open intervals
        county_outline={'color': 'rgb(255,255,255)', 'width': 0.5}, 
        legend_title='drought level', title= title
    )
    # curr_name = 'DroughtLevel_wk_'+ str(date)
    
    # py.plot(fig, filename = curr_name)
    if save:
        pio.write_image(fig, png_filename)
    # time.sleep(0.5)
    # return fig

