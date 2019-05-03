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
from makeGif import * 


def makeDroughtPNG(input_folder, output_folder, pattern):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    scope = set()
    counties = set()
    for csv_file in sorted(os.listdir("../data/"+ input_folder )):
        if pattern not in csv_file: continue
        csv_file="../data/"+ input_folder  + "/" +csv_file
        df_sample = pd.read_csv(csv_file)
        rel_states = np.unique(df_sample['state'])
        rel_counties = np.unique(df_sample['county'])
        scope = scope.union(set(rel_states))
        counties = counties.union(set(rel_counties))
    print(counties)
    print(scope)
    return 
    colorscale = ["#ffffe5", "#feeba2","#febc46","#f07818","#b74203","#662506"]

    for csv_file in sorted(os.listdir("../data/"+ input_folder )):
        if pattern not in csv_file: continue
        print(csv_file)
        date = csv_file.split("_")[-1]
        date = date.split(".")[0]
        csv_file="../data/"+ input_folder  + "/" +csv_file
        df_sample = pd.read_csv(csv_file)
        # rel_states = np.unique(df_sample['state'])
        # print(rel_states)
        values = df_sample['level_d'].tolist() # Read in values contained within your file
        fips = df_sample['fips'].tolist() # Read in FIPS Codes
         # Create a colorscale
        # "#ffffe5"
        # For colorscale help: https://react-colorscales.getforge.io/

        # endpts = list(np.linspace(0, 4, len(colorscale) - 1)) # Identify a suitable range for your data
        title = 'Drought level for week of '+ str(date)
        fig = ff.create_choropleth(
            fips=fips, values=values, colorscale=colorscale, scope= list(scope),\
            binning_endpoints = [1, 2, 3, 4, 5],
            show_state_data=True, # If your values is a list of numbers, you can bin your values into half-open intervals
            county_outline={'color': 'rgb(150,150,150)', 'width': 0.5}, 
            legend_title='drought level', title= title
        )
        curr_name = 'DroughtLevel_' + pattern + '_wk_'+ str(date) 
        
        pio.write_image(fig, output_folder + "/" +curr_name+".png")

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
        county_outline={'color': 'rgb(200,200,200)', 'width': 0.5}, 
        legend_title='drought level', title= title
    )
    # curr_name = 'DroughtLevel_wk_'+ str(date)
    
    # py.plot(fig, filename = curr_name)
    if save:
        pio.write_image(fig, png_filename)
    # time.sleep(0.5)
    # return fig
# test_g = 33

grps = [10, 14, 16, 21, 24, 27, 43]
scopes = []
for g in grps:
    print(g)
    makeDroughtPNG('weeklyDroughtSubset', 'maps_grp'+ str(g), 'group'+str(g)+"_")
    # makeGif('maps_grp'+str(g), 80, 'grp'+ str(g)+ '.gif')
