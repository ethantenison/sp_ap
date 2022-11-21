##############################################Set up
# streamlit run "suggested_points_10perc.py"
# cd /Users/et/Desktop/Data_Projects/Sunthetics-Algorithm/
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json
import matplotlib.pyplot as plt
from sklearn.preprocessing import PowerTransformer
import pickle

exec(open("app_functions.py").read())

st.set_page_config(layout="wide")
st.title('Testing Suggested Points with Uncertain Values Included')
#st.subheader('Categorical-5 initial Experiments')

#y = d10['ann']['max']['objective_5']['randomseed_0']['initial_y']

##################################################Data Cleanup
with open('data/ann_max_35678_init10inp4sp_10_regular.pkl', 'rb') as f:
  d10 = pickle.load(f)

with open('data/ann_max_35678_init10inp4sp_25_regular.pkl', 'rb') as f:
  d25 = pickle.load(f)

with open('data/ann_max_35678_init10inp4sp_50_regular.pkl', 'rb') as f:
  d50 = pickle.load(f)


regular = {
  10 : d10,
  25 : d25,
  50 : d50
}

with open('data/ann_max_35678_init10inp4sp_10_regular_unc.pkl', 'rb') as f:
  d10 = pickle.load(f)

with open('data/ann_max_35678_init10inp4sp_25_regular_unc.pkl', 'rb') as f:
  d25 = pickle.load(f)

with open('data/ann_max_35678_init10inp4sp_50_regular_unc.pkl', 'rb') as f:
  d50 = pickle.load(f)

unc = {
  10 : d10,
  25 : d25,
  50 : d50
}
##################################################App layout


one, two, three = st.columns([2,6, 6])

sp = one.selectbox(
  'Suggested Number of Points',
  (10,25,50)
)

obj = one.selectbox(
  "Objective",
  ("objective_3", "objective_5", "objective_6", "objective_7", "objective_8")
  )

seed = one.selectbox(
  "Seed",
  ('randomseed_0','randomseed_1', 'randomseed_2', 'randomseed_3', 'randomseed_4',
   'randomseed_5', 'randomseed_6', 'randomseed_7')
  )


##################################################Functions

fig, opt, fig_his = st_plots_sp(regular, sp, obj)
fig_train, fig_pred = algo_plots(regular, sp, obj, seed)

fig2, opt, fig_his = st_plots_sp(unc, sp, obj)
fig_train2, fig_pred2 = algo_plots(unc, sp, obj, seed)

################################################## Plot!

# one.subheader('Select Data')

one.pyplot(fig_his)

two.header("Current")
two.plotly_chart(fig, use_container_width=True)
two.plotly_chart(fig_train, use_container_width=True)
two.plotly_chart(fig_pred, use_container_width=True)

three.header("10%")
three.plotly_chart(fig2, use_container_width=True)
three.plotly_chart(fig_train2, use_container_width=True)
three.plotly_chart(fig_pred2, use_container_width=True)




