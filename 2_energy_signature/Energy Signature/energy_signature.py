#!/usr/bin/env python
# coding: utf-8

# In[1]:


#%matplotlib notebook
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from sklearn.metrics import mean_squared_error


# In[2]:


df=pd.read_csv('my_data.csv',sep=',',decimal='.',index_col=0)


# In[3]:


df.index=pd.to_datetime(df.index,unit='s')


# In[4]:


df.index


# In[5]:


df.columns


# In[6]:


df['deltaT']=df.T_in.astype(float)-df.T_ex.astype(float)
df['T_in']=df.T_in.astype(float)
df['T_ex']=df.T_ex.astype(float)
df['Heat']=df.Heat.astype(float)
df['Cool']=df.Cool.astype(float)
df['Solar']=df.Solar.astype(float)


# # 10 minutes

# In[7]:


# Extract the cooling and heating dataframes and drop null values
cool_d = df.where(df['Cool']!=0.0).dropna()
heat_d = df.where(df['Heat']!=0.0).dropna()


# In[8]:


# Fit the regression models
model_h = sm.OLS(heat_d.Heat,sm.add_constant(heat_d.deltaT))
model_c = sm.OLS(cool_d.Cool,sm.add_constant(cool_d.deltaT))


# In[9]:


results_h = model_h.fit()
results_c = model_c.fit()


# In[10]:


# Plot the results
fig = plt.figure(figsize=(8,5))
plt.plot(heat_d.deltaT,results_h.predict(),'r', linewidth=1, label='Heating Regression Line')
plt.plot(cool_d.deltaT,results_c.predict(),'k',linestyle='-.', linewidth=1, label='Cooling Regression Line')
plt.scatter(heat_d.deltaT,heat_d.Heat, s=10, label='Observations heating')
plt.scatter(cool_d.deltaT,cool_d.Cool, s=25, color='r',marker='s', label='Observations cooling')
plt.xlabel('\u0394T [\u00B0C]')
plt.ylabel('Energy Consumption [kWh]')
plt.ylim(-0.01,2.9)
plt.legend()
plt.show()

# In[11]:


rmse_c = mean_squared_error(cool_d.Cool.values, results_c.predict(), squared=False)
print('The RMSE for 10 minutes of resolution for Cooling Consumptions is: '+str(rmse_c)+' kWh')
results_c.summary()


# In[12]:


rmse_h = mean_squared_error(heat_d.Heat.values, results_h.predict(), squared=False)
print('The RMSE for 10 minutes of resolution for Heating Consumptions is: '+str(rmse_h)+' kWh')
results_h.summary()


# # Hourly

# In[13]:


df=df.resample('H').mean()
df=df.dropna()


# In[14]:


df.index


# In[15]:


# Extract the cooling and heating dataframes and drop null values
cool_d = df.where(df['Cool']!=0.0).dropna()
heat_d = df.where(df['Heat']!=0.0).dropna()


# In[16]:


# Fit the regression models
model_h = sm.OLS(heat_d.Heat,sm.add_constant(heat_d.deltaT))
model_c = sm.OLS(cool_d.Cool,sm.add_constant(cool_d.deltaT))


# In[17]:


results_h = model_h.fit()
results_c = model_c.fit()


# In[18]:


# Plot the results
fig = plt.figure(figsize=(8,5))
plt.plot(heat_d.deltaT,results_h.predict(),'r', linewidth=1, label='Heating Regression Line')
plt.plot(cool_d.deltaT,results_c.predict(),'k', linestyle='-.', linewidth=1, label='Cooling Regression Line')
plt.scatter(heat_d.deltaT,heat_d.Heat, s=10, label='Observations heating')
plt.scatter(cool_d.deltaT,cool_d.Cool, s=25, color='r', marker='s',label='Observations cooling')
plt.xlabel('\u0394T [\u00B0C]')
plt.ylabel('Energy Consumption [kWh]')
plt.ylim(-0.01,1.27)
plt.legend()
plt.show()

# In[19]:

rmse_c = mean_squared_error(cool_d.Cool.values, results_c.predict(), squared=False)
print('The RMSE for hourly resolution for Cooling Consumptions is: '+str(rmse_c)+' kWh')
results_c.summary()


# In[20]:

rmse_h = mean_squared_error(heat_d.Heat.values, results_h.predict(), squared=False)
print('The RMSE for hourly resolution for Heating Consumptions is: '+str(rmse_h)+' kWh')
results_h.summary()


# # Dayly

# In[21]:


df=df.resample('D').mean()
df=df.dropna()


# In[22]:


# Extract the cooling and heating dataframes and drop null values
cool_d = df.where(df['Cool']!=0.0).dropna()
heat_d = df.where(df['Heat']!=0.0).dropna()


# In[23]:


# Fit the regression models
model_h = sm.OLS(heat_d.Heat,sm.add_constant(heat_d.deltaT))
model_c = sm.OLS(cool_d.Cool,sm.add_constant(cool_d.deltaT))


# In[24]:


results_h = model_h.fit()
results_c = model_c.fit()


# In[25]:


# Plot the results
fig = plt.figure(figsize=(8,5))
plt.plot(heat_d.deltaT,results_h.predict(),'r', linewidth=1, label='Heating Regression Line')
plt.plot(cool_d.deltaT,results_c.predict(),'k', linestyle='-.',linewidth=1, label='Cooling Regression Line')
plt.scatter(heat_d.deltaT,heat_d.Heat, s=10, label='Observations heating')
plt.scatter(cool_d.deltaT,cool_d.Cool, s=30, color='r',marker='s', label='Observations cooling')
plt.xlabel('\u0394T [\u00B0C]')
plt.ylabel('Energy Consumption [kWh]')
plt.ylim(-0.01,0.8)
plt.legend(loc='upper left')
plt.show()

# In[26]:

rmse_c = mean_squared_error(cool_d.Cool.values, results_c.predict(), squared=False)
print('The RMSE for dayly resolution for Cooling Consumptions is: '+str(rmse_c)+' kWh')
results_c.summary()


# In[27]:


rmse_h = mean_squared_error(heat_d.Heat.values, results_h.predict(), squared=False)
print('The RMSE for dayly resolution for Heating Consumptions is: '+str(rmse_h)+' kWh')
results_h.summary()


# # Weekly

# In[28]:


df=df.resample('W').mean()
df=df.dropna()


# In[29]:


# Extract the cooling and heating dataframes and drop null values
cool_d = df.where(df['Cool']!=0.0).dropna()
heat_d = df.where(df['Heat']!=0.0).dropna()
heat_d.index


# In[30]:


heat_d.drop(heat_d.index[0], inplace=True)


# In[31]:


# Fit the regression models
model_h = sm.OLS(heat_d.Heat,sm.add_constant(heat_d.deltaT))
model_c = sm.OLS(cool_d.Cool,sm.add_constant(cool_d.deltaT))


# In[32]:


results_h = model_h.fit()
results_c = model_c.fit()


# In[33]:


# Plot the results
fig = plt.figure(figsize=(8,5))
plt.plot(heat_d.deltaT,results_h.predict(),'r', linewidth=1, label='Heating Regression Line')
plt.plot(cool_d.deltaT,results_c.predict(),'k', linestyle='-.', linewidth=1, label='Cooling Regression Line')
plt.scatter(heat_d.deltaT,heat_d.Heat, s=10, label='Observations heating', marker='o')
plt.scatter(cool_d.deltaT,cool_d.Cool, s=30, color='r', marker='s',label='Observations cooling')
plt.xlabel('\u0394T [\u00B0C]')
plt.ylabel('Energy Consumption [kWh]')
plt.ylim(-0.01,0.32)
plt.legend(loc='upper left')
plt.show()

# In[34]:


rmse_c = mean_squared_error(cool_d.Cool.values, results_c.predict(), squared=False)
print('The RMSE for weekly resolution for Cooling Consumptions is: '+str(rmse_c)+' kWh')
results_c.summary()


# In[35]:


rmse_h = mean_squared_error(heat_d.Heat.values, results_h.predict(), squared=False)
print('The RMSE for weekly resolution for Heating Consumptions is: '+str(rmse_h)+' kWh')
results_h.summary()


# # By adding solar radiation as further regressor

# # 10 minutes

# In[36]:


df=pd.read_csv('my_data.csv',sep=',',decimal=',',index_col=0)


# In[37]:


df.index=pd.to_datetime(df.index,unit='s')


# In[38]:


df['deltaT']=df.T_in.astype(float)-df.T_ex.astype(float)
df['T_in']=df.T_in.astype(float)
df['T_ex']=df.T_ex.astype(float)
df['Heat']=df.Heat.astype(float)
df['Cool']=df.Cool.astype(float)
df['Solar']=df.Solar.astype(float)


# In[39]:


# Extract the cooling and heating dataframes
cool_d = df.where(df['Cool']!=0.0).dropna()
heat_d = df.where(df['Heat']!=0.0).dropna()


# In[40]:


# Fit the regression models
y_h = heat_d.Heat
y_c = cool_d.Cool
X_h = heat_d.loc[:, ['deltaT', 'Solar']]
X_c = cool_d.loc[:, ['deltaT', 'Solar']]
X_h = sm.add_constant(X_h)
X_c = sm.add_constant(X_c)

model_h = sm.OLS(y_h,X_h)
model_c = sm.OLS(y_c,X_c)


# In[41]:


results_h = model_h.fit()
results_c = model_c.fit()


# In[42]:


# Plot the results
xx1c, xx2c = np.meshgrid(np.linspace(cool_d.deltaT.min(), cool_d.deltaT.max(), 100), 
                         np.linspace(cool_d.Solar.min(), cool_d.Solar.max(), 100))
Zc = results_c.params[0] + results_c.params[1] * xx1c + results_c.params[2] * xx2c

xx1h, xx2h = np.meshgrid(np.linspace(heat_d.deltaT.min(), heat_d.deltaT.max(), 100), 
                         np.linspace(heat_d.Solar.min(), heat_d.Solar.max(), 100))
Zh = results_h.params[0] + results_h.params[1] * xx1h + results_h.params[2] * xx2h


fig = plt.figure(figsize=(12, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(cool_d.deltaT, cool_d.Solar, cool_d.Cool, marker = 's', label='Observations Cooling')
ax.scatter(heat_d.deltaT, heat_d.Solar, heat_d.Heat, label='Observations Heating')
ax.plot_surface(xx1c, xx2c, Zc, color='r', alpha=0.8, linewidth=0)
ax.plot_surface(xx1h, xx2h, Zh, color='black', alpha=0.8, linewidth=0)

ax.set_xlabel('\u0394T [\u00B0C]')
ax.set_ylabel('Solar Radiation [W/m2]')
ax.set_zlabel('Energy Consumptions [kWh]')
ax.set_title('Energy Consumption Regression using Solar Radiation (10 minutes resolution)')
ax.view_init(elev=34, azim=-35)
# # rotate the axes and update
# for angle in range(0, 360):
#     ax.view_init(45, angle)
#     plt.draw()
#     plt.pause(.001)

scatter_cool = Line2D([0], [0], marker='s',color='white', label='Observations Cooling',
                      markerfacecolor='blue', markersize=15)
scatter_heat = Line2D([0], [0], marker='o',color='white', label='Observations Heating',
                      markerfacecolor='orange', markersize=15)
surf_cool = mpatches.Patch(color='red',label='Surface Cooling')
surf_heat = mpatches.Patch(color='black',label='Surface Heating')
ax.legend(handles=[scatter_cool,scatter_heat,surf_cool,surf_heat])
plt.show()

# In[43]:

rmse_c = mean_squared_error(cool_d.Cool.values, results_c.predict(), squared=False)
print('The RMSE for 10 minutes resolution for Cooling Consumptions is: '+str(rmse_c)+' kWh')
results_c.summary()


# In[44]:

rmse_h = mean_squared_error(heat_d.Heat.values, results_h.predict(), squared=False)
print('The RMSE for 10 minutes resolution for Heating Consumptions is: '+str(rmse_h)+' kWh')
results_h.summary()


# # Hourly

# In[45]:


df=df.resample('H').mean()
df=df.dropna()


# In[46]:


# Extract the cooling and heating dataframes
cool_d = df.where(df['Cool']!=0.0).dropna()
heat_d = df.where(df['Heat']!=0.0).dropna()


# In[47]:


# Fit the regression models
y_h = heat_d.Heat
y_c = cool_d.Cool
X_h = heat_d.loc[:, ['deltaT', 'Solar']]
X_c = cool_d.loc[:, ['deltaT', 'Solar']]
X_h = sm.add_constant(X_h)
X_c = sm.add_constant(X_c)

model_h = sm.OLS(y_h,X_h)
model_c = sm.OLS(y_c,X_c)


# In[48]:


results_h = model_h.fit()
results_c = model_c.fit()


# In[49]:


# Plot the results
xx1c, xx2c = np.meshgrid(np.linspace(cool_d.deltaT.min(), cool_d.deltaT.max(), 100), 
                         np.linspace(cool_d.Solar.min(), cool_d.Solar.max(), 100))
Zc = results_c.params[0] + results_c.params[1] * xx1c + results_c.params[2] * xx2c

xx1h, xx2h = np.meshgrid(np.linspace(heat_d.deltaT.min(), heat_d.deltaT.max(), 100), 
                         np.linspace(heat_d.Solar.min(), heat_d.Solar.max(), 100))
Zh = results_h.params[0] + results_h.params[1] * xx1h + results_h.params[2] * xx2h


fig = plt.figure(figsize=(12, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(cool_d.deltaT, cool_d.Solar, cool_d.Cool, marker = 's', label='Observations Cooling')
ax.scatter(heat_d.deltaT, heat_d.Solar, heat_d.Heat, label='Observations Heating')
ax.plot_surface(xx1c, xx2c, Zc, color='r', alpha=0.8, linewidth=0)
ax.plot_surface(xx1h, xx2h, Zh, color='black', alpha=0.8, linewidth=0)

ax.set_xlabel('\u0394T [\u00B0C]')
ax.set_ylabel('Solar Radiation [W/m2]')
ax.set_zlabel('Energy Consumptions [kWh]')
ax.set_title('Energy Consumption Regression using Solar Radiation (Hourly resolution)')
# ax.view_init(elev=34, azim=-35)
# ax.view_init(elev=17, azim=-83)

# # rotate the axes and update
# for angle in range(0, 360):
#     ax.view_init(45, angle)
#     plt.draw()
#     plt.pause(.001)

scatter_cool = Line2D([0], [0], marker='s',color='white', label='Observations Cooling',
                      markerfacecolor='blue', markersize=15)
scatter_heat = Line2D([0], [0], marker='o',color='white', label='Observations Heating',
                      markerfacecolor='orange', markersize=15)
surf_cool = mpatches.Patch(color='red',label='Surface Cooling')
surf_heat = mpatches.Patch(color='black',label='Surface Heating')
ax.legend(handles=[scatter_cool,scatter_heat,surf_cool,surf_heat])
plt.show()

# In[50]:

rmse_c = mean_squared_error(cool_d.Cool.values, results_c.predict(), squared=False)
print('The RMSE for hourly resolution for Cooling Consumptions is: '+str(rmse_c)+' kWh')
results_c.summary()


# In[51]:

rmse_h = mean_squared_error(heat_d.Heat.values, results_h.predict(), squared=False)
print('The RMSE for hourly resolution for Heating Consumptions is: '+str(rmse_h)+' kWh')
results_h.summary()


# # Dayly

# In[52]:


df=df.resample('D').mean()
df=df.dropna()


# In[53]:


# Extract the cooling and heating dataframes
cool_d = df.where(df['Cool']!=0.0).dropna()
heat_d = df.where(df['Heat']!=0.0).dropna()


# In[54]:


# Fit the regression models
y_h = heat_d.Heat
y_c = cool_d.Cool
X_h = heat_d.loc[:, ['deltaT', 'Solar']]
X_c = cool_d.loc[:, ['deltaT', 'Solar']]
X_h = sm.add_constant(X_h)
X_c = sm.add_constant(X_c)

model_h = sm.OLS(y_h,X_h)
model_c = sm.OLS(y_c,X_c)


# In[55]:


results_h = model_h.fit()
results_c = model_c.fit()


# In[56]:


# Plot the results
xx1c, xx2c = np.meshgrid(np.linspace(cool_d.deltaT.min(), cool_d.deltaT.max(), 100), 
                         np.linspace(cool_d.Solar.min(), cool_d.Solar.max(), 100))
Zc = results_c.params[0] + results_c.params[1] * xx1c + results_c.params[2] * xx2c

xx1h, xx2h = np.meshgrid(np.linspace(heat_d.deltaT.min(), heat_d.deltaT.max(), 100), 
                         np.linspace(heat_d.Solar.min(), heat_d.Solar.max(), 100))
Zh = results_h.params[0] + results_h.params[1] * xx1h + results_h.params[2] * xx2h


fig = plt.figure(figsize=(12, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(cool_d.deltaT, cool_d.Solar, cool_d.Cool, marker = 's')
ax.scatter(heat_d.deltaT, heat_d.Solar, heat_d.Heat)
ax.plot_surface(xx1c, xx2c, Zc, color='r', alpha=0.8, linewidth=0)
ax.plot_surface(xx1h, xx2h, Zh, color='black', alpha=0.8, linewidth=0)

ax.set_xlabel('\u0394T [\u00B0C]')
ax.set_ylabel('Solar Radiation [W/m2]')
ax.set_zlabel('Energy Consumptions [kWh]]')
ax.set_title('Energy Consumption Regression using Solar Radiation (Dayly resolution)')
ax.view_init(elev=17, azim=-83)
# # rotate the axes and update
# for angle in range(0, 360):
#     ax.view_init(45, angle)
#     plt.draw()
#     plt.pause(.001)

scatter_cool = Line2D([0], [0], marker='s',color='white', label='Observations Cooling',
                      markerfacecolor='blue', markersize=15)
scatter_heat = Line2D([0], [0], marker='o',color='white', label='Observations Heating',
                      markerfacecolor='orange', markersize=15)
surf_cool = mpatches.Patch(color='red',label='Surface Cooling')
surf_heat = mpatches.Patch(color='black',label='Surface Heating')
ax.legend(handles=[scatter_cool,scatter_heat,surf_cool,surf_heat])
plt.show()

# In[57]:

rmse_c = mean_squared_error(cool_d.Cool.values, results_c.predict(), squared=False)
print('The RMSE for dayly resolution for Cooling Consumptions is: '+str(rmse_c)+' kWh')
results_c.summary()


# In[58]:

rmse_h = mean_squared_error(heat_d.Heat.values, results_h.predict(), squared=False)
print('The RMSE for dayly resolution for Heating Consumptions is: '+str(rmse_h)+' kWh')
results_h.summary()


# # Weekly

# In[59]:


df=df.resample('W').mean()
df=df.dropna()


# In[60]:


# Extract the cooling and heating dataframes
cool_d = df.where(df['Cool']!=0.0).dropna()
heat_d = df.where(df['Heat']!=0.0).dropna()
heat_d.drop(heat_d.index[0], inplace=True)
heat_d.drop(heat_d.index[-1],inplace=True)


# In[61]:


# Fit the regression models
y_h = heat_d.Heat
y_c = cool_d.Cool
X_h = heat_d.loc[:, ['deltaT', 'Solar']]
X_c = cool_d.loc[:, ['deltaT', 'Solar']]
X_h = sm.add_constant(X_h)
X_c = sm.add_constant(X_c)

model_h = sm.OLS(y_h,X_h)
model_c = sm.OLS(y_c,X_c)


# In[62]:


results_h = model_h.fit()
results_c = model_c.fit()


# In[78]:


# Plot the results
xx1c, xx2c = np.meshgrid(np.linspace(cool_d.deltaT.min(), cool_d.deltaT.max(), 100), 
                         np.linspace(cool_d.Solar.min(), cool_d.Solar.max(), 100))
Zc = results_c.params[0] + results_c.params[1] * xx1c + results_c.params[2] * xx2c

xx1h, xx2h = np.meshgrid(np.linspace(heat_d.deltaT.min(), heat_d.deltaT.max(), 100), 
                         np.linspace(heat_d.Solar.min(), heat_d.Solar.max(), 100))
Zh = results_h.params[0] + results_h.params[1] * xx1h + results_h.params[2] * xx2h


fig = plt.figure(figsize=(12, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(cool_d.deltaT, cool_d.Solar, cool_d.Cool, marker = 's',label='Observations Cooling')
ax.scatter(heat_d.deltaT, heat_d.Solar, heat_d.Heat, label='Observations Heating')
ax.plot_surface(xx1c, xx2c, Zc, color='r', alpha=0.8, linewidth=0)
ax.plot_surface(xx1h, xx2h, Zh, color='black', alpha=0.8, linewidth=0)

ax.set_xlabel('\u0394T [\u00B0C]')
ax.set_ylabel('Solar Radiation [W/m2]')
ax.set_zlabel('Energy Consumptions [kWh]]')
ax.set_title('Energy Consumption Regression using Solar Radiation (Weekly resolution)')
#ax.view_init(elev=18, azim=-77)
ax.view_init(elev=29, azim=-72)
# # rotate the axes and update
# for angle in range(0, 360):
#     ax.view_init(45, angle)
#     plt.draw()
    

scatter_cool = Line2D([0], [0], marker='s',color='white', label='Observations Cooling',
                      markerfacecolor='blue', markersize=15)
scatter_heat = Line2D([0], [0], marker='o',color='white', label='Observations Heating',
                      markerfacecolor='orange', markersize=15)
surf_cool = mpatches.Patch(color='red',label='Surface Cooling')
surf_heat = mpatches.Patch(color='black',label='Surface Heating')
ax.legend(handles=[scatter_cool,scatter_heat,surf_cool,surf_heat])
plt.show()



rmse_c = mean_squared_error(cool_d.Cool.values, results_c.predict(), squared=False)
print('The RMSE for weekly resolution for Cooling Consumptions is: '+str(rmse_c)+' kWh')
results_c.summary()


# In[58]:

rmse_h = mean_squared_error(heat_d.Heat.values, results_h.predict(), squared=False)
print('The RMSE for weekly resolution for Heating Consumptions is: '+str(rmse_h)+' kWh')
results_h.summary()




