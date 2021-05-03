# ICT in Building Design
This work is part of the project for the ICT in Building Design course at Politecnico di Torino.  
The main purpose of the work is to develop a simple office building in Oslo (NOR) in order to evaluate its energy efficiency. Furthermore the building energy classification is performed through the Energy Signature technique and finally a machine-learning based predictive control system for the internal temperature and annual energy consumptions is attempted. 


<!-- ABSTRACT -->
## Abstract
With the emerging of smart grids as an answer to the need of sustainability, it rises in relevance the ability to predict the consumption of a building. This is particularly true for offices, where the consumption to
guarantee visual and thermal comfort are constantly high. A reliable forecast of electricity usage helps now the energy manager to adjust the scheduling
and reduce wastes and in the next future, it will allow the grid operator to prevent peak usage and push demand-response policy. In this paper, we
analyzed the strategies already proposed to predict multiple consumption, with a small time granularity. We modelled an office in a building in Oslo by
using DesignBuilder software, and exploited EnergyPlus to collect simulated measurements of internal energy absorption from the year 2017 to the year
2019. Firstly, we applied the Energy Signature method twice: with a simple linear regression model and with a multilinear regression model, having the
difference between indoor and outdoor temperatures and global horizontal solar radiation as independent variables. The analysis was applied on data
sampled at different time intervals: 10 minutes, hourly, daily and weekly. The results show that the best model is the multi linear one obtained with
a weekly resolution, with an adjusted coefficient of determination R2_adj=0.93 for cooling consumption and R2_adj=0.90 for heating consumption. Then we built our version of a state-of-the-art deep neural network, composed of a
convolutional model and a long short term memory one, trained to predict the next hour given the past one. Lastly, we simultaneously extended both the
past window and the forecast period finding that it is possible to predict the next week with MAE 0.04 kWh for heating consumption and MAE 0.02
kWh for cooling and internal light electricity consumption.
To see the full work please see the [documentation](https://github.com/FrancescoConforte/ICT-in-Building-Design/blob/main/Report.pdf)

### Built With
* [DesignBuilder](http://designbuilderitalia.it/)
* [Energy+](https://energyplus.net/)
* [Python3](https://www.python.org/download/releases/3.0/)


## 
(c) 2020, Francesco Conforte, Paolo De Santis, Andrea Minardi, Can Akgol, Federico Fabiani.
