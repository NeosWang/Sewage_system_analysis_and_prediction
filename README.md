# Data Analysis and time series model prediction for sewage system in Den Bosch

pls place the extract files according to the folder structure as flow, the files in \data are the originally raw data provided by this course
```
|-----\data
|	|-----\rainfall
|	|-----\sewer_data
|	|-----\sewer_data_db
|	|-----\sewer_model	
|
|-----\code
	|-----\1.Sewage_preprocessing
	|	|-----\Bokhoven_M3perMeter.ipynb
	|	|-----\Haarsteeg_M3perMeter.ipynb
	|	|-----\Oude_Engelenseweg_M3perMeter.ipynb
	|
	|-----\2.Rainfall_analysis
	|	|-----\Rainfall_Analysis.ipynb
	|
	|-----\3.Model
	|	|-----\0_Bokhoven_prediction_model.ipynb
	|	|-----\0_Haarsteeg_prediction_model.ipynb
	|	|-----\0_Oude_Engelenseweg_prediction_model.ipynb
	|	|-----\Final_Integrated_prediction_decision_moldel.ipynb
	|
	|-----\asset
	|	|-----\data
	|	|-----\DataSupply.py
	|	|-----\MyPlot.py
	|
	|-----\README.md

```
required packages:
pandas / numpy / matplotlib / tqdm / keras(tensorflow based) / sklearn / itertools / holidays / xgboost / mlxtend

the deliverable from Group 11 is a integrated prediction and decision model to achieve constant inflow to WWTP.
base on the out-flow from 3 pumps stations "Bokhoven" "Haarsteeg" "Oude_Engelenseweg"

all jupyter script could be independently execute 

\1. Sewage_pregrocessing:	data analysis on the specification of sewage system, such as minimum level, level changes and how much volume water will increase 1 meter of water level
\2. Rainfall_analysis:		evaluate the the accuracy of rainfall prediction data "hirlam", bulit predicting model about if the rainfall prediction is accurate base on RNN+LSTM
\3. Models:		predicting the volume of inflow water to the sewage system in dry-day with feature selection, model selection, and parameter tuning.  statistical hypothsis on if feature is significant
	   		make decision about if to switch-off a certain pump for buffer, so that the total out-flow from all sewage is close to the the constant, also won't cause over-flow
\asset:			supporting scripts for data processing, data generating and visualization.