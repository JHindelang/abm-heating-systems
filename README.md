# Agent-based heating system diffusion model
Agent based model of energy transition in the residential sector

## Requirements
Python 3.6

Packages and dependencies can be installed using pip from requirements.txt

## Description of scripts

Model Scripts: <ol>

model_geo.py -> basic file of the abm

agents_geo.py -> household agents and central agents classes are defined here

heating_system.py -> heating system calss is defined here

global_functions.py -> functions reading the input data

</ol>Executables: <ol>

agent_preprocessing.py -> runs preprocession; dialogue opens to select shape files for households and for district heating grid, resulting shape file is saved as /input/geo_data/shape_file_households/households.shp; input files for geospatial data are searched in /input/geo_data/ and need to have the same names as the sample data .dat file containig the input factors is saved to /input/ind_fact.dat

run_geo.py -> runs the simulation with the data in the input folder and the households.shp as basis for households
	number of time steps, starting data and model parameters are specified in /input/modelsettings.xlsx
	simulation results are stored in /output/simulation_result.xlsx

web_app.py -> starts the web app to display the simulation, data in input folder are used

<ol> buttons: 	 <ol>

step: computes one time step
      
reset: resets the models and uses the present parameters
			
run: simulation is started
			
stop: simulation is stopped
			
exit: web app is shut down, browser can be closed
			
save: saves the present model parameters to /output/params_calib

</ol></ol> web_app_calibration.py -> similar to, web app: RMSD is coputed for calibration data and group plot for calibration is displayed
