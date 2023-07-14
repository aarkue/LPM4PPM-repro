# LPM4PPM - Reproduction

During the _Machine Learning Applications in Process Mining_ Seminar I will assess the paper titled __Encoding High-Level Control-Flow Construct Information for Process Outcome Prediction__, published by Vazifehdoostirani et al. in 2022.
## Structure
The main approach implementation is in the file `AllInOne.py`. The visualization scripts are located in the `vis/` folder, while the data and results are stored in `data/`.

The `data/` directory contains two folders, the folder `main` where the used event logs and results for the main evaluation are stored.
The `imf-alternative` folder contains the data and results for the alternative process model discovery technique mentioned in the report.

Note, that the available event logs and processed data (i.e., all files inside the `logs`, `xes_logs` and `processed` folders) are zipped for efficiency. Before using them, these files thus need to be extracted.

## Usage
The visualization scripts are Jupyter notebooks, where the folder paths for the results and figures can be changed through the corresponding variables.
The main approach in `AllInOne.py` can be either used as a python script without arguments, by modifying the corresponding variables at the bottom of the file, or with arguments, as described below.

- `log`: the name of the event log to be used. If omitted, the list of event logs included in the python file are used.
- `folder`: the folder in which the LPMs. processed csv and results are stored or should be stored
- `repeats`: the repetitions to execute. The integer values determine the random seed used for the iteration. Multiple values should be seperated using a `,` without a space
- `discoverLPMs`: if true, discover LPMs using the IMf alternative approach, based on the input event log and store them in the lpms directory.
- `skipWithMaxPrefix`: skip pre-processing and additinally use the corresponding value as the maximal trace prefix length.
- `skipPreprocessing`: skip pre-processing (max prefix size will be determined based on the processed cvs file).
- `maxTrials`: maximal number of trials for the HP search

Example commands:
- `python3 AllInOne.py --maxTrials 25 --folder "main" --repeats "0" --log BPIC11_f1 --discoverLPMs`
- `python3 AllInOne.py --maxTrials 25 --folder "main" --repeats "1,2,3,4" --log BPIC11_f1 --skipPreprocessing`
