# Closed-form solution for shallow, high-altitude, atmospheric flight

## Quick Notes

- To run all simulations at once and obtain results, run `all_sims.py`
- To run a single simulation and obtain results, run `main2_0.py`
- `closed_form_solution.py` contains helper functions to compute the current closed form solution
- `data_analysis.py` contains parsing functions for the drag passage results in the `DragPassageResults` folder
- `lamberts_problem.py` contains helper functions to solve Lambert's problem in different ways
- the `old_code` folder should not be necessary

## Specifics

- In `data` folder, you will find the csv file `data_all_sims.csv`, containing data resulting from running the corresponding script, `all_sims.py`. This csv file is being used to study the relationship between the error coefficients and known parameters to the user in the python script `data_corelation.py`