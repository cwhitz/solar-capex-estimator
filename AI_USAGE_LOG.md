# AI Usage Log

## Tools Used (Most to Least)
- **Copilot**: Assisted with code generation, refactoring, and documentation throughout.
- **Claude Code**: Used to initiate project structure, add additional commentary to notebook, and generate tests.
- **Copilot on Pull Requests**: Used to review and suggest improvements on code changes, many of which were accepted with minor modifications.
- **ChatGPT**: Misc. troubleshooting. Areas of especially helpful guidance included:
    - accessing hyperparameters for tuning.
    - calculating prediction intervals from Random Forests.
    - misc error handling.


## Example Prompts (Claude Code)

Project initialization
> Set up initial project structure for machine learning model development and lightweight deployment. There should be a folder for:  notebooks used for model developments, data, model storage, src, and a [main.py](http://main.py) containing an empty SolarCapexEstimator class. Docstring style should be numpy.
> 
> Do not fill in any of the files beyond preliminary skeleton code and do not make any assumptions about the task.

Notebook commentary
> Add commentary to the Model_Development notebook that explains the rationale behind each step of the workflow where it is not already clear. Commentary should be concise but informative, and should be written for an audience of data scientists who are familiar with the general process of model development but may not be familiar with the specific techniques used in this notebook.

Tests
> Write and run a set of tests in a new folder called "tests" using pytest. Be mindful of memory usage. Please write tests for each of the files in the steps/ folder
>

## Accepted, modified, or rejected

I accepted most suggestions from AI during code review as I found them helpful for catching edge cases. I modified a lot of inline suggestions for Copilot that increased complexity of the code. 

The only instance I fully rejected was an initial Claude code request to set up the project initially, where I did not include the modifier "Do not fill in any of the files beyond preliminary skeleton code and do not make any assumptions about the task." - it made a ton of assumptions about the nature of the project and wrote a bunch of irrelevant code. 
