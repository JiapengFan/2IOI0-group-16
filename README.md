# 2IOI0-group-16

### Instruction virtual/python environment
Set up your own environment in the desired directory using: *py -m venv .venv*

Install Python environment: *pip install -r requirements.txt*.

### Running the tool
Run final_tool.py.

A GUI will show up. 
The input fields that can be filled in the recommended order are
  1. Path to training and test data set from the root.
  2. The name of your output csv file containing the predicted and actual result.
  3. Whether you want to make use of event or time prediction or both.
  4. Select the algorithm you want to apply.
  (5.)If LSTM is selected as the prediction algorithm, 
      i. The max number of epoch you wish to train on can be provided.
      ii. (optional) Load a trained model to apply the test dataset.
      iii. (optional) Load a trained model and will be trained for the max number of epochs given in i.
  6. Give the column names for the three identifiers as shown in the tool. The model will be training on these three features at any rate.
  7. (optional) Give extra features that you wish the model to train on.

#### Play with different combinations of extra features, we challenge you to beat the accuracy yielded by only training on the core features!

The tool will then print progress messages in the terminal.
When you see 'Finished processing request!!'. The tool will terminate and the output csv can be found in output/"name given in input field 2"

#### Enjoy using the tool!!
