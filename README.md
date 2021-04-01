# 2IOI0-group-16

### Instruction virtual/python environment

There are several prerequisites to gete everything working as intended. The tool could possibly function without these, but this makes setup easier.
An setup guide without VS Code is also included, but this is more technical
These prerequisites are:

- Anaconda3
- (Optional, but highly recommended) VS Code

### Mandatory for any setup:

For the tool to work correctly, many different python packages are required. To set this up, we will need Anaconda3 and the provided requirements.yml file in the main folder.
Start by opening Anaconda3 and head over to environments on the left side. On the bottom press the import button and use the folder icon and navigate to the requirements.yml file.
Give the environment a suiting name and click import, this could possibly take several minutes.
Continue with the section of either Installation with VS Code or the section on Installation without VS Code, as per your preference.

### Installation with VS Code

After setting up the environment go to the home tab on the left side and launch, or possibly install and then launch, VS Code.
The final step is to make sure that VS Code uses the newly created environment. To do this, click on the bottom left of VS Code where it says Python 3.VERSION.
This will open a menu in the middle of the screen with different Python environments. Select Python 3.6.13 64-bit ('YOUR ENVIRONMENT NAME': conda)".
Now everything is properly set up to work with the tool.

To run the tool, press the green run tool button on the top right. Continue with the section on using the tool.

### Installation without VS Code

In the left column, press the arrow button associated with the newly created environment and press "open terminal". Navigate to the folder containing the final_tool.py file
using the "cd .." and the "cd FOLDER_TO_ENTER" commands. Once in the folder containing the final_tool.py file run the following command "python final_tool.py". Continue with the section on using the tool.

### Using the tool

A GUI will show up.
The input fields that can be filled in, in recommended order are:

1. File names (including extensions) of train and test datasets, these files must be inside the data folder.
2. The name of your output csv file containing the predicted and actual result, this file will be outputted to the output folder.
3. Whether you want to make use of event or time prediction or both.
4. Select the algorithm you want to apply.
5. If LSTM is selected as the prediction algorithm, otherwise skip 5

   i. The max number of epoch you wish to train on can be provided.

   ii. (optional) Load a trained model to apply the test dataset.

   iii. (optional) Load a trained model and will be trained for the max number of epochs given in i.

6. Give the column names for the three identifiers as shown in the tool. The model will be training on these three features at any rate.
7. (optional) Give extra features that you wish the model to train on.

#### Play with different combinations of extra features, we challenge you to beat the accuracy yielded by only training on the core features!

The tool will possibly freeze shortly, while the program is training on the datasets.
To still have feedback on what is going on, the terminal in VS Code shows what the algorithm is working on.
When done training, this is also where the accuracy and/or RMSE will be shown for the chosen prediction algorithms.
When you see 'Finished processing request!!'. The tool will terminate and the output csv can be found in output/"name given in input field 2".

#### Enjoy using the tool!!
