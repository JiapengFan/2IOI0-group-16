# 2IOI0-group-16

### Instruction virtual/python environment

For the tool to work correctly, many different python packages are required. To set this up, we will need Anaconda3 and the provided requirements.yml file in the main folder.
Start by opening Anaconda3 and head over to environments on the left side. On the bottom press the import button and use the folder icon and navigate to the requirements.yml file.
Give the environment a suiting name and click import, this could possibly take several minutes.

After setting up the environment go to the home section in Anaconda3 and make sure the newly created environment is selected in the dropdown menu from the top.
When it is launch, or possibly install and then launch, VS Code. In VS Code open the final_tool.py file that is in the main folder.

To run the tool, press the green run tool button on the top right. Continue with the next section.

### Using the tool

A GUI will show up.
The input fields that can be filled in the recommended order are

1. Path to training and test data set from the root.
2. The name of your output csv file containing the predicted and actual result.
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
