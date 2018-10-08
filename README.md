# A metal parts classifier application using Multi-Layer Perceptron and Decision Tree Algorithm.

### Members:

1. Aravindh Kuppusamy (axk8776@rit.edu)
2. Karan Jariwala (kkj1811@rit.edu)
3. Deepak Sharma (ds5930@rit.edu)

------

### Problem Statement:

You own a start-up company that sorts and then sells used metal parts. As one of your first contracts, RIT has asked you to do this for parts taken from an old printing press. You need to sort the parts into four categories: 1. bolts, 2. nuts, 3. rings, and 4. scrap. After price negotiations with the school and some research of your own, you produce the table below defining the profit or cost in cents ($0.01) obtained after sorting each part, based on the actual type of the part, and the class assigned automatically to the part by your system in development, SORT-R.

![](https://github.com/AravindhKuppusamy/MLP-and-DT/blob/master/problem.PNG)

The task is to use machine learning algorithms to create a classifier that accurately identifies the type of each part as it arrives on the conveyor belt.

**Input Data:**
CSV format training and testing datasets are provided. Both files contain one sample per line, with data as follows: 

| Rotational symmetry | Eccentricity | class (1-bolt, 2-nut, 3-ring,4-scrap) |
| ------------------- | ------------ | ------------------------------------- |



------

### Classifier 1: Multi-Layer Perceptron (MLP)

Created a program to train and execute MLPs with one hidden layer with weights initialized randomly using values in the interval [-1,1]. Used the sigmoid (logistic) function as the activation function, and a learning rate of 0.1. Train MLP file takes a training data as an input and produce the final weight output using batch gradient descent with different number of epochs such as 0, 10, 100, 1000, 10000 etc. and plotted a graph containing total sum of squared error (SSE) over all training samples after each epoch (i.e. one complete pass over all training samples).

executeMLP.py takes a file defining a trained network and a data file, and runs the network on the data file. The program produce Recognition rate (% correct), Profit obtained, Confusion Matrix, and An image file containing a plot of the test samples (using shapes/color to represent classes) along with the classification regions



------

### Execution of program:

**Main file name:** trainMLP.py, executeMLP.py

**Python version:** 3.5.1

**Required packages:** matplotlib, numpy

***To train the network, run the trainMLP.py file***

```powershell
python3 trainMLP
```

- It generates the SSE-vs-Epochs graph and writes the weights of the network in a csv file

  ![SSE vs Epochs](https://github.com/AravindhKuppusamy/MLP-and-DT/blob/master/output/mlp/SSE_Vs_Epochs.png)

***To test the data on the trained network, run the executeMLP.py file***

```powershell
python3 executeMLP <Weight_file>
E.g: python executeMLP.py weights_1000.csv
```

- It generates the decision boundary graph and prints the accuracy, profit, and confusion matrix for test data in the console

  ![Decision Boundary](https://github.com/AravindhKuppusamy/MLP-and-DT/blob/master/output/mlp/desicion_boundary.png)

```
*********************************************Accuracy Testing*****************************************************

Accuracy: correct_prediction/total:  1.0
Mean Per Class Accuracy:  1.0

******************************************************************************************************************

************************************* Confusion Matrix for Test data  *******************************************
Pre/Act *	 Class 1 	|	 Class 2 	|	 Class 3 	|	 Class 4 	*	total
Class 1 *	 5     		|	 0     		|	 0     		|	 0     		*	 5    
Class 2 *	 0     		|	 6     		|	 0     		|	 0     		*	 6    
Class 3 *	 0     		|	 0     		|	 5     		|	 0     		*	 5    
Class 4 *	 0     		|	 0     		|	 0     		|	 4     		*	 4    
*******************************************************************************************************************
Total  *	 5     		|	 6     		|	 5     		|	 4     		|	


*******************************************************Profit******************************************************

Total Profit:  203
*******************************************************************************************************************
```
------

### Classifier 2: Decision Tree (DT)

Implemented a modified form of decision tree that allows continuous features to be partitioned using binary splits (similar to what is used in the C4.5 algorithm). 

To find the best split for a continuous feature: 

1. Sort samples in increasing order by their values for one attribute
2. Compute the information gain at every possible split point between adjacent samples, using their midpoint as the split value ‘s’. 

Samples with an attribute value greater than equal to ‘s’ are then put in the ‘right’ child of the current node, and otherwise the left child of the current node. Unlike with discrete attributes, we will need to re-use features. Also, applied Chi-Squared pruning to the learned tree, to avoid overfitting. For the Chi-Squared test, we tried with a significance level of 5% or 1%.

Train Decision tree file takes the training data and Two files representing the trained decision tree 1) after automatic induction, and 2) after subsequent pruning using the Chi-Squared test. And two images illustrating splits in the two decision trees by drawing a box around both regions created at every internal (‘split’) node in the tree. On the terminal, printed the number of nodes, number of leaf nodes, maximum, minimum and average depth of root-to-leaf paths in the decision tree.

------

### Execution of program:

**Main file name:** trainDT.py, executeDT.py

**Python version:** 3.5.1

**Required packages:** matplotlib, numpy

***To train the tree, run the trainDT.py file***

```powershell
python3 trainDT
```

- It generates a decision boundary graph for trained data and writes the tree in DTree.csv and pruned tree in PDTree.csv. 
- It also prints the number of nodes, number of leaf (decision) nodes, maximum, minimum and average depth of root-to-leaf paths.

  1. **Train data Decision Boundary without pruning**

  ![Trained data without Pruning](https://github.com/AravindhKuppusamy/MLP-and-DT/blob/master/output/decision_tree/train_decision_boundary_unpruned.png)

  2. **Train data Decision Boundary with pruning**

  ![Trained data with Pruning](https://github.com/AravindhKuppusamy/MLP-and-DT/blob/master/output/decision_tree/train_decision_boundary_pruned.png)

```powershell
---------------------------- DECISION TREE ---------------------------

 [0, 0, 0, 0] (2, 0.25143) CLASS: None
     [0, 22, 22, 2] (1, 0.29935) CLASS: None
         [0, 0, 22, 1] (2, 0.10739) CLASS: None
             [0, 0, 14, 0] Base case CLASS: 3
             [0, 0, 8, 1] (2, 0.108765) CLASS: None
                 [0, 0, 0, 1] Base case CLASS: 4
                 [0, 0, 8, 0] Base case CLASS: 3
         [0, 22, 0, 1] (2, 0.099544) CLASS: None
             [0, 17, 0, 0] Base case CLASS: 2
             [0, 5, 0, 1] (1, 0.431805) CLASS: None
                 [0, 0, 0, 1] Base case CLASS: 4
                 [0, 5, 0, 0] Base case CLASS: 2
     [15, 0, 0, 13] (2, 0.7854) CLASS: None
         [1, 0, 0, 13] (2, 0.7355700000000001) CLASS: None
             [0, 0, 0, 11] Base case CLASS: 4
             [1, 0, 0, 2] (1, 0.32968) CLASS: None
                 [1, 0, 0, 0] Base case CLASS: 1
                 [0, 0, 0, 2] Base case CLASS: 4
         [14, 0, 0, 0] Base case CLASS: 1

------------------------------ SUMMARY -----------------------------
No. of Nodes     :  19
No. of Leaf Nodes:  10
Maximum Depth    :  4
Minimum Depth    :  2
Average Depth    :  3.5
----------------------------------------------------------------------


------------------------ PRUNED DECISION TREE ------------------------

 [0, 0, 0, 0] (2, 0.25143) CLASS: None
     [0, 22, 22, 2] (1, 0.29935) CLASS: None
         [0, 0, 22, 1] (2, 0.10739) CLASS: 3
         [0, 22, 0, 1] (2, 0.099544) CLASS: 2
     [15, 0, 0, 13] (2, 0.7854) CLASS: None
         [1, 0, 0, 13] (2, 0.7355700000000001) CLASS: 4
         [14, 0, 0, 0] Base case CLASS: 1

------------------------------ SUMMARY -----------------------------
No. of Nodes     :  7
No. of Leaf Nodes:  4
Maximum Depth    :  2
Minimum Depth    :  2
Average Depth    :  2.0
----------------------------------------------------------------------

------------------------------- END ----------------------------------
```

***To test the data on the trained network, run the executeDT.py file***

```powershell
python3 executeDT <decision tree csv file>
E.g: python3 executeDT.py dt.csv
```

- It generates a decision boundary graph for test data( with and without prune )
- It prints the accuracy, profit, and confusion matrix for test data ( with and without prune )

   ![Test data with Pruning](https://github.com/AravindhKuppusamy/MLP-and-DT/blob/master/output/decision_tree/test_decision_boundary_pruned.png)

   ```powershell
   ********************************* DECISION TREE *************************************

    ('2', '0.25143') CLASS: None
        ('1', '0.29935') CLASS: None
            Base case CLASS: 3
            Base case CLASS: 2
        ('2', '0.7854') CLASS: None
            Base case CLASS: 4
            Base case CLASS: 1

   *************************************************************************************

   ************************************* TESTING ***************************************


   --------------------------------- Recognition Rate ----------------------------------
   Total Accuracy           :  0.95
   Mean Per Class Accuracy  :  0.9375
   -------------------------------------------------------------------------------------


   --------------------------------- Confusion Matrix ----------------------------------
   Pre/Act  *	 Class 1  	|	 Class 2  	|	 Class 3  	|	 Class 4  	*	Total
   Class 1  *	 5     		|	 0     		|	 0     		|	 1     		*	 6    
   Class 2  *	 0     		|	 6     		|	 0     		|	 0     		*	 6    
   Class 3  *	 0     		|	 0     		|	 5     		|	 0     		*	 5    
   Class 4  *	 0     		|	 0     		|	 0     		|	 3     		*	 3    
   -------------------------------------------------------------------------------------
   Total    *	 5     		|	 6     		|	 5     		|	 4     		*	 20   
   -------------------------------------------------------------------------------------


   -------------------------------------- Profit ---------------------------------------
   Total Profit     :  199
   -------------------------------------------------------------------------------------


   *************************************** END *****************************************

   ```

------

### Report:

To read more about this project, please refer to the project [report](https://github.com/AravindhKuppusamy/MLP-and-DT/blob/master/Project_Report.pdf).
