# Decision-Tree
Decision tree learning algorithm

"experiment report.pdf" is experiment report,and you can regrad it as instruction.
"decisiontree.py" is python script.
"data.txt" is the Iris Dataset from the UCI Machine Learning Repository

Note:All the dataset file and python script need to be taken in the same subdirectory.If you want to run it, you should run it in python3.And modules "copy" and "numpy" are necessary.Otherwise,you could get back errors.

I didn't realize the display of data structure in my code.If you want to check the tree structure,please debug the code.You can follow the following step to check.I use PyCharm as my IDE.

1.Set breakpoint after function TreeGenerate()
![](https://github.com/Deep-Lan/Decision-Tree/blob/master/screenshots/1.png)
2.Debug
![](https://github.com/Deep-Lan/Decision-Tree/blob/master/screenshots/2.png)
3.You are free to check if you like
root_node is the root node of the binary tree._TreeNode__attribute is the attribute that sample is classified in.When the node is not leaf node,it makes sense._TreeNode__threshold is the threshold that sample is classified by.When the node is not leaf node,it makes sense._TreeNode__classtype is the classtype that node contain.When the node is leaf node,it makes sense._TreeNode__leftnode and _TreeNode__rightnode are the sub-node of it.
![](https://github.com/Deep-Lan/Decision-Tree/blob/master/screenshots/3.png)
