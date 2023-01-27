## Evaluation Metrics on Confusion Matrix for Binary Classification

Confusion Matrix

![Confusion Matrix
](https://miro.medium.com/max/1400/1*fxiTNIgOyvAombPJx5KGeA.webp)

$Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$

$Precision = \frac{TP}{TP + FP}$

$Recall = \frac{TP}{TP + FN}$

$F1 = \frac{2}{\frac{1}{Precision} + \frac{1}{Recall}}$


## Evaluation Metrics on Confusion Matrix for Multi-Class Classification

Accuracy can be easily calculated. 

You can calculate Precision, Recall, and F1 for each label. 

Marco F1 score is the average F1 scores for all labels. 


When data of labels are balanced, accuracy is similar to F1. Otherwise, they are different. If it is unbalanced, people use F1 more. 