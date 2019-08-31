# CSE474/574: Introduction to Machine Learning(Fall 2018)

## Instructor: Sargur N. Srihari

```
************* September 6, 2018****************
```
## Project 2: Learning to Rank using Linear Regression

## Due Date: Monday, Oct 8, Report: Wednesday, Oct. 10

## 1 Overview

The goal of this project is to use machine learning to solve a problem that arises in Information Retrieval,
one known as the Learning to Rank (LeToR) problem. We formulate this as a problem of linear regression
where we map an input vectorxto a real-valued scalar targety(x,w).
There are two tasks:

1. Train a linear regression model on LeToR dataset using a closed-form solution.
2. Train a linear regression model on the LeToR dataset using stochastic gradient descent (SGD).

The LeToR training data consists of pairs of input valuesxand target valuest. The input values are
real-valued vectors (features derived from a query-document pair). The target values are scalars (relevance
labels) that take one of three values 0, 1, 2: the larger the relevance label, the better is the match between
query and document. Although the training target values are discrete we use linear regression to obtain real
values which is more useful for ranking (avoids collision into only three possible values).

## 1.1 Linear Regression Model

Our linear regression functiony(x,w)has the form:

```
y(x,w) =w>φ(x) (1)
```
wherew= (w 0 ,w 1 ,..,wM− 1 )is a weight vector to be learnt from training samples andφ= (φ 0 ,..,φM− 1 )>
is a vector ofMbasis functions. Assumingφ 0 (x)≡ 1 for whatever input,w 0 becomes the bias term. Each
basis functionφj(x)converts the input vectorxinto a scalar value. In this project, you are required to use
the Gaussian radial basis functions

```
φj(x) = exp
```
#### (

#### −

#### 1

#### 2

```
(x−μj)>Σ−j^1 (x−μj)
```
#### )

#### (2)

whereμjis the center of the basis function andΣjdecides how broadly the basis function spreads.

## 1.2 Noise model and Objective function

We use a probabilistic model in which the output target value is subject to noise. More specifically, we
assume that the output has a normal distribution, with meany(x,w)and precisionβ. With input samples


X={x 1 ,..,xn}and target valuest={t 1 ,..tn}, the likelihood function has the form:

```
p(t|X,w,β) =
```
#### ∏N

```
n=
```
#### N

#### (

```
tn|w>φ(xn),β−^1
```
#### )

#### (3)

Maximizing (3) is equivalent to minimizing the sum-of-squares error

```
ED(w) =
```
#### 1

#### 2

#### ∑N

```
n=
```
```
{tn−w>φ(xn)}^2 (4)
```
1.2.1 Regularization to contain over-fitting

To obtain better generalization and avoid overfitting, we add a regularization term to the error function,
with the form:
E(w) =ED(w) +λEW(w) (5)

where theweight decayregularizer is

```
EW(w) =
```
#### 1

#### 2

```
wTw (6)
```
The coefficientλin (5) governs the relative importance of the regularization term.
The goal is to findw∗that minimizes (5). This can be done by taking the derivative of (5) with respect
tow, setting it equal to zero, and solving forwWe consider two linear regression solutions: closed-form
and stochastic gradient descent (SGD).

### 1.3 Closed-form Solution forw

The closed-form solution of (4), i.e., sum-of-squares errorwithoutregularization, has the form

```
wML= (Φ>Φ)−^1 Φ>t (7)
```
wheret={t 1 ,..,tN}is the vector of outputs in the training data andΦis the design matrix:

#### Φ=

#### 

#### 

#### 

#### 

#### 

```
φ 0 (x 1 ) φ 1 (x 1 ) φ 2 (x 1 ) ··· φM− 1 (x 1 )
φ 0 (x 2 ) φ 1 (x 2 ) φ 2 (x 2 ) ··· φM− 1 (x 2 )
..
.
```
#### ..

#### .

#### ..

#### .

#### ..

#### .

#### ..

#### .

```
φ 0 (xN) φ 1 (xN) φ 2 (xN) ··· φM− 1 (xN)
```
#### 

#### 

#### 

#### 

#### 

The quantity (7) is known as the Moore-Penrose pseudo-inverse of the matrixΦ.
The closed-form solution with least-squared regularization, as defined by (5) and (6) is

```
w∗= (λI+Φ>Φ)−^1 Φ>t (8)
```
### 1.4 Stochastic Gradient Descent Solution forw

The stochastic gradient descent algorithm first takes a random initial valuew(0). Then it updates the value
ofwusing
w(τ+1)=w(τ)+ ∆w(τ) (9)

where∆w(τ)=−η(τ)∇Eis called the weight updates. It goes along the opposite direction of the gradient
of the error.η(τ)is the learning rate, deciding how big each update step would be. Because of the linearity
of differentiation, we have
∇E=∇ED+λ∇EW (10)

in which
∇ED=−(tn−w(τ)>φ(xn))φ(xn) (11)
∇EW=w(τ) (12)


### 1.5 Evaluation

Evaluate your solution on a test set using Root Mean Square (RMS) error, defined as

```
ERMS=
```
#### √

```
2 E(w∗)/NV (13)
```
wherew∗is the solution andNVis the size of the test dataset.

### 1.6 Dataset

You are required to implement linear regression on a learning to rank (LeToR) dataset. In the LeToR
dataset the input vector is derived from a query-URL pair and the target value is human value assignment
about how well the URL corresponds to the query.
The Microsoft LETOR 4.0 Dataset is a benchmark data set for research released by Microsoft Research
Asia. It can be found at

```
http://research.microsoft.com/en-us/um/beijing/projects/letor/letor4dataset.aspx
```
It contains 8 datasets for four ranking settings derived from the two query sets and the Gov2 web page
collection. For this project, downloadMQ2007. There are three versions for each dataset: “NULL”, “MIN”
and “QueryLevelNorm”. In this project, only the “QueryLevelNorm” version “Querylevelnorm.txt” will be
used. The entire dataset consists of 69623 query-document pairs(rows), each having 46 features. Here are
two sample rows from the MQ2008 dataset.

```
2 qid:10032 1:0.056537 2:0.000000 3:0.666667 4:1.000000 ... 46:0.
```
```
#docid = GX029-35-5894638 inc = 0.0119881192468859 prob = 0.
```
```
0 qid:10032 1:0.279152 2:0.000000 3:0.000000 4:0.000000 ... 46:1.
```
```
#docid = GX030-77-6315042 inc = 1 prob = 0.
```
The meaning of each column are as follows.

1. The first column is the relevance label of the row. It takes one of the discrete values 0, 1 or 2. The
    larger the relevance label, the better is the match between query and document. Note that objective
    outputyof our linear regression will give a continuous value rather than a discrete one– so as to give
    a fine-grained distinction.
2. The second columnqidis the query id. It is only useful for indexing the dataset and not used in
    regression.
3. The following 46 columns are the features. They are the 46-dimensional input vectorxfor our linear
    regression model. All the features are normalized to fall in the interval of[0,1].
4. We would NOT use itemdocid(which is the ID of the document),inc, andprobin this project. So
    just ignore them.

## 2 Plan of Work

### 2.1 Tasks On The LeToR Dataset

```
1.Extract feature values and labels from the data: Process the original text data file into a Numpy
matrix that contains the feature vectors and a Numpy vector that contains the labels.
```
```
2.Data Partition: Partition the data into a training set, a validation set and a testing set. The training
set takes around 80% of the total. The validation set takes about 10%. The testing set takes the rest.
The three sets should NOT overlap.
```

```
3.Train model parameter: For a given group of hyper-parameters such asM,μj,Σj,λ,η(τ), train
the model parameterwon the training set.
```
```
4.Tune hyper-parameters: Validate the regression performance of your model on the validation set.
Change your hyper-parameters and repeat step 3. Try to find what values those hyper-parameters
should take so as to give better performance on the validation set.
```
```
5.Test your machine learning scheme on the testing set: After finishing all the above steps, fix
your hyper-parameters and model parameter and test your model’s performance on the testing set.
This shows the ultimate effectiveness of your model’s generalization power gained by learning.
```
## 3 Strategies for Tuning Hyper-Parameters

### 3.1 Choosing Number of Basis FunctionsM and Regularization Termλ

For tuningMandλ, you can simply use grid search. Starting from small values ofMandλ, gradually try
bigger values, until the performance does not improve. Please refer to

```
https://en.wikipedia.org/wiki/Hyperparameter_optimization#Grid_search.
```
### 3.2 Choosing Basis Functions

```
1.Centers for Gaussian radial basis functionsμj: A simple way is to randomly pick upMdata
points as the centers.
```
```
2.Spread for Gaussian radial basis functionsΣj: A first try would be to use uniform spread for
all basis functionsΣj= Σ. Also constrainΣto be a diagonal matrix
```
#### Σ =

#### 

#### 

#### 

#### 

#### 

```
σ 12
σ^22
..
.
σ^2 D
```
#### 

#### 

#### 

#### 

#### 

#### (14)

```
Chooseσ^2 i to be proportional to theith dimension variance of the training data. For example, let
σi^2 = 101 vari(x).
```
```
3.k-means clustering: A more advanced method for choosing basis functions is to first usek-means
clustering to partition the observations intoMclusters. Then fit each cluster with a Gaussian radial
basis function. After that use these basis functions for linear regression.
```
### 3.3 Choosing Learning Rateη(τ)

The simplest way would be to use fixed learning rateη. But this would lead to very poor performance.
Choosing too big a learning rate could lead to divergence and choosing too small a learnng rate could lead to
intolerably slow convergence. A more advanced method is to use Learning Rate Adaption based on changing
of performance. Please refer to

```
https://en.wikipedia.org/wiki/Gradient_descent.
```
## 4 Deliverables

There are two parts in your submission:


1. Report
    The report describes your implementations and results using graphs, tables, etc. Write a concise project
    report, which includes a description of how you implement the models and tune the parameters. Your
    report should be edited in PDF format. Additional grading considerations will include the performance
    of the training, creativity in paramter tuning and the clarity and flow of your report. Highlight the
    innovative parts and do not include what is already in the project description. You should also include
    the printed out results from your code in your report.
    Submission:
    Submit the PDF on a CSE student server with the following script:
    submit_cse474 proj2.pdffor undergraduates
    submit_cse574 proj2.pdffor graduates
    In addition to the PDF version of the report, you also need to hand in the hard copy version on the
    first class after due date or else your project will not be graded.
2. Code
    The code for your implementations. Code in Python is the only accepted one for this project. You
    can submit multiple files, but the name of the entrance file should be main.py. All Python code files
    should be packed in a ZIP file namedproj2code.zip. After extracting the ZIP file and executing
    commandpython main.pyin the first level directory, it should be able to generate all the results and
    plots you used in your report and print them out in a clear manner.
    Submission:
    Submit the Python code on a CSE student server with the following script:
    submit_cse474 proj2code.zipfor undergraduates
    submit_cse574 proj2code.zipfor graduates

## 5 Due Date and Time

The due date is11:59PM, Oct 8. Hard copy of your project report must be handed in before the end of
the first class after the due date. After finishing the project, you may be asked to demonstrate it to the TAs
if your results and reasoning in your report are not clear enough.


