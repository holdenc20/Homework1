<a name="br1"></a> 

EECE 5644 Homework 1

Casey Holden

December 20, 2023

View my github page for my source code:

[hꢀps://github.com/holdenc20/EECE5644-Homework](https://github.com/holdenc20/EECE5644-Homework)

1



<a name="br2"></a> 

1\.

To start, I generated the 4-dimensional sample data using the mulꢁvariate Gaussian

probability density funcꢁons. The two Gaussians are labels L ∈ {0, 1} where P(L = 0) =

0\.35 and P(L = 0) = 0.65. Of the 10,000 samples generated, 3,500 of the samples

correspond to the Gaussian g(x|m , C ) where

0

0

−1

−1

−1

−1

2

−0.5

0\.3 −0.5

0

−0.5 0.3

0

0

0

2

1

−0.5

푚<sub>0</sub>

\=

,

퐶<sub>0</sub> =

1

0

0

And 6,500 of the samples correspond to g(x|m , C ) where

1

1

1

1

1

1

1

0\.3

−0.2 0.3

0

0\.3 −0.2

0\.3

0

0

0

3

2

푚<sub>1</sub>

\=

,

퐶<sub>1</sub> =

1

0

0

**1.A** ERM Classiﬁcaꢁon Using the Knowledge of True Data PDF

**1.A.1** Minimum Expected Risk Classiﬁcaꢁon Rule

The classiﬁcaꢁon rule we will be using is a likelihood raꢁo test.

p(x|L = 1)

p(x|L = 0)

g(x|m , C )

p(L = 0) λ<sub>01</sub>-λ<sub>00</sub>

0

0

\=

\> 훾 =

\*

g(x|m , C )

p(L = 1) λ<sub>10</sub>-λ<sub>11</sub>

1

1

For loss matrix where D is the decision label, the loss is 1 when L ≠ D and 0 when L = D.

Loss Matrix Values

λ<sub>ij</sub> where D = i | L = j

λ<sub>00</sub> = 1

λ<sub>01</sub> = 0

λ<sub>10</sub> = 0

λ<sub>11</sub> = 1

True Negaꢁve

False Negaꢁve

False Posiꢁve

True Posiꢁve

Figure 1. The Loss Matrix Costs

From Figure 1, we can see that 훾 should be as follows:

2



<a name="br3"></a> 

p(L = 0) λ<sub>01</sub>-λ<sub>00</sub>

0\.35 1 - 0

훾 =

\*

\=

\*

= 0.538

p(L = 1) λ<sub>10</sub>-λ<sub>11</sub>

0\.65 1 - 0

**1.A.2** Implementaꢁon of the ERM Classiﬁer

Using the genrated samples, we ran the classiﬁer on varrying threshold (훾) values. For

each of these 훾 values we calculated the expected predicꢁons for each of the samples. From

this set of decisions, we were able to compare them with the label to determine the true

positve, false posiꢁve, false negaꢁve, true negaꢁve rates. Using the false posiꢁve and false

negaꢁve rates, we were able to calculate the error of each of the sets of predicꢁons. From

the probabiliꢁes of false posiꢁves and the probabiliꢁes of true posiꢁves, we were able to

generate the folowing ROC curve.

**Commented [CH1]:** Add chart

Figure 2. ROC Curve of ERM Classiﬁcaꢁon

**1.A.3** Threshold with Minimum Probability of Error

From the errors, calculated using the following equaꢁon:

P(error; 훾) = P(D = 1|L = 0; 훾)P(L = 0) + P(D = 0|L = 1; 훾)P(L = 1)

3



<a name="br4"></a> 

We were able to ﬁnd the 훾 with the minimum error to be

훾<sub>experimental</sub> = 0.637

P(error; 훾<sub>experimental</sub>) = 0.045

Compared to the 훾<sub>theoretical</sub> = 0.539, the thresholds are preꢀy similar considering the

wide range of threshold values considered (e<sup>-30</sup>, e<sup>30</sup>). For this problem, I varried exponenꢁal

term form -30 to 30 for 1000 thresholds so there was a lot of values in the (0,1) range.

**1.B** ERM Classiﬁcaꢁon with Incorrect Knowledge of the Data Distribuꢁon

For this problem, I used the same sample data from the g(x|m , C ) and g(x|m , C )

0

0

1

1

distribuꢁons. But when classifying and the data, we used the diagonal matrix of C and C .

0

1

**1.B.1** Minimum Expected Risk Classiﬁcaꢁon Rule

The classiﬁcaꢁon rule for this quesꢁon is exactly the same as part 1.A.1 results.

**1.B.2** Implementaꢁon of the ERM Classiﬁer

We used the exact same implementaꢁon as the previous secꢁon but when classifying

the data we used the incorrect data distribuꢁon. Using that we generated the following ROC

curve.

4



<a name="br5"></a> 

Figure 3. ROC Curve of ERM Classiﬁcaꢁon with Incorrect Knowledge

**1.B.3** Threshold with Minimum Probability of Error

Using the same process as 1.A.3, we were able to ﬁnd the 훾 with the minimum error to

be

훾<sub>experimental</sub> = 0.472

P(error; 훾<sub>experimental</sub>) = 0.047

Although the 훾<sub>experimental</sub> is not that similar to the 훾<sub>experimental</sub> from part A, the error is

almost exactly the same which is surprising. So a minimal change did occurred that

increased the minimum probability of error. Although the ROC curve looks almost exactly

the same, because of the minimal change in the minimum probability of error, we can

assume that there was also a minimal change in the ROC curve.

**1.C** Fisher Linear Discriminact Analysis Based Classiﬁer

5



<a name="br6"></a> 

By calculaꢁng the mean and covarience of the sample data for both class labels, we can

determine the w<sub>LDA</sub> value for the LDA classiﬁer. To determine the mean and covarience of the

sample data, we esꢁmated using the built in sample mean and sample covarience funcꢁons.

From these esꢁmated mean and covarience values for each of the labels, we are able to

determine the w<sub>LDA</sub> value.

Within Class Scatter Matrix = SW = 0.5 \* C<sub>sample,0</sub> + 0.5\*C<sub>sample,1</sub>

<sup>m</sup><sub>avg</sub> = 0.5 \* m<sub>sample,0</sub> + 0.5\*m<sub>sample,1</sub>

Between Class Scatter Matrix = SB = 0.5 \* ||m<sub>sample,0</sub> - m<sub>avg</sub>||<sup>2</sup> + 0.5\* ||m<sub>sample,1</sub> - m<sub>avg</sub>||<sup>2</sup>

From that we can get the eigenvalues and eigenvectors of SW<sup>-1</sup> ∙ SB

w<sub>LDA</sub> is the normalized eigenvector of the maximum eigenvalue.

Using wLDA, we can classify the sample data with 휏 ∈ (-inf, inf).

x⃗ ∙ w<sub>LDA</sub> < 휏

Using the LDA classiﬁer, we generated the following ROC curve.

Figure 4. ROC Curve of LDA Classiﬁcaꢁon

6



<a name="br7"></a> 

The minimum error for the LDA Classiﬁcaꢁon was 0.045 for a threshold of 0.424. This

minimum error is less than the minimum error of the ERM Classiﬁcaꢁon with Incorrect

Knowledge but more than the minimum error of the ERM Classiﬁcaꢁon.

7



<a name="br8"></a> 

**2**

For this quesꢁon, we are going to be creaꢁng a sample 3-dimensional vectors that are

generated by 4 Gaussians. The ﬁrst Gaussian with p(L=1) = 0.3 will have a mean, and covariance

as follows:

-30

푚<sub>1</sub> = -12 ,

-10

25 -5

퐶<sub>1</sub> = -5 10 -5

-5 10

3

3

The second Gaussian with p(L=2) = 0.3:

12

푚<sub>2</sub> = 14 ,

24

15 -5

퐶<sub>2</sub> = -5 13 -5

-5 10

0

3

The third and fourth Gaussians with p(L=3) = 0.4 with equal weights:

45

푚<sub>3a</sub> = -34 ,

7

10

3

-2

3

20

퐶<sub>3a</sub> = 3 20

-2

3

3

1

푚<sub>3b</sub> = 35 ,

-35

10

-2

3

20

퐶<sub>3b</sub> = 3 20

-2

3

I scaled up the covarience matrices inorder so that there was a signiﬁcant amount of overlap.

**2.A** Minimum Probability of Error Classiﬁcaꢁon with MAP classiﬁer

The MAP classiﬁer used a loss matrix of:

0

Λ = 1

1

1

0

1

1

1

0

By ﬁnding the minimum probability of error using this 0-1 loss, we were able to classify the

sample data resulꢁng in the following confusion matrix.

8



<a name="br9"></a> 

Figure 5. Confusion Matrix of 0-1 MAP Classiﬁer

From this confusion matrix, we can see that each of the true predicꢁons have similarly high

predicꢁon accuracy. From this classiﬁcaꢁon there was about a 10% misclassiﬁcaꢁon rate so

there is a signiﬁcant amount of overlap as can be seen in Figure 5.

9



<a name="br10"></a> 

Figure 6. Scaꢀer Plot of the 0-1 MAP Classiﬁer

**2.B** ERM Classiﬁcaꢁon with Bias Towards Label 3

For this quesꢁon, we will be using loss matrices that care more about classifying the L = 3

samples correctly. To do this, we will use the following matrices.

0

Λ<sub>10</sub> = 1

1

10 10

0

1

10

0

0

Λ<sub>100</sub> = 1

1

100 100

0

1

100

0

10



<a name="br11"></a> 

Figure 7. Scaꢀer Plot of the ERM Classiﬁcaꢁon Using the Λ<sub>10</sub> loss matrix.

11



<a name="br12"></a> 

Figure 8. Confusion Matrix of the ERM Classiﬁcaꢁon using the loss matrix Λ<sub>10</sub>

As we can see in the confusion matrix from Λ<sub>10</sub>, we can see that most of the samples with L=3

are classiﬁed correctly. We can also see that Labels 1 and 2 are oꢂen incorrectly classiﬁed as L=3

which makes sense. Compared to Figure 5, we can see that the correct predicꢁons for labels 1

and 2 have decreased but the correct predicꢁons for label 3 has increased as expected. From

this loss matrix, about 18% of the samples are misclassiﬁed. This is an increase from the 10% in

part 1 which is expected when prioriꢁzing a class.

For the ERM classiﬁcaꢁon with the Λ<sub>100</sub> loss matrix, we can see similar changes as Λ<sub>10</sub> but to a

greater extent.

12



<a name="br13"></a> 

Figure 9. Confusion Matrix of the ERM Classiﬁcaꢁon using the loss matrix Λ<sub>100</sub>

As we can see in the confusion matrix from Λ<sub>100</sub>, we can see that almost all of the samples

with L=3 are classiﬁed correctly. We can also see that Labels 1 and 2 are incorrectly classiﬁed as

L=3 at a high percentage. Compared to Figure 5, we can see that the correct predicꢁons for

labels 1 and 2 have decreased but the correct predicꢁons for label 3 has increased as expected.

From this loss matrix, about 38% of the samples are misclassiﬁed. This is an increase from the

10% in part 1 which is expected when prioriꢁzing a class.

13



<a name="br14"></a> 

Figure 10. Confusion Matrix of the ERM Classiﬁcaꢁon using the loss matrix Λ<sub>100</sub>

14



<a name="br15"></a> 

**3**

For this quesꢁon, we will be analyzing the UCI Wine Quality dataset and the Human

Acꢁvity Recogniꢁon dataset. We will be calculaꢁng the Gaussian class condiꢁonal pdfs for each

of the labels. Using these class condiꢁonal pdfs, we are going to generate decisions using the

minimum-probability of error classiﬁer. For the loss matrix, we are going to be using the

following formula to generate the loss matrix:

αtrace(C<sub>SampleAverage</sub>

)

λ =

rank(C<sub>SampleAverage</sub>

)

From this loss matrix, we can calculate the label with the minimum probability of error for each

of the samples. From these decisions, we can calulate the percentage of the samples that are

correctly labeled.

3\.1 Wine Quality Dataset

The class priors for this dataset are as follows:

P(L = 3)

P(L = 4)

P(L = 5)

P(L = 6)

P(L = 7)

P(L = 8)

P(L = 9)

0\.0041

0\.0333

0\.2975

0\.4487

0\.1797

0\.0357

0\.0010

Figure 11. Class Priors of the Wine Quality Dataset

The sample mean and covariance of the dataset are calculated using the following formulas:

N

1

X̅ = ∑ X<sub>i</sub>

N

i=1

N

1

∑(X -X̅ )(X<sub>ik</sub>-X̅̅<sub>k</sub>̅)

ij

q<sub>jk</sub>

\=

j

N - 1

i=1

15



<a name="br16"></a> 

Aꢂer generaꢁng the loss matrix of:

0

8\.95

8\.95

0

8\.53

2\.51

0

1\.95

1\.62

1\.67

1\.31

8\.38

2\.36

1\.95

0

1\.48

1\.52

1\.17

8\.06

2\.04

1\.62

1\.48

0

8\.11

2\.09

1\.67

1\.52

1\.20

0

7\.75

1\.73

1\.31

1\.17

0\.84

0\.89

0

8\.53 2.51

8\.38 2.36

8\.06 2.04

8\.11 2.09

7\.75 1.73

1\.20

0\.84

0\.89

Figure 12. Loss Matrix for Wine Quality Dataset

Using the class priors, loss matrix, the sample mean, and sample covariance, we can now

calculate the minimum probability of error for each of the samples. The decisions will be label

with the minimum probability of error. From these decisions and correct labels, we can

determine the percentage of correctly classiﬁed samples, which I found to be 50.02%. We can

also ﬁnd the confusion matrix(Figure 13). From Figure 12 and Figure 13, we can see that a large

percentage of the class samples are labeled 5, 6, or 7.

Figure 13. Confusion Matrix of True Labels and the Predicted Labels

16



<a name="br17"></a> 

3\.2 Human Acꢁvity Dataset

For this dataset, we preꢀy much follow the exact same process as part 1. Aꢂer

generaꢁng the Gaussian distribuꢁons for each of the class labels and the loss matrix, we are

able to classify the data. I tried using the training data in the classiﬁer and I was geꢃng 100%

correct classiﬁcaꢁon rate. So instead I trained the model with the training dataset then

classiﬁed the tesꢁng dataset. From this tesꢁng dataset, I achieved a 94.2% classiﬁcaꢁon

accuracy which can be seen in the decision matrix in Figure 14.

Figure 14. Confusion Matrix for the Human Acꢁvity Dataset

3\.3 Discussion

As we can clearly see from the decision matrices for both datasets, the classiﬁer does a

much beꢀer job classifying the Human Acꢁvity Dataset than the Wine Quality Dataset. Gaussian

class condiꢁonal models assume that the data within each class is normally distributed. These

17



<a name="br18"></a> 

models are beꢀer suited for datasets that have certain characterisꢁcs. The characterisꢁc is

Conꢁnuous Data – data that follows a normal distribuꢁon.

For the Human Acꢁvity Dataset, all of the data was conꢁnuous – all the data was from -1

to 1. So the GMM was well suited for this data. For the Wine Quality dataset, the data was also

conꢁnuous. But I think the issue with the dataset is that because all of the data is concentrated

around a quality of 5,6,7 and the quality rankings of the wine is an ambinuous process in the

frist place, there is probably a lot of overlap in the decisions. Whereas the classes for the

Human Acꢁvity Dataset are very disꢁct opꢁons. You can see this in the confusion matrix where

there is a lot of overlap in the decisions for 5,6,7.

Appendix:

Figure 15. A Scaꢀer Plot of the Subset of Fixed Acidity, Volaꢁle Acidity, and Citric Acid.

18



<a name="br19"></a> 

Figure 16. A Scaꢀer Plot of the Subset of tBodyAcc-mean()-X, tBodyAcc-mean()-Y, and tBodyAcc-

mean()-Z

19

