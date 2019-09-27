# PS-L2-AROM Project

The objective of the PS-L2-AROM project is to reproduce the experiments reported in [4] using the datasets described there ( **Table 1** ). The results and our implementation of the code are found in this repository. A comprehensive step by step guide is found in the .ipynb file. 

**Table 1.**
*Platform and number of features per dataset [4]*


![data_tab.png](https://github.com/Dfupa/PS-L2-AROM/blob/master/images/data_tab.png)

## State-of-the-art: a brief summary regarding feature selection methods
One of the main problems when dealing with biological information is the curse of dimensionality that in combination with the relatively low number of samples makes the task of feature selection (FS) crucial in prediction models development. FS can be applied both for supervised and unsupervised learning, but it is the former the one which has experienced a higher development in the recent years. The goal for FS methods is principally to return a subset of features that sufficiently represent the complexity within the dataset.

**Table 1.**
*A taxonomy of feature selection techniques and some of its characteristics and highlights. [1]*

![table.png](https://github.com/Dfupa/PS-L2-AROM/blob/master/images/table.png)

Feature selection techniques are critically used:

* To **avoid the overfitting** of the model and thus **improve its performance**

* To provide **faster and less computationally expensive models**.

* To provide a **better understanding of the underlying processes** that generated the data

However, this advantages come at a certain cost as it is now required instead of just optimize the model's parameter; to find the optimum model's parameters for an optimum features subset. This provides a new dimension in the model hypothesis space that adds a new layer of complexity to be tackled head-on.

### Feature selection taxonomy
Feature selection techniques differ and are classified from each other in the way they incorporate this search of feature subsets in the model selection 

* **Filter techniques**:  this set of techniques is characterized by the selection of relevant features based only in the data, without paying attention to the ulterior classifier to be used. These are simple methods, easily scalable, fast, and once performed the FS, different classifiers can be used. However, the independence with the classifier can be a problem as the subset of features selected can work quite well with a certain model but worse with others. Filter methods include univariate (independent features) and multivariate (feature dependencies) methods.

    
    
* **Wrapper techniques**: these techniques consider a specific classifier, and make a feature subset selection trying to optimize the result with that given classifier. For subset exploration both deterministic and randomized search algorithms can be used. These methods, alike multivariate filter techniques, have the ability of considering feature dependencies. The interaction with the classifier constitutes also an advantage. The main drawback is the high computational requirements for the exploration of possible feature subsets.


* **Embedded techniques**: in these methods the FS is part of the model construction. One clear example is the LASSO method that penalizes regression coefficients that are less useful for fitting a linear model using the L1-norm. Those features shrinked to 0 are discarded. These methods have the advantage of the interaction with a certain model, while they are much less computationally demanding than wrapper methods.
        
### Feature selection applications and implementations in bioinformatics
FS has applications in several biological problems, like sequence analysis, microarray and mass
spectra analysis

1) **Sequence analysis**: In the context of FS two types of problems tend to be described: content analysis and signal analysis. 


     - *Content analysis*: it focuses on the prediction of sequences coding for proteins, that use different Markov model derived techniques, and also tries to predict protein function from sequence, using combinations of techniques like genetic algorithms and the Gamma test for scoring.


     - *Signal analysis*: it focuses on finding discrete motifs: binding sites, translation initiation sites or splice sites. Several algorithms are used in this domain, like regression approaches and support vector machines (SVM).
 
2) **Microarray analysis**: FS has become crucial in microarray analysis, provided the large number of dimensions (tens of thousands of genes) and the lack of samples. Two paradigms: univariate and multivariate filters:

     - *Univariate filters*: probabily the most used due to their efficiency and simplicity. Can be divided into parametric and model-free methods. Although parametric approaches, with Gaussian assumptions are very frequent, non-parametric methods are usually preferred due to the typical uncertainty regarding data distributions. Usually the whole set of samples is considered for the identification of differentially expressed genes but there are also methods able to capture gene alterations within small subsets of samples, which could be determinant for patient-specific level diagnostics.
  
     - *Multivariate filters*: recently are gaining relevance as they consider gene-gene interactions. Applications for filtering, wrapper and embedded techniques have been developed. In the case of wrapper methods, the scoring function has special relevance. The most used include the ROC curve and the optimization of LASSO function.
  
  
3) **Mass spectra analysis**: Similar techniques are being applied to the mass spectra analysis field, where the initial number of features can reach 100000. Combinations of feature extraction and feature selection methods are implemented to reduce the number of features to a range of 300-500.


As it has been broadly discussed above, biological data adolesce from small sample sizes. This can lead to overfitting and imprecision. To control for these problems is important in the first place to have an adequate evaluation criteria (for example: not using training data for evaluation). Another method to perform better approximations consists in using ensemble methods, that combine several FS approaches. This latter methods have proved to perform better than previous ones, even though they require additional computational resources. Finally, FS methods are gaining representation in upcoming fields, like single nucleotide polymorphism (SNP) analysis and biomedical text mining. In the first case, FS techniques try to select subsets of SNPs from which the rest can be derived. Text mining has relevant applications in the biological field for hypothesis discovery, extraction of medical parameters of interest, etc.
In conclusion the problem that high-dimensionality/sample size ratio represents in biodata analysis imposes the need for efficient FS methods that let us build accurate models and avoid overfitting.

In this project we are going to work with L2-AROM and PS-L2-AROM. L2-AROM approximately minimizes the "L0-norm" by iteratively solving several optimization problems regularized by the L2-norm. As this method is based around training several times a Support Vector Machine (SVM) over the re-scaled versions of the training set, it can be concluded that it is an **embeded feature selection method**[3]. Provided PS-L2 algorithm is based on L2-AROM, it is also an embeded FS technique.

## The PS-L2-AROM 
To explain L2-AROM we need to briefly explain what AROM (Approximation of the zero-norm Minimization) is: AROM is used to find a hyper-plane w separating the data in two classes in such a way that many of the components of *w* are zero[2]. That is, if a particular component of *w*, *wj* , is zero, the attribute values corresponding to that component are irrelevant for the decision process. This is equivalent to discarding features for prediction. In L2-AROM this approximation of the zero-norm minimization is achieved through the L2-norm (*Image 1*).


![l2.png](https://github.com/Dfupa/PS-L2-AROM/blob/master/images/l2.png)
<span style="font-size:60%">*Image 1*: A graphical representation of the L2-norm, also known as Tikhonov regularization. It pushes the entries toward zero but without enforcing some of them to be identically zero; as opposed to the L1 norm (which it does, enforcing sparsity).</span>



L2-AROM final algorithm implementation boils down to a linear SVM estimation with iterative rescaling of the inputs. A smooth feature selection occurs during this iterative process since the weight coefficients along some dimensions progressively drop below a provided threshold while other dimensions become more significant[3]. A final ranking on the absolute values of each dimension can be used to obtain a fixed number of features. Finally, it returns a vector with the selected features indices.


## References 
[1] Yvan Saeys, Iñaki Inza, and Pedro Larrañaga. A review of feature selection techniques in bioinformatics. Bioinformatics, 23:2507–2517, 2007.   
[2] Jason Weston, André Elisseeff, Bernhard Schölkopf, and Mike Tipping. Use of the zero norm with linear models and kernel methods. Journal of Machine Learning Research, 3:1439–1461,2003.   
[3] T. Helleputte and P. Dupont. Partially supervised feature selection with regularized linear models. In Proceedings of the 26th Annual International Conference on Machine Learning, 2009.   
[4] Thibault Helleputte and Pierre Dupont. Feature selection by transfer learning with linear regularized models. In European Conference on Machine Learning and Knowledge Discovery in Databases, pages 533–547, 2009.
