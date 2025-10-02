## Welcome! for those who are interested:

This repository is my personal archive documenting the projects I did in the past attempting to apply different machine learning and data science methods to deal with different domain-specific problems, for the sake of <br>
- demonstrating skills for achieving machine learning modelling projects in academics and industry R&D
- simulating similar real-world projects encountered in past work experiences
- applying machine learning research in a niche topic / field

Most of the codes were written in Python, while a few on R, the major machine learning framework adopted included Tensorflow, Keras and Pytorch. 

## Content Preview:

Refer to below for a brief summary on the algorithms used in different projects within the directories:

### Computer Vision

- Agricultural crop (paddy) disease classification
    - Image Classfication on 9 classes of paddy diseases and 1 class of normal condition.
    - Vision Transformer (ViT),  Convolution-based Vision Transformer (Cvt), Pooling-based Vision Transformer (PiT)
    - Inception and ResNet block modules

- Manufactured steel defect detection
    - Object Detection and Semantic Segmentation, to locate the positions of defect pixels and classify them into one out of 4 classes of defect types on steel products.
    - U-Net with Inception and ResNet block modules and grouped convolutions
    - Wasserstein GAN with Gradient Penalty (WGAN-GP)

### Geospatial-Environmental Data Science

- Analysis of earthquakes
    - Time Series Forecasting for the earthquake related variables (magnitudes, depths, time until next occurrence, location or tectonic plates of next occurrence) based on the sequence of occurences of major large earthquakes.
    - Spatial Clustering with H-DBSCAN and hoversine distances
    - Hidden Markov Chain (HMM), Conditional Random Field (CRF), Bayesian Network (BN)
    - Deep Belief Network (DBN), Restricted Boltzmann Machine (RBM)
    - LSTM, GRU

- Typhoon clustering and predictions
    - Clustering typhoons in terms of their similarities during development, peak, and decline; using satellite image time series from typhoon monitoring observational records to perform nowcasting on the intensity, wind speed, air pressure, and trajectory in coming hours.
    - Time Series Clustering (Times Series K-Means, Global Alignment Kernel, DTW distances)
    - hybrid model of CNN-LSTM

### Healthcare Research

- US Work Injury Claim DAta Survival Analysis
    - modelling the time to event ("Accident / Injury Date" to "Case Close Date")
    - Kaplan-Meier estimator and Log-Rank Test for the difference in case close time between groups
    - assessing demographic and the nature of the occupational injury effects with Cox PH regression, Survival SVM, Survival Forest, Survival Gradient Boosting

- UPDRS Parkinson's Disease progression prediction
    - hybrid models combining recurrent, convolution and transformer components
    - forecasting individually for changes of the 3 dimensions up to 3 time steps (6 months, 12 months, 24 months)

### Natural Language Processing

- Bilingual product label classfication
    - Text Classification task on over 20 product categories in a social enterprise based on their product labels.
    - BERT Transformer Embeddings
    - Bi-directional LSTM (BiLSTM) Encoder

- Clincial note feature recognition
    - Named Entity Recognition (NER) task to extract words and phrases with implications on critical patient characteristics.
    - SpaCy RoBERTa based noun chunk and verb phrase matching
    - Char2Vec, FastText, SentenceBERT

- Topic Modelling and Sentiment Analysis on Amazon fine food review dataset
    - Latent Dirichlet Allocation (LDA), Non-negative Matrix Factorization (NMF)
    - Logistic Regression, Random Forest, Support Vector Machine (SVM)

### Sales and E-commerce Analytics

- Pet adoption conversion prediction
    - a regression problem to predict the 4 categorized time intervals of pet adoption since the pet registration in a website PetFinder.com (Kaggle competition)
    - Convolutional Autoencoder
    - Gradient Boosting Tree (XGB, LightGBM)
    - imbalanced class learning with SMOTE

- Recommender System on Amazon fine food products
    - Feature Engineering
        - Association Rule (Apriori, FP-Growth)
        - Biclustering
        - Latent Embeddings obtained from factorization methods
            - Multiple Correspondence Analysis
            - Weighted Alternating Least Square (WALS)
            - Stochastic Gradient Descent (SGD)
            - Non-negative Matrix Factorization (NMF)
        - Doc2Vec (from fine food reviews)
    - Modelling
        - Learning-to-Rank (Point-wise) by Feed-forward Neural Network
        - CatBoost optimized for normalized discounted cumulative gain (NDCG)

### Supply Chain and Logistics Analytics

- Airport passenger time series prediction
    - Demand forecasting with Seasonal ARIMA models

- Flight delay and cancellation prediction
    - a regression problem to predict the time required to depart from the gate until setting off from source airport; and to land on the destination airport
    - Generalized Linear Mixed-effect model (GLMM), Genelized Additive Model (GAM), Robust Regression
    - Hidden Markov Chain (HMM)
    - Baynesian Network (BN)

- Supply chain order lead time management through data mining
    - Anomaly Detection of excessively long lead time with DBSCAN, Local outlier Factor (LOF)
    - K-mode Clustering on mixed types of variables of production and material arrival lead time, lotsize, capacity, suppliers, etc.
    - Elastic-Net / LASSO / Ridge Regression to forecast lead time
    - Generalized Linear Mixed-effect model (GLMM), Genelized Additive Model (GAM)

- Mixed Integer Programming on a procurement problem
    - optimization with decision variable of selecting sourcing vendors and objective of minimizing total lead time, constrained on minimum order quantity, demand fulfillment, production capacity

- OR-GYM Reinforcement Learning 
    - experimenting two environments from the repository OR-GYM:  https://github.com/hubbs5/or-gym, including a) Multiple lead time period News-vendor problem, and b) 4-stage multi-echelon inventory management problem with backlogs
    - experimenting with algorithms and agents available in Tensorflow-Agents API, including DDPG, PPO, SAC and TD3, which are commonly used for continuous action space (i.e., the two problems are trying to solve an optimal ordered quantities for the next time step, based on  parameters of inventory, lead time, forecasted demand, backlogs, etc.)

- Transportation behaviours and vehicle routing problems
    - using the Green-minibus data from the Hong Kong Open Data platforam, https://www.data.gov.hk, the impementation did the simulation of a dataset which resembles the characteristics of a dial-a-ride (pickup-and-delivery, point-to-point) service
    - geospatial data visualization using geoplot package
    - spatial clustering detection of passenger sources and frequently visited destinations (public amenities) 
    - testing an optimization of vehicle routing with Google-developed ORTools package
    - testing a Reinforcement Learning agent with actor-critic networks to do the trip assignment and vehicle routing tasks

