
# Handbook to architect machine learning systems


#### SECTION I: Overview of the Challenges in operationalizing a machine learning system vs. operationalizing a traditional software system (Using this as an intro chapter about different challenges)
1. Software 1.0 vs. Software 2.0

2. How do we tackle it? 

3. Data Governance 

#### SECTION II: Data Pipelining and ETL Processes 
#### SECTION III: Versioning and Governance of Models and Data -- Reproducibility 
1. Crisis due to lack of reproducibility
2. Version control of machine learning experiments
i. Data versioning
ii. Model versioning
3. Tools to manage reproducibility of experiments 
i. Sagemaker
ii. MLflow
iii. DVC
iv. Datmo

#### SECTION IV: Resiliency of Models (Regression Testing and Validation)
1. Tests for model evaluation during training
2. Ongoing tests for model evaluation during prediction / scoring
#### SECTION V: Production based deployment
1. Deploy model using serverless as REST APIs
2. Deploy models and maintain on phones - MLCore
3. Deploy models using Qualcomm SDK on phones
4. Debt due to process management
i. Version control of models in production (mature 6. systems may have dozens or hundreds of models running simultaneously)
ii. Airflow to manage processes as DAGs
#### SECTION VI: Monitoring of models in production (Measurable Evaluation) 
1. Monitoring tools for deployments - what needs to be monitored?
2. Tools for monitoring in production
3. Performing online experiments in production
#### SECTION VII: Continuous Integration
1. Maintaining reproducibility and performance in development loop
2. Examples of loops you can create with tools available today
#### SECTION VIII: Continuous Delivery -- Putting it all together
1. The machine learning development loop 
2. Examples of loops you can create  with tools available today
-----


## Section I 

### Overview of the Challenges in operationalizing a machine learning model

Our goal in writing this book is to cover the key topics to consider in operationalizing machine learning and to provide a practical guide for navigating the modern tools available along the way. The intent is also to make this a reference for those who are machine learning engineers, ranging from junior to senior, and are looking to integrate models into broader, scalable software systems. To that end, the subsequent sections will include code snippets with working examples to test out the concepts and apply them to your own code base. 

The goal of the book is not to explain machine learning concepts, as it is understood that the audience would already have that experience, but rather to tackle the auxiliary challenges like dealing with large data sets, computational requirements and optimizations, and the deployment of models and data to large software systems. 

In this section we’ll give a brief overview of each of the key sections we will cover. This overview will serve as a map for you to flip to the appropriate section of interest for your current problem. 


#### 1. Software 1.0 vs Software 2.0 

Most classical software applications are deterministic where the developer writes explicit lines of code that encapsulate the logic for a desired behaviour.

Where as, the ML software applications are probabilistic where the developer writes a more abstract code and lets the computer write the code in a human unfriendly language i.e. the weights or parameters required for the ML model

This requires us to look at new ways of getting data, cleaning, training and deployments methods since apart from code, we have the weights and data which keep changing.


##### ETL

One of the first steps in starting machine learning projects is to gather data, clean the data, and make it ready for the purpose of experimenting and building models 
The initial techniques may start with doing the above in a manual way but without an automated pipelines to operationalize these ETL processes, the technical debt increases over time. 

In addition, there needs to be a way to store large data either on cloud storage or file storage system. Storage also means proper tooling for gathering, labeling and making the data access scalable. 

Finally, as the data is being transformed, it is key to keep track of versions of data so downstream, when the data is being used for experimentation, training or testing of algorithms, there is a trackable version of data that run can be associated with.

##### Machine Learning Experiments

Once data is gathered and explored, it is time to perform feature engineering and modeling. While some methods require strong domain knowledge to make sensible decisions feature engineering decisions, others can learn significantly from the data. Models such as logistic regression, random forest or deep learning techniques are then run to train the algorithms.

There are multiple steps involved here and keeping track experiment versions are essential for governance and reproducibility of previous experiments. 

Hence, having both the tools and IDE around managing experiments with Jupyter notebook, scripts, and others is essential. Such tools require provisioning of hardware and proper frameworks to allow data scientists to perform their jobs optimally.


##### Deployment and management of model and ML pipelines

After the model is trained and performing well, in order to leverage the output of this machine learning initiative, it is essential to deploy the model into a product whether that is on the cloud or directly “on the edge”. 

##### Deployment in Machine Learning systems can be classified into following ways: 

1. Offline predictions: If you have large set inputs you would like to get the predictions on them without any immediate latency requirements, you can run batch inference in a regular cycle or with a trigger
2. Online predictions: If you would like to make predictions soon after the request is made then this deployment helps while making calls either by serving as REST APIs or RPC calls. 
3. Edge deployment: You can perform online predictions while keeping on the device to decrease the delay in making online calls. This has trade off with accuracy vs power consumed

##### Monitoring of models in production

After the model has been deployed it is also essential we understand the performance of models in production to avoid issues with model or concept drift. 

Tools for monitoring visualizations of data distributions and creating metrics around how the test data differs from the training data can be leveraged to track ongoing model performance and ensure the best performance. 

##### Continuous Integration

Usual software continuous integration (CI) principles do not directly translate into the world of ML. Data scientists and ML engineers are not writing code according to a prototype specification, so it feels unnatural to write unit tests.

CI for ML has two key goals:
1. Reproducibility: Ensure the key bits of code are working correctly and consistently
2. Performance: Monitor progress and improvements in our predictions over time
   
CI should be built by using test data and running prediction scripts while validating the outputs received from the model on ground truth data at regular intervals.

##### Continuous delivery

This step combines all the above mentioned steps into a reproducible pipeline, which can be run sequentially. Generally it will include the following steps:

**ETL of data:** Create a robust self-service data pipeline along with a set of utility tools that make gathering, loading, transforming and building datasets a much faster and simpler task

**Continual re-training of your models**

1. ML service needs to be integrated with a customer-facing production feature, we need to be able to ensure we can sustain it through time. One part of sustaining a model is the ability to retrain it or recreate it from scratch
2. Once finished, after training, store accuracy test results into a DB and the associated models in a File Storage system (e.g. AWS S3) 
   
Establish a Continuous Integration pipeline around using test data and run predictions scripts to validate the reproducibility of the desired output and the performance of the model.

Then we can deploy the model into production either for online prediction as REST API, offline predictions or have it deployed on the edge. In the deployed code establishing a method for ongoing monitoring of these models enables full end-to-end performance modeling of the model. 

Continuous delivery ensures that a pipeline for all of the above steps are created. Traditional software tools for continuous delivery can still be leveraged with a few tweaks, along with specific data processing and machine learning tools to establish an automated and reliable workflow. 



## SECTION III

### Versioning and Governance of Models and Data -- Reproducibility 
1. Crisis due to lack of reproducibility
2. Version control of machine learning experiments
i. Data versioning
ii. Model versioning
3. Tools to manage reproducibility of experiments 
i. Sagemaker
ii. MLflow
iii. DVC
iv. Datmo

As data scientists frequently intersperse model training and model building, there are many challenges in reproducing experiments and ensuring a reliable process. Here are some of the major aspects that make reproducibility a difficult task:
Managing libraries: Many have faced the issue of installing the magical permutation of packages needed to run your code. Sometimes it’s TensorFlow version breaking if you upgraded CUDA to a new version. For others, it’s solving the Rubik’s cube of PyTorch, CuDNN, and GPU drivers. There are a growing number of ever-evolving frameworks and tools for building machine learning models and they are being developed independently by various third parties — managing their interactions is a huge pain.
Managing experiments: What happens when the test accuracy was higher three runs ago but I forgot what hyperparameter configuration I used? Or trying to remember which version of preprocessing resulted in the best model from the latest batch of runs? There are countless examples of needing to record experiments along with the software environment, code and data, and other experiment metadata but they are decoupled in the status quo. 
In order to track machine learning experiments and pipelines, there are a few common components that are required:
Code:
Managing code is key to all parts of software engineering. In machine learning it is no different, having a clear commit of the software code that was run to create a result is essential. Traditional tools like git, mercurial, etc. are perfect to ensure this. 
Environments:
This refers to the right hardware configurations (e.g. GPU vs. CPU, etc) and software libraries necessary to run the model code. 
Setting up software dependencies can be done with dependency management tools such as virtualenv, conda environment, pip, among others. Software dependencies however are also being handled through containers and virtual environments such as Docker, Kubernetes, etc. Although those tools emulate hardware as well, they can be an effective tool to ensure all dependencies are present immediately. 
Hardware entails the memory, and compute necessary to run the code. Oftentimes, modern solutions such as virtual machines and docker containers also will have all software dependencies already contained. Hardware dependencies can be achieved with orchestration tools such as kubernetes, docker swarm and terraform (or another cloud managed solution) and have reduced the operations side of recreating a self-contained hardware environments. 
Data:
Versioning of data is essential in order to retrace back to the data used during experimentation and for training purposes. There are many ways to capture and keep track of data during experimentation processes. 
For any given experiment it is key to have a unique image of the data that was used. If any changes in the data occur, provenance of that data must be tracked
Parameters and results
Model parameters and key results also need to be incorporated in tracking the experiment to ensure that you have a way to compare your experiments against each other. For example, if I have only one hyperparameter `\alpha` and run an experiment with an `\alpha=0.1` and `f1=0.8`, then if I keep the same `\alpha` I should expect the same output result of `f1=0.8` or that `f1=0.8 +- 0.1`. 
Artifacts from the experiments
Artifacts are a term that is used to reference files that are generally not captured within the “Code” or “Data”. The catch-all term refers to output images or graphs which encapsulate certain metrics, or weights files (e.g. `.pkl` files or `.pt` files). 
In order to solve the issues we might face with the above here a few practical solutions we can use to encapsulate each one of these key factors and some examples of how they can be used.
 
Solution 1: Amazon Web Services Ecosystem 
 
Data: Data can be stored in a number of ways. Here we will give the most typical categories of storage and describe tools in each category to allow 
No-SQL databases
MongoDB
Cassandra
Couchbase 
AWS DynamoDB
SQL databases
AWS RDS
MySQL
PostgreSQL
File Storage 
AWS S3 Service
 
 
Code: There are a number of code versioning methodologies including, but not limited to, git, mercurial, cvs, monotone, etc. Most modern stacks and companies today use git, although it is not universal. As that is the case, the tools we consider are git focused 
AWS CodeCommit
Github
BitBucket
 
Parameters and Results: Storing parameters and key results from a training run is typically done in a few ways. Traditionally, these have been stored in research notes, google docs, or other file documents. More modern ways involve integrating these values within a database, or external clients. Below are some key categories and key examples
File systems
AWS S3 
Google Sheets 
Microsoft Excel 
Databases [See above in Data]
Third-party Systems 
Sagemaker
This is a cloud service provided by Amazon and provides the ability to build, train and deploy the machine learning models faster. This cloud service provides the capability to run multiple jobs and helps in quickly tuning and optimizing the model. This repository[LINK] provides many examples to run on sagemaker
Mlflow
This tool records and compares parameters and results. This can be setup by either locally running it for tracking local experiments or deployment of mlflow server to work collaboratively with other team members. 
DVC
This allows to keep track of versions of code and data required to reproduce the pipeline completely. One can run all commands which is a part of the pipeline and track the components necessary to reproduce the experiment. However, the environment has to be tracked by the user while using this tool. An example code for reproducing the ML pipeline 
Datmo: 
This allows you to locally run experiments, track parameters and results along with environment. One can rerun the previous experiment. An example code for reproducing ML experiment 
 
Code Example
 
MLflow - An example code for tracking parameters and results.
DVC - example
Datmo - example

#### SECTION IV: Resiliency of Models (Regression Testing and Validation)
1. Tests for model evaluation during training
2. Ongoing tests for model evaluation during prediction / scoring


Tests for model evaluation during training 
Problem definition [REF]
Regression testing in traditional software generally are a subset of unit and functional tests that are already executed to specifically ensure new code changes haven’t ruined the programs function
Testing in machine learning is more than just the standard input and output in traditional software. A dataset is used to train an algorithm via logic and algorithms written into code, which then creates an artifact or state of the model. In this way, testing, requires not just the code, but a given set of data and a set of artifacts that can be tested against. 
Machine learning training can be a non-linear process since it has so many components and their effects can be confounded. You can go 1 step backwards only to go 10 forward later. 


Solution
Creating regression tests are critical to ensuring that human error doesn’t cause machine learning models to become worse. 
Regression tests require a base dataset to be used with the code to develop a final output. 
One methodology is to exactly replicate the model weights, however, this is only practical by using a pre-defined seed for all optimization algorithms. 
Here we would need to completely match the exact weights, which means we would have to remove all statistical variation. This may not be favorable because it artificially forces an inherently non-deterministic process to have a deterministic result. 
Another practice is to instead come up with a proxy for the output, by defining ranges of specific metrics that would allow for statistical variation but still ensure the model is moving in the right direction. 
Some examples of regression tests of that nature are ensuring that certain metrics, like accuracy, F1 score, or false positive rate are within a certain range. 
The hard part of machine learning is that these regression tests have to be slightly modified everytime you come up with a better model or “commit” of your machine learning model.
One way to solve this is creating code that takes the thresholds as inputs and adjusts these thresholds based on the “version” of the model you last trained. 
[EXAMPLE] : In this example, we have code that runs an LSTM algorithm to train a classifier to take in 3 words and predict the next word in the sequence. The algorithm has a number of parameters that determine the learning rate, meta data, among others. However, these need to be constant as part of the “code”. For example, the number of epochs that the training is run, must always be the same. Similarly, we have a validation set of 3 words, and the subsequent word for the dataset. The code and datasets are kept consistent. In the regression test, the training code is run on the same validation set. The output, is not completely deterministic, however, it generally lies in a range of accuracy. The regression test, tests against that and asserts the output is in the range -- if it isn’t, the test fails and we see that the change has caused a decrease in performance. 
“Regression Tests” are similar to what practicioners call “validation tests” which are generally run at regular intervals during the training process. However, these validation tests are usually for the practicioner to eyeball the number to see if it makes sense. By adding a quantitative range bound to the validation test, it becomes a regression test as used in traditional software engineering processes. 

Ongoing tests for model evaluation during prediction / scoring
Problem definition: 
Traditional software, can use the same tests in staging and testing as in production, because the functionality of code determined by input and output. 
For machine learning, the validation set is separate from the test set, which means that our “true” test is done with a larger test set. This test set is larger, thus more statistically relevant, and is the preferred “metric” to be used when comparing algorithms. 
The issue in production machine learning, is that there can be data drift that occurs in the input that can cause the machine learning model to produce non-sensical results. 
This means that the distribution of the input given during prediction time may not match that of the historical data, whether that data is in the training, validation, or test set. 
Solution
Ongoing model evaluation
In order to ensure a high degree of accuracy there is a statistically sound way to create a validation loop for prediction using tools like Amazon’s Mechanical Turk, Upwork, Playment, MightyAI, CrowdAI, CrowdFlower, among many others. However, doing this for all inputs would defeat the purpose of machine learning in the first place. Thus, randomly sampling a subset of the inputs, similar to selecting the validation set within a training set, will suffice to check the accuracy over a statistically significant number of samples. The right number varies depending on the problem, though at least 1000, is usually correct. 
[EXAMPLE]: Using the same LSTM example above, we can consider the implications of running this in prediction mode. Here the user will input 3 words, and the algorithm will output the next word via forward propagation of the LSTM model. We have added another script which randomly selects a subset of inputs to take in the ground truth input via the command line. Of course, in a production system you may use one of the tools and their APIs to automatically provide this input rather than have a user input this via command line. After 10 such inputs, the script will output the evaluation metrics. We can see the word predicted, the ground truth, and various aggregate measures such as accuracy defined as the a complete match of the string or none. We can also calculate accuracy by averaging the inverse of the Levenshtein distance. That is, the lower the Levenshtein distance, the more accurate the model is. 
Data drift 
The key goal in data drift is to identify it immediately as it becomes apparent. This typically means monitoring distributions of input data and comparing this against the distribution of a large initial historical dataset. Then by overlaying the two distributions you can compare the difference in area, difference in mean, and difference in standard deviation / variance. Enterprise paid tools like StreamSets [LINK], Domino Data Lab [LINK], Dataiku [LINK], among others, offer an ability to monitor data in this way. 
[EXAMPLE]: Using the same LSTM example, we can consider the input values of the 3 words and compare the distribution of these features. In the code example, we look at a few ways to view and compare these distributions. Since these features are not quantitative and continuous, we cannot simply look at a standard continuous distribution of values. Rather, we can break it down in a few ways: 1) word counts, 2) bi-gram, 3) tri-gram, 4) n-gram. The code is broken down in these 4 ways and graphs the historical data distribution in red, and the input data in blue. 

#### SECTION V: Production based deployment
1. Deploy model using serverless as REST APIs
2. Deploy models and maintain on phones - MLCore
3. Deploy models using Qualcomm SDK on phones
4. Debt due to process management
i. Version control of models in production (mature 6. systems may have dozens or hundreds of models running simultaneously)
ii. Airflow to manage processes as DAGs


Section V
https://github.com/asampat3090/awesome-data-machine-learning-tools

https://github.com/alirezadir/Production-Level-Deep-Learning?files=1

At a particular stage in the life cycle of machine learning models is to run the models at scale than having it run in local.  To be able to make it usable by different stakeholders getting insights from the model. This requires it to be deployed in any of the following ways,
Accessing the model as RESTful API
Accessing the model on the edge
Performing batch predictions using the pipeline

There exists different challenges in model deployment which needs to be considered in teams,
Environment of models: Multiple algorithms require different tools based on the problem being solved and this requires the data scientist to be able to keep an exact version of environment used during the model building phase but trimmed to the essential libraries. Docker, Conda env and virtual env helps in this challenge. 
Connectivity: As building for the business problem, their can be different algorithms which need to be connected in order to finally solve it. For example, consider building a video surveillance system, where it needs to detect threat based on human posture. For building this system, you may have the following model,


	This requires us to build connections between different algorithms which can have 
            different versions and environments. Hence, it is essential to have a tool to connect 
            models in a pipeline. 
System metrics: Managing system metrics is essential to monitor the hardware utilization for compute and memory units. Tools such as grafana along with prometheus or datadog helps in monitoring system metrics
Model performance metrics: As the data scientist keep deploying new algorithms, it is essential to keep a version mapped to the metrics associated with it. 
Scalability: Models in production have usage in short bursts with multiple users accessing it. At the end, in order to manage user requirements, it needs to scale. Deployment needs to autoscale as per the requirements. 
Model Governance: Agile environments requires data scientists to have continuous deployment of algorithms to staging and then in production. However, it is essential that a model version has all necessary meta information to be able to recreate the model outputs as the version in production. This is helpful for future reference, auditing and transparency on model behaviour. 


Deploy model using serverless as REST APIs
Link
Link
Challenges: 1. Heterogeneity, 2. Composability, 3. Performance and success metrics, 4. Iteration, 5. Infrastructure, 6. Scalability, 7. Audibility and Governance
Model Deployment ● Connectivity ● Serving & Scaling ● Management & Governance ● Ongoing Service & Support


Deploy model using sagemaker as REST APIs
https://aws.amazon.com/getting-started/tutorials/build-train-deploy-machine-learning-model-sagemaker/
Deploy models and maintain on phones - MLCore
Deploy models using Qualcomm SDK on phones
Debt due to process management
Version control of models in production (mature systems may have dozens or hundreds of models running simultaneously)
Airflow, Kubeflow, argo to manage processes as DAGs

#### SECTION VI: Monitoring of models in production (Measurable Evaluation) 
1. Monitoring tools for deployments - what needs to be monitored?
2. Tools for monitoring in production
3. Performing online experiments in production
#### SECTION VII: Continuous Integration
1. Maintaining reproducibility and performance in development loop
2. Examples of loops you can create with tools available today
#### SECTION VIII: Continuous Delivery -- Putting it all together
1. The machine learning development loop 
2. Examples of loops you can create  with tools available today


References

Previous Work 
[READ] LINK_HERE
NOTES HERE

[Anand - READ] https://www.oreilly.com/ideas/operationalizing-machine-learning (dinesh nirmal) 
5 pillars of “fluid ML” - Managed (versioning / governance of models and data), Resilient (regression testing for ML) , Performant (fast training & scoring with cross-platform support), Measureable (Monitoring + evaluation), Continuous (Automated re-training + deployment) 
[Anand - READ] https://www.gartner.com/binaries/content/assets/events/keywords/catalyst/catus8/preparing_and_architecting_for_machine_learning.pdf 
Framework: Input data -> Learning -> Output data (predictive or prescriptive)
Challenges:
Compute
Large Datasets required
Specialized talent
Complex data integration
Data context changes (data drift) 
Algorithmic bias, privacy, ethics 
Process: Classify -> acquire data -> process data -> model problem -> validate -> deploy
ML types: Exploratory, Predictive, Unsupervised Learning, Supervised Learning
Largest proportion of work on data prep, modeling, not deployment
Process: Data -> ETL -> feature eng -> model eng -> model eval


Recs: 
Build taxonomy for classifying problems / challenges
Evaluate self-serve platforms with data prep and applied ML
Offer ML as a toolkit to DS rather than allowing to build their own customized algos 
Use public cloud to start initiative because of scale

[READ] https://papers.nips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems.pdf
Summary
Machine learning is very different from traditional software and is prone to tech debt
“Traditional abstractions and boundaries may be subtly corrupted or invalidated by the fact that data influences ML system behavior”
Traditional software has CLEAR abstraction boundaries, ML systems do not
Entanglement: CACE principle -- Changing Anything Changes Everything 
Things that can change: input signals, hyper-parameters, learning settings, sampling methods, convergence thresholds, data selection, etc
Solutions: 
Isolate models / serve ensembles 
Monitor changes in prediction results and alerts
Correction Cascades: create new models which are dependent on each other -- each is correlated and any change in one affects all others
Undeclared Consumers: aka “Visibility Debt” in traditional software. Parts of the system are unknowingly using the outputs of model predictions for another part of the system. This causes a hidden dependency. 
Data Dependencies are worse than Code Dependencies: code deps can be detected by linkers / compilers, data deps cannot
Unstable Data Dependencies: data comes from another model or the data changes over time
Solutions: 
Versioned copy of the input signal → downside is that there is potential staleness and there is a “cost” to maintain multiple versions (no current tool to do this) 
Underutilized data dependencies: input signals with LITTLE incremental benefit (similar to mostly unneeded code dependencies) 
Examples of these :
Legacy features -- e.g. early feature superceded by new features w/ the same information
Bundled features -- all features are added b/c no time to explore and choose
Incremental features -- features that add small improvements 
COrrelated features -- 2 features where one is more causal than the other
Solution: 
Leave one feature out evaluations
Static Analysis of Data Dependencies: similar to code w/ static analysis of dependency graphs
Solution:
Automated features management system
Feedback loops: ML systems update themselves → “analysis debt”
Direct Feedback loops:  model directly affects selection of features (e.g. bandit)
Hidden Feedback Loops: one model affects another based on it’s results (e.g. 2 unrelated stock buying models affect each other through the market) 
Anti-patterns: THINGS YOU SHOULDN’t DO
Glue Code: supporting code is unordered -- solution: wrap black box packages into common APIs for your specific use case (using standard modular coding paradigms)
Pipeline Jungles: creating data pipelines without intermediate steps 
Solution: 
Hybrid engineering and research teams
Thinking holistically about data collection / feature extraction
Dead Experimental Codepaths: putting code for experiments directly in production and leaving it there without cleaning [ e.g. knight capital lost $465M in 45 min bc of this error) 
Solution: add “Dead” flags to remove experimental branches of code
Abstraction Debt: there is no abstraction in ML currently: maybe Map-Reduce or parameter-server abstraction: solution is to create your own currently
Plain-Old-Data Smell: only using simple data types and not wrapping predictions and results in objects
 Multiple-Language Smell: using languages as you wish, no common language makes testing difficult
Prototype Smell: using too many prototype environments rather than having proper production env.
Configuration Debt
Researchers and engineers do not consider configurations as in need of maintenance
Include: feature selection, data selection, algorithm-specific learning settings, pre-, post- processing, verification methods, etc.
Solution
Easy to specify config as a small change from previous config (diff) 
Hard to make manual errors, omissions, or oversights
Easy to see visually differences in configurations b/w models
**Easy to easy to assert and verify basic facts about the config
**Detect unused or redundant settings
Configurations should undergo a full code review and be checked into a repo
External Changes in the World
Set dynamic self-learning thresholds in dynamic systems
Monitoring system behavior in real time and and automated responses 
Prediction bias: distribution of predicted labels matches that of observed labels? Drastic changes can trigger alerts
Action Limits: if models take actions, set limits after which it stops and alerts the user for investigation
Up-Stream Producers: data passed to the ML system or down from the ML system should meet service-level objectives that are strict and checked 
Other
Data Testing Debt: check to ensure the data meets requirements 
Reproducibility Debt: create systems that allow for reproducing results (hard with parallelism, initial conditions, non-determinism, etc
Process Management Debt: tooling to manage multiple models 
Cultural Debt: reward simplicity, better reproducibility, stability, and monitoring on the same level as accuracy. 
Conclusion
How easily can a new algorithmic approach be tested as FULL SCALE
How precisely can a change be measured?
How does improving  one model degrade others?
How quickly can new members of the team be brought up to speed?
Challenges that still need to be addressed
Overly dependent on accuracy, NOT focused on reproducibility, stability, and monitoring 
ML systems do NOT have clear abstraction boundaries
Entanglement : changing anything changes everything
Correction cascades: models are dependent on each other. Upstream issues propagate downstream very easily
Undeclared Consumers: downstream consumers span various fields
Data Dependencies!
Feedback Loops
Configuration Debt 
Reproducibility is not built in
Poor processes to handle multiple models

[READ] https://venturebeat.com/2017/11/28/infrastructure-3-0-building-blocks-for-the-ai-revolution/
Summary:
“ML/AI remains a limited and expensive discipline reserved for only a few elite engineering organizations”
“ the systems and tooling that helped usher in the current era of practical machine learning are ill-suited to power future generations of the intelligent applications they spawned”
“ the next great opportunity in infrastructure will be to provide the building blocks for systems of intelligence” -- infrastructure 3.0 
Infrastructure 2.0 :  Linux, KVM, Xen, Docker, Kubernetes, Mesos, MySQL, MongoDB, Kafka, Hadoop, Spark, and many others — defined the cloud era. My colleague Sunil Dhaliwal described this shift as Infrastructure 2.0.
“How do we make sense of the world?”
Graduate from a research to engineering discipline
Opportunities:
Specialized hardware with many computing cores and high bandwidth memory (HBM) very close to the processor die. These chips are optimized for highly parallel, numerical computation that is necessary to perform the rapid, low-precision, floating-point math intrinsic to neural networks.
Systems software with hardware-efficient implementation that compiles computation down to the transistor level.
Distributed computing frameworks, for both training and inference, that can efficiently scale out model operations across multiple nodes.
Data and metadata management systems to enable reliable, uniform, and reproducible pipelines for creating and managing both training and prediction data.
Extremely low-latency serving infrastructure that enables machines to rapidly execute intelligent actions based on real-time data and context.
Model interpretation, QA, debugging, and observability tooling to monitor, introspect, tune, and optimize models and applications at scale.
End-to-end platforms that encapsulate the entire ML/AI workflow and abstract away complexity from end users. Examples include in-house systems like Uber’s Michelangelo and Facebook’s FBLearner and commercial offerings like Determined AI*.
Challenges that still need to be addressed
Systems and tooling that brought practical machine learning CANNOT power future intelligent applications
[READ] https://www.forbes.com/sites/janakirammsv/2017/12/28/3-key-machine-learning-trends-to-watch-out-for-in-2018/#556db29d1280
Summary: 
Devops for Data Science: DS’s need a simple way to do round trip b/w local and cloud-based env.
Ex: Amazon SageMaker and Azure ML Workbench
setting up the right development environment,
 packaging the code as container images, 
scaling the containers during the training and inference, versioning existing models, 
configuring a pipeline to upgrade models with newer versions seamlessly
Edge computing : deploy as FaaS in serverless environment after heavy training
Ex: AWS Deep Lens and Azure IoT Edge
AI for IT ops: 
Ex: Amazon Macie and Azure Log Analytics 
Challenges that still need to be addressed
Hard to do a round-trip between local and cloud-based environments

[READ] https://www.technologyreview.com/s/609646/2017-the-year-ai-floated-into-the-cloud/
Summary: 
Driven by large companies, AI services in the cloud have ballooned, will continue to increase in 2018

[READ] https://www.forrester.com/report/Predictions+2018+The+Honeymoon+For+AI+Is+Over/-/E-RES139744
Summary:
55% of firms have not yet achieved tangible business outcomes from AI
Garbage-in, garbage-out still applies to data 
20% of firms will use AI for decision-making and real-time instructions→ data driven decision making
Unstructured text data can be processed by various firms
⅓ enterprises will pull the plug on investing in data lakes → need to get tangible business outcomes (increased rev or decreased costs) 
50% of firms will adopt cloud-first strategies bc flexible pricing, more control 
> 50% of CDOs will report to CEO
Data engineer is the hot job title of 2018 (13x more than data scientist) → train people internally 
Buy insights and business outcomes, NOT more software
Partner with Academia for insights 

[READ] “operationalizing” the data science stack: https://www.nextplatform.com/2017/12/14/enterprises-challenged-many-guises-ai/
Summary: 
Open source frameworks aren’t useful for operationalizing 
Enterprises are looking for solutions to take the gruntwork out of building these systems themselves
Ex: uber michaelangelo: leverage system eng for infrastructure, and let data scientists focus on the data science
Current solutions have no process supervision or resiliency of disaster recovery
Once they had successful models, they wanted to go to production → but this fell on the data scientists
Many vendors: domino data lab, mapd, kinetica
Open sources frameworks bring down the “mathematical and library perspective” down so more can access it (10 years ago this was not the case) → now the same thing is happening with ML in production and using from a “development standpoint”, new systems and tools in the next 10 years will change that.
Challenges that still need to be addressed
No process supervision or resiliency or disaster  recovery in DS pipelines
Open source frameworks aren’t useful for operationalizing the very algorithms they create

[READ] Preparing and Architecting Machine Learning: https://www.gartner.com/binaries/content/assets/events/keywords/catalyst/catus8/preparing_and_architecting_for_machine_learning.pdf
Summary:
Broken down into parts:
Data acquisition
Data processing
Data Modeling
Execution
Deployment
Alternative break down 
Acquire
Organize
Analyze
Deliver
Recommendations
Build taxonomy for classifying problems to be solved by ML
Evaluate self-service platforms for data prep / applied ML
Offer ML as a toolkit to data scientists rather than have them build their own custom algs
Use public cloud to start your ML initiative bc it can elastically scale  to accommodate any req

[READ] Road to enterprise AI: https://www.gartner.com/imagesrv/media-products/pdf/rage_frameworks/rage-frameworks-1-34JHQ0K.pdf
Summary
Predictions
2019, >10% of IT in customer service will write scripts for bot interactions
2020, orgs w/ cognitisystem design in AI projects will achieve long-term success 4x more than others
2020, 20% of companies will dedicate workers to monitor/guide NNs
2019, startups will overtake amazon, google, ibm, microsoft in driving AI w/ disruptive biz solutions
2019, AI platform services will cannibalize revenues for 30% of market-leading companies
Current Findings
Magnus revang
Orgs not approaching AI / smart machines as systems different from traditional IT (hint: BUT THEY ARE)
AI in early stages of innovation and have “high churn” -- this is a major concern for IT orgs 
Orgs looking into AI focused prematurely on evaluating tech/vendors instead of ACQUIRING skills for implementation (hint: PROVIDE TRAINING) 
Success in PoCs is followed by difficulties in scaling and maintaining system performance 
Jim Hare
Many enterprises view ai as magical investments that DO NOT need tending after deployment (THEY DO) 
Data science teams are mantinaed OUTSIDE of the IT app portfolio
CIOs have not prioritized talent for NNs
Not focused on where AI is alredy used
Frances Karamouzis
Every software /biz process will be more intelligent w/ AI
CIOs must find AI vendors w/ domain-specific solns
Many startups are led by employees who often worked on previous AI projects in big companies
Whit Andrews
2010-2015 funding in AI increased 7x
Since 2000, 52% of large global companies have bankrupt or ceased to exist
Companies failures resulted from not seeing disruptive forces and failed to adapt	
Challenges that still need to be addressed
Successful PoCs with AI is achieved with readily available tools, but scaling and maintaining system performance is not possible with current tools 

[READ] https://hackernoon.com/the-ai-hierarchy-of-needs-18f111fcc007



[READ- Shabaz] https://blog.paperspace.com/ci-cd-for-machine-learning-ai/


Data:
Most important difference between traditional web apps and ML pipeline is the fact that the primary input to the system is not just code but code and data.
Data orchestration, cleaning, versioning etc. is important which is required solve the reproducibility crisis for modern ML. Powerful ETL tools such as Airflow, Argo must be used for maintaining reproducible pipelines along with Pachyderm or Quilt for versioning data. 
Accelerators
Applications require using GPUs or TPUs instead of CPUs. Requires us to have our CI/CD platform take advantage of these new hardware.
Training step
Training step in a ML pipeline usually takes the most computation time. Requirement of training on new data needs proper training setup programmatically. Distributed training (involving multiple distinct compute nodes) has made big progress very recently (with tools like Uber's Horovod, TensorFlow Distributed, and a suite of libraries built in to PyTorch, to name just a few) which adds additional complexity to the task.
Having a system for easy deployment allowing either for batch, streaming or single inference easier management for different throughputs and frameworks
Refitting/online
Holy grail of ML pipelines — Realtime, online predictive engines that not only deploy forward but are actually continuously updated. 
Traditional web application development is largely “unidirectional” in the sense that it flows from code —> deployment. Usually there are static releases and modern software development oftentimes necessitates distinct “development”, “staging” and “production” environments. Technologies such as JenkinsX support things like distinct deployments for each developer or branch but this doesn’t change the fact that most deployment processes looks pretty static from the outside. 
Self-updating deep learning engine opens up a host of problems around model drift where the outputted model of classification or prediction engine beings to diverge from the input in an undesirable way. 
[READ-Shabaz] https://medium.com/onfido-tech/continuous-integration-for-ml-projects-e11bc1a4d34f

Usual software CI principles do not directly translate into the world of ML. Data scientists and ML engineers are not writing code according to a prototype specification, so it feels unnatural to write unit tests.
CI to ML should have two key goals:
To ensure the key bits of code are working correctly (reproducibility)
To see the progress we are making in out predictions (performance

We used this framework quite effectively in one of our ML project. We wrote a git hook to execute the code to check for reproducibility and performance.
Reproducibility. On every commit we would re-run our key jupyter notebooks to make sure that their output did not change (this was especially useful for our data preprocessing).
Performance. We wrote a daemon that monitored our shared models directory for when a new model file was added. The daemon automatically ran a battery of tests on that model and recorded the results: both the jupyter notebook and a csv file of key metrics.
We created a simple dashboard interface to view the results for every commit and model file. We instituted that everyone in the team should link these in their pull requests.


CI for Software engineering:

          

CI for Machine Learning:

[READ-Shabaz] https://medium.com/onfido-tech/continuous-delivery-for-ml-models-c1f9283aa971

Gathering data: We created a robust self-service data pipeline along with a set of utility tools that make gathering and building datasets a much faster and simpler task
Continually re-train your models
ML service has been integrated with a customer-facing production feature, we need to be able to ensure we can sustain it through time. One part of sustaining a model is the ability to retrain it or even recreate it from scratch

Training phase takes the following steps:
An engineer will set a parameterized build in a Jenkins job to kick off the process
The Jenkins job will build a docker image (if necessary) with the new code.
With some of the parameters in (1) and the docker image built in (2) we are ready to kick off training in AWS Batch. At this stage the batch job can do all that’s needed without human intervention.
Once finished, Batch will store accuracy test results (showing how good the model is) and the associated models in s3.
Deployment of ML services to production
HTTP API: usually Flask on top of Gunicorn and gevent
Monitoring: Datadog and Datadog Trace
Container definition using Dockerfile
Container orchestration configuration using k8s template
Jenkins takes care of combining the code, bundling the ML models and then triggering deployment to Kubernetes after CI.
[READING - Shabaz] Mastering the mystical art of model deployment https://medium.com/@julsimon/mastering-the-mystical-art-of-model-deployment-c0cafe011175

[READ - Shabaz ]SYSML paper - Towards High-Performance Prediction Serving Systems http://www.sysml.cc/doc/74.pdf

Prediction systems have three main performance requirements in order to be usable by customers as ML-as-a-service,
Latency has to be minimal with high SLA
Small resource usage-such as memory and CPU-to save operational costs
High throughput to handle as many concurrent requests as possible


In order to achieve the above three performance requirements following additions/modifications were performed when multiple DAGs are being used in production,
Caching Layer: This design is based on the insights that many DAGs have similar structures; sharing operators and, when possible, also operators’ state (parameters) can considerably improve memory footprint, and consequently the number of predictions served per machine. An example is language dictionaries used for input text featurization, which are often in common among many models and use a relatively large amount of memory.
Mapping Layer: While the Parameter Store is populated, the mapping layer builds a logical representation of each input model DAG composed of operators’ metadata (e.g., type) and links to related parameters (state) in the Caching Layer. Offline, the logical representation is compiled into stages. Inside each stage, (logical) operators are fused together when possible to improve memory locality and end-to-end performance. 
Scheduling Layer: Once model DAGs are assembled and compiled into stages (offline), they are deployed for execution in an environment where they share resources with other DAGs. The Scheduling Layer coordinates the execution of multiple stages via an event-based scheduling 
[READING - Shabaz] https://medium.com/onfido-tech/caspian-a-serverless-self-service-data-pipeline-using-aws-and-elk-stack-6d576f8ce369

[READING] Best practices for ML implementation (from Google): http://martin.zinkevich.org/rules_of_ml/rules_of_ml.pdf

SYSML paper -  Runway: machine learning model experiment management tool http://www.sysml.cc/doc/26.pdf

SYSML paper - Mobile Machine Learning Hardware at ARM: A Systems-on-Chip (SoC) Perspective  http://www.sysml.cc/doc/48.pdf

SYSML paper - Deploying deep ranking models for search verticals http://www.sysml.cc/doc/59.pdf

SYSML paper - Towards High-Performance Prediction Serving Systems http://www.sysml.cc/doc/74.pdf

SYSML paper - SLAQ: Quality-Driven Scheduling for Distributed Machine Learning∗ http://www.sysml.cc/doc/86.pdf

SYSML paper - Towards Interactive Curation & Automatic Tuning of ML Pipelines http://www.sysml.cc/doc/118.pdf

SYSML paper - Better Caching with Machine Learned Advice http://www.sysml.cc/doc/121.pdf

SYSML paper - Not All Ops Are Created Equal!  http://www.sysml.cc/doc/137.pdf

Concept Drift Introduction 
https://machinelearningmastery.com/gentle-introduction-concept-drift-machine-learning/

Use Cases / Examples (Interviews) 
[READING] Linkedin Data Pipeline: https://www.youtube.com/watch?v=Rt8JRrfsmg0&t=176s
[READING] Hakka Labs Pipeline: https://www.youtube.com/watch?v=Khdbhw2UfS8
[READING] Realtime data pipelines (spark summit) -- kafka  + spark streaming: https://www.youtube.com/watch?v=wMLAlJimPzk
[READING] Pydata Talk -- luigi, folium, pandas, scikit learn (with anaconda): https://www.youtube.com/watch?v=TYtHzvys33A


Existing Tools in the Market [https://github.com/asampat3090/awesome-data-machine-learning-tools]

https://github.com/alirezadir/Production-Level-Deep-Learning?files=1



Data Pipeline
Airflow → Setup DAG Data pipelines and manage them 
https://www.youtube.com/watch?v=oYp49mBwH60
https://www.youtube.com/watch?v=cHATHSB_450&t=81s

Data Acquisition / Data Lake Tech 
Petaho
Search Technologies

Data Processing / ETL
Databricks
Cloudera
MapR

Data Modeling / Execution
Kubernetes → orchestration tool to manage containers
Amazon SageMaker
Azure ML Workbench

Model Experimentation Layer
IBM Runway Experimentation Framework → http://www.sysml.cc/doc/26.pdf
Aloha from Zefr → http://www.sysml.cc/doc/13.pdf

Model Deployment 
Algorithmia → “The AI Layer”, selling function as a service 
OpenFaas → https://github.com/openfaas/faas → open source function as a service
Kubernetes → orchestration tool to manage containers
Amazon SageMaker and Azure ML Workbench -- easy to deploy as a microservice 
AWS Deep Lens / Azure IoT Edge  -- deploy inference as FaaS in serverless environment after heavy training
Clipper.ai → “low latency prediction serving” -- http://clipper.ai/overview/, https://github.com/ucbrise/clipper
Kubeflow - https://github.com/google/kubeflow
Singnet - https://github.com/singnet/singnet
Pachyderm - http://pachyderm.readthedocs.io/en/latest/getting_started/local_installation.html
ATM  - https://github.com/HDI-Project/ATM
Prototype @ Microsoft → http://www.sysml.cc/doc/74.pdf
Using Kubernetes CRDs → http://www.sysml.cc/doc/135.pdf
