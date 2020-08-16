## Under Development Currently
# DocumentClassifier

## Project Background and Description

Data Literacy is a important goal for large organizations to help their users navigate through the complex data landscape. It helps to empower data consumers in producing high-value analysis and reporting for effective decision-making. As these documents are made available securely online, preferably at a centralized location, curating this wide inventory of content becomes equally important. This content could be stored in SharePoint sites, Teams channels, KP learn resources as work document etc. Part of curation is to identify the correct metadata tags like subject, domain, region, keywords, owner, data steward, date of publishing to be attached to each resource. This is where machine learning techniques can come in handy to auto classify documents. Identifying the right metadata tags helps in fast and efficient data retrieval.

The goal of this project is to come up with the best classification algorithms that gives good accuracy for classifying a new document.

## Datasets

1. One of the datasets used for running the algorithms is available from https://www.kaggle.com/c/learn-ai-bbc/data. The one is used for building the model is available from http://mlg.ucd.ie/datasets/bbc.html. It consists of 2225 documents from the BBC news website corresponding to stories in five topical areas from 2004-2005. These news articles fall under 5 labels : business, entertainment, politics, sport and tech.

1. The second dataset was build using internal company documents which were classified into six categories - Membership, Claims, Product and Benefits, Pharmacy, Healthcare utilization and Provider and Network. This corpus used word, powerpoint, excel and pdf documents. Refer to [file ](/Analysis/KPDocumentCorpus_Creation and Pre-Processing.ipynb) for the Pre-processing steps involved in building this dataset. No content has been retained for confidentiality reasons. These documents are domain specific and do involve some overlap between the few categories like Provider and Network with Product and Benefits and each of these area could be referring to in-network vs out-of-network from the Benefits or Provider perspective. 

## Preprocessing of Documents:
1.	Extracting text from SharePoint pages - Office365-REST-Python-Client package to HTML code. You might need to have a Sharepoint Content Administrator (SCA) role to be able to authenticate.
2.	Identify the right python libraries for extracting data from pdf (PyPDF2), doc, docx files.
3.	Ensure that the documents available for the different classes are balanced and if not come up with sampling techniques to circumvent the issue. 
4.	Removing tags, accented characters, Stemming and lemmatization, Stop words
5.	Ensure that the documents are correctly labeled

## Feature Extraction:
Identify the right features extraction technique that will result in classification model for obtaining better accuracy. As an example, we will use:
1.	Term frequency-Inverse document frequency
2.	Feature extraction using word embedding (doc2vec)

Potential Classification Methods (at least 3-4 methods from the list below) to be used on pre-processed data:
1.	Naive Bayes Classifier
2.	Linear Classifier
3.	Support Vector Machine
4.	Bagging/Boosting Models
5.	Latent Dirichlet Allocation which is a powerful machine learning technique used to sort documents by topic.
6.	Explore the potential use of (deep) neural networks for text classification


## References

D. Greene and P. Cunningham. "Practical Solutions to the Problem of Diagonal Dominance in Kernel Document Clustering", Proc. ICML 2006.
