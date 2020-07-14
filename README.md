## Under Development Currently
# DocumentClassifier

## Project Background and Description

Data Literacy is a important goal for large organizations to help their users navigate through the complex data landscape. It helps to empower data consumers in producing high-value analysis and reporting for effective decision-making. As these documents are made available securely online, preferably at a centralized location, curating this wide inventory of content becomes equally important. This content could be stored in SharePoint sites, Teams channels, KP learn resources as work document etc. Part of curation is to identify the correct metadata tags like subject, domain, region, keywords, owner, data steward, date of publishing to be attached to each resource. This is where machine learning techniques can come in handy to auto classify documents. Identifying the right metadata tags helps in fast and efficient data retrieval.

The goal of this project is to come up with the best classification algorithms that gives good accuracy for classifying a new document.

## Datasets

One of the datasets used for running the algorithm is available from:
https://www.kaggle.com/c/learn-ai-bbc/data

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

