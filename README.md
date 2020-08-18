# Document Classifion using Supervised Machine Learning Techniques

## Project Background and Description

Data Literacy is a important goal for large organizations to help their users navigate through the complex data landscape. It helps to empower data consumers in producing high-value analysis and reporting for effective decision-making. As these documents are made available securely online, preferably at a centralized location, curating this wide inventory of content becomes equally important. This content could be stored in SharePoint sites, Teams channels, File sahres resources as work document etc. Part of curation is to identify the correct metadata tags like subject, domain, region, keywords, owner, data steward, date of publishing to be attached to each resource. This is where machine learning techniques can come in handy to auto classify documents. Identifying the right metadata tags helps in fast and efficient data retrieval.

The goal of this project is to pick the best classification algorithms that gives good accuracy for classifying a new document that is added to the corpus.

## Project Goals:

1. The primary goal of the project is to use supervised text classification techniques to accuractely predict domain/category for a new document that shall be added to the corpus.
1. A secondary goal is also to explore text summarization techniques that can be used for documents

## Datasets

1. One of the datasets used for running the algorithms is available from https://www.kaggle.com/c/learn-ai-bbc/data. The one is used for building the model is available from http://mlg.ucd.ie/datasets/bbc.html. It consists of 2225 documents from the BBC news website corresponding to stories in five subject areas from 2004-2005. These news articles fall under 5 labels : business, entertainment, politics, sport and tech.

1. The second dataset was build using internal company documents which were classified into six categories - Membership, Claims, Product and Benefits, Pharmacy, Healthcare utilization and Provider and Network. This corpus used word, powerpoint, excel and pdf documents. Refer to [file ](/Analysis/KPDocumentCorpus_Creation and Pre-Processing.ipynb) for the Pre-processing steps involved in building this dataset. No content has been retained for confidentiality reasons. These documents are domain specific and do involve some overlap between the categories like Provider and Network with Product and Benefits as each of these area could be referring to in-network vs out-of-network from the Benefits or Provider perspective. 

## Text Classification Process Flow:

Following flow outlines the steps used in building the document classifier:

![](/images/ProcessFlow.JPG)

## Preprocessing of Documents:
1.	Extracting text from source documents
  1. SharePoint pages - Office365-REST-Python-Client package, BeautifulSoup and Spacy for document processing. Note that given internal company Sharepoint site is fairly secured basic authentication techniques did not work. It required oauth authentication which needs Sharepoint Content Administrator (SCA) role on the site. However given that i could get access to only one regional site it did not work out to be a viable source for this project. It did help provide me some good links though to other sources.
  1. Python-pptx : for extracting text from PowerPoint Presentations
  1. Docx2txt: Extracting text from Word Documents
  1. PyPDF2 : Extracting text from PDF
  1. xlrd : Extracting text from Excel worksheets
1. Removing tags, accented characters, Stop word removal, spelling corrections, removed email addresses as well, handle contractions (words like won't, havn't), apply Stemming or lemmatization, identify other patterns that might apply to the document corpus that need to be removed and use it in a custom pre-processing function. These steps are also referred to as text normalization. 
1. Ensure that the documents do not have null data and are correctly labeled


## Extracting Features/Terms/Entities from input text/documents:

Once the input text or documents have been pre-processed we want to extract numerical features from them which shall be used by the machine learning models. Several techniques like Term frequency-Inverse document frequency(tf-idf), word2vec, doc2vec have been used in this project. Here is a brief overview of the various methodologies:

**Bag of words (BoW)**

This is the most basic technique used to convert text to numbers. It does not take into account the order in which the words appear in a document and only calculates the frequency of words in a document.

**Count Vectorizers**

Count vectorizers like the one available from sklearn library convert raw text into a numerical vector representation of individual words and n-grams. These numerical features (signals) can then be used in machine learning tasks for e.g. text classification. Refer to the [notebook ](/SampleCode-NLPConcepts/sklearn_Vectorization_samplecode.ipynb)for more details on this vectorizer. A variation of count vectorizer is HashingVectorizer where in the case of large document sets we can hash words to integers and then tokenize and encode documents into vectors. Here we cannot covert the encoded words back to the original word. The benefit of this approach is we can saved on memory in the case of large datasets. Here we do not save the vocabulary (dictionary of tokens) and hence it is also faster to reload a HashingVectorizer object (Ganesan, n.d. )

**Tf-idf (term frequency–inverse document frequency) features**

Term frequency is to used for the raw count of terms in a document, i.e., the number of times that term 't' occurs in document 'd'. There are several variations to this approach like using Boolean frequencies, term frequency adjusted for document length, logarithmically scaled frequency. 

Inverse document frequency is the logarithmically scaled inverse fraction of the documents that contain the word which is obtained by dividing the total number of documents (D) by the number of documents (d) containing the term (t), and then taking the logarithm of that quotient (td-idf, n.d)
 
idf (t,D)=log N/(|{dϵD ∶ tϵd}| )
This calculation penalizes a word that appears a lot across documents. Refer to the [notebook](/SampleCode-NLPConcepts/sklearn_tfidf_usage.ipynb) for more details on extracting this feature from a document corpus.

**n-grams**
bi-grams and tri-grams can capture contextual information compared to just unigrams. For e.g. New Mexico can carry a different meaning than “New” “Mexico” considered as separate words.

**Word Embeddings**

Word embeddings are a type of word representation which stores the contextual information in a low-dimensional vector. Word2vec was introduced in 2013 and uses a simple neural network with a single hidden layer to learn the weights. Here instead of making predictions we are interested in weights of the hidden layer since they are the word embedding or vectors we want to learn. It uses 2 popular algorithms:

1. **Skip-gram model:** In this model given a input word in a sentence we want the vector to predict how likely it is for each word in the vocabulary to be nearby. We can control how many surrounding words we are looking for by controlling the window size. 
Input > One-hot vector of dimension 1X V, where V is the vocab size
Hidden Layer > Dimension of the weight matrix for hidden layer is V X N where N is the size of the hidden layer
Output > Vector with dimension of one-hot vector containing for every word in the vocabulary the probability that a randomly selected nearby word is that vocabulary word
1. **Continuous bag of words model (CBOW)**
Here given the context of words, surrounding a word in a sentence the network predicts how likely it is for each word in the vocabulary to be that word. Since this model can be hard to train because of the large matrix size some improvements like use of Word pairs/Phrases (to reduce vocab size), Subsampling frequent words (little impact on creation of good word embedding) and negative sampling (training a small portion of the weights at one time using “negative” and also “positive” words at one time). 
1. **Doc2Vec**
Doc2Vec is an extension of wordvec in which we add another vector (paragraph ID or Document ID) to the input. We in the process will be training another vector which is document specific. 


## Classification Methods used on pre-processed vectorized input 
1. Multinomial Naive Bayes Classifier
1. Random Forest
1. Logistic Regression 
1. CNN Sequential Model 
1. RRN with LSTM

## Summary of Text Classifiers Evaluation with KP Document Corpus (all using tf-idf vectorization)

|Classifiers|	 Best Test Accuracy reached	|Precision,Recall,F-Score by Category|
| :------------ | ------:| -----:|
| Random Forest | 0.62 | <img src="/images/kp_rf_cr.JPG" width="400px" height ="300px"> |
| Multinomial Naive Bayes | 0.69 | <img src="/images/kp_mnb_cr.JPG" width="400px" height ="300px"> |
| Logistic Regression | 0.86 (reached 0.93 with word2vec embedding | <img src="/images/kp_lr_cr_tfidf.JPG" width="400px" height ="300px"> |
|Sequential CNN | 0.72 (reached 0.79 in other runs) | <img src="/images/kp_cnn_cr.JPG" width="400px" height ="300px"> |

Comparison of models using different word embeddings with Logistic Regression Classifier on KP Document Corpus:

|With tf-idf |	With Word2Vec	|With Doc2vec|
| :------------ | ------:| -----:|
| <img src="/images/kp_lr_cr_tfidf.JPG" width="300px" height ="240px"> | <img src="/images/kp_lr_cr_word2vec.JPG" width="300px" height ="240px"> | <img src="/images/kp_lr_cr_Doc2vec.JPG" width="300px" height ="240px"> |

Here is a summary of the kind of datasets on which each of the approaches could work well:

| - |	TF-IDF	| Word2vec|Doc2vec|
| :------------ | ------:| -----:|-----:|
|Corpus Size |Small/Medium  | Trains word vectors and find similarity between words for small to large datasets|Needs large to Large Unbounded dataset like Google's datasets|
|Corpus Content|Limited|Varied to Domain Specific| Varied|
|Knowledge Manager| - Small team can model the entire ontology. - Subject areas are well established - Irrelevant words can be eliminated| - Medium to Vast| - Ontology should be vast that is hard to model - Difficult to identify term relevance|

*(Alianna,2020)*

Given that the KP Corpus was a small domain focused source of 104 documents  we were able to do well with tf-idf vectorization. The pre-trained word2vec vector from Google raised the accuracy by 7 % given that this approach also takes context into consideration. 
## Text Summarizer
Refer to the following subpage for details on this topic:

[Readme](/TextSummarization/Introduction_to_Text_Summarization.md)


## Conclusion and Recommendations:

1. Even though deep learning neural networks are known to perform well there is no one size fits all model for every task. We found from this project that the simple Logistic Regression Model could reach an accuracy of 93 % for a supervised text classification task with a complex dataset of only 104 documents. This was achieved using the word2vec embeddings (bag of words architecture). We can explore this further with other trained word vectors from FastText and Glove with the skip gram architecture.
1. We tried the extractive text summarization approaches using LSA, Luhn's summarizer to text summarization during this project. The results were found to be average. Hence we can try the abstractive text summarization techniques to demonstrate if they can give better results. 
1. Once the access and security issues to Sharepoint documents can be resolved we can use topic modeling or text clustering to label the data instead of manual labeling of training documents which can be time consuming. This will allow us to work with a larger dataset of documents
1. Reference data available from domain specific sites can be used to resolve acronymns in the data. For e.g for Database documents the column names are shortened using standard rules which can be fed as a input dictionary using a custom pre-processor



## References

Sarkar, Diplankar.(2016).Text Analytics with Python, A Practical Real-World Approach to Gaining Actionable Insights you’re your data

Li, Susan. Dec 8th 2019. Multi Class Text Classification with LSTM using TensorFlow 2.0. Retrieved from https://towardsdatascience.com/multi-class-text-classification-with-lstm-using-tensorflow-2-0-d88627c10a35

Goyal, P., Pandey, S., Jain, K. (2018). Deep learning for natural language processing

Alianna J. Maren. (Feb 2020). NLP: Tf-Idf vs Doc2Vec - Contrast and Compare.Retrieved Aug 9th from https://www.youtube.com/watch?v=iSkbq6Tjkj0

Fan, Shuzhan. (2018). Understanding Word2Vec and Doc2Vec. Retrieved from https://shuzhanfan.github.io/2018/08/understanding-word2vec-and-doc2vec/

D. Greene and P. Cunningham. "Practical Solutions to the Problem of Diagonal Dominance in Kernel Document Clustering", Proc. ICML 2006.

tf–idf. (n.d.). In Wikipedia. Retrieved from https://en.wikipedia.org/wiki/Tf%E2%80%93idf

Ganesan, Kavita.(n.d.). HashingVectorizer vs. CountVectorizer. Retrieved from https://kavita-ganesan.com/hashingvectorizer-vs-countvectorizer/#.XzmFq-hKiUk
