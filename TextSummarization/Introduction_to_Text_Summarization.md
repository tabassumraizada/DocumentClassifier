## What is Automatic Text Summarization
Process of shortening a text document using software automation to create a summary of the major points of the original document. The goal of this summarizer is to optimize topic coverage from the original document while keeping it readable. 

**What is relevant to KP Document Catalog**
There is an abundance of data in any organization and we want to get to the relevant document that meets our needs by reading through the summary to decide which search results meets our need. This saves time in information retrieval. 

**Approaches to text summarization**
1)	Extraction based summarization
This approach uses key phrases from source document and combines them to make a summary. Here we could be extracting entire sentences or a piece of a sentence. This is a simpler and hence more popular approach.
2)	Abstraction based summarization 
This technique involves paraphrasing and shortening parts of the source document. The algorithms used will create new phrases and sentences to convey the original intent of the document. Hence the output is more grammatically correct as well. It utilizes advanced deep learning models which are more resource intensive and requires domain knowledge for good results. 

**Summary of Text Summarization Techniques:**

|Period|	 Type	|Method	|Description|
| :------------ | ------:| -----:|-----:|
|Pre-2000|	Extractive Method	|Positional Methods: Introduced by P.Baxendale in 1958	|In technical literature topic sentence was found to be mostly the first and last sentence of the paragraphs. It used Naïve approach but was simple and worked most of the time|
|Pre-2000	|Extractive Method	|Luhn’s Method (1958)|Used the frequency of content terms for creation of literature abstracts. Significant words are in the middle range of frequency|
|Pre-2000	|Extractive Method	|Edmunsons Method |Uses cue words (bonus, stigma and null words) which needs domain specific knowledge|
|Pre-2000	|Extractive Method	|Kupiec in 1995	|Around 1995 several Machine learning-based classification techniques starting getting used for test summarization.Improvements were found by moving from a Naive Bayes classification approach.|
|2000 and Later|Extractive Method	|LexRank 	|Graphical Approach that used lexical centrality for finding similar sentences. Similarity between sentences based on established connections with other sentences. This well-connected sentence recommends similar sentences to the reader. |
|2000 and Later|Extractive Method	|TextRank	|Another graph based ranking algorithm similar to Google PageRank.  It extracts the topics, creates nodes out of them and captures the relationship between them to summarize the text.|
|2000 and Later	|Abstractive Methods	|Sequence to Sequence	|Uses Deep learning Technique with encoder and decoder model.  

**Metrics for evaluating text Summarization:**

Precision and Recall metrics can be used for comparing summarization results that are produces by humans with machine generated summaries. Sometimes the sentences can be very similar and even hard for a human to decide. Hence another measure called utility is used where humans are required to rank the sentence too. Summaries are scored based on the rank.
Pyramid Method: Used for evaluating multi-document summarization. Higher weight if a semantic content unit appears in more documents. 
ROUGE stands for Recall-Oriented Understudy for Gisting Evaluation. It includes a set of metrics for evaluating automatic summarization of texts as well as machine translation. It works by comparing an automatically produced summary or translation against a set of reference summaries (typically human-produced).

**Python Libraries used in text summarization:**

1)	[Sumy ](https://pypi.org/project/sumy/)comes with a automatics text summarizer for HTML pages and plain text documents. Several summarization methods can be implemented for e.g:
Luhn’s
Edmundson
LexRank
TextRank among others.  
2)	Genism library that is used for topic modeling can also be used for summarization based on ranks of text sentences using a variation of the TextRank algorithm

**References: **

Nekic, Masa (2019). Automatic text summarization. Presented in Oslo Conference. Retrieved from https://www.youtube.com/watch?v=_d0OXm0dRZ4




