
# Medical Keyword Extraction

The task is to develop a natural language processing (NLP) system for handling domain-specific medical transcriptions. The goal is to fine-tune a pre-trained language model on a given dataset, incorporating both internal and external language models, and evaluate its effectiveness in comparison to a pre-trained baseline for domain-specific NLP tasks.

### Task Distribution:

#### Dataset Understanding:

The dataset consists of medical transcriptions with two main features:

- Transcription: The actual medical text transcriptions.
- Keywords (Labels): Relevant keywords associated with each transcription.

#### Data Preprocessing:

- Text Cleaning:
    - Remove special characters.
    - Convert text to lowercase.
    - Tokenize the text for further analysis.
    - Handling Missing Values and Duplicates:

- Remove rows with missing values in the 'transcription' and 'keywords' columns.

- Splitting Data: Split the dataset into training and validation sets for model training and evaluation.

- Train/Fine-tune: Utilize a pre-trained language model, i.e. BART, and fine-tune it on the medical transcription dataset for better performance on domain-specific tasks.

- Evaluation

-----


##### I addressed the specified problem statement by employing three distinct models:

##### 1. OneVsRestClassifier:

- This model is applied for multi-label classification to predict relevant keywords for medical transcriptions.
- Utilizes a Stochastic Gradient Descent (SGD) classifier with a logistic loss function.

##### 2. Facebook BART Base Model:
- A pre-trained BART (Bidirectional and Auto-Regressive Transformers) model is fine-tuned on the provided medical transcription dataset.
- The objective is to enhance the language model's performance specifically for domain-specific NLP tasks in the medical field.

##### 3. KeyBERT Model:
- KeyBERT, a keyword extraction model, is employed to extract key terms from medical transcriptions.
- This approach focuses on unsupervised keyword extraction as an alternative to the supervised methods used in the previous models.

-----

### Libraries Used
* `matplotlib.pyplot`: For creating visualizations.
* `numpy`: For numerical operations.
* `seaborn`: For statistical data visualization.
* `train_test_split`: For splitting the dataset into training and testing sets.
* `OneVsRestClassifier`: For handling multi-label classification using multiple binary classifiers.
* `SGDClassifier`: Stochastic Gradient Descent classifier for linear models.
* `accuracy_score, jaccard_score`: Evaluation metrics for classification performance.
* `CountVectorizer and TfidfVectorizer`: For converting text data into numerical features.
* `torch`: PyTorch library for building and training neural networks.
* `Dataset, DataLoader`: PyTorch utilities for handling datasets and data loading.
* `Adam`: An optimization algorithm for updating network weights in neural networks.
* `tqdm`: A library for displaying progress bars.
* `pandas`: For data manipulation and analysis.
* `Transformers`: Library for natural language processing with pre-trained models. Provides pre-trained models like BART (used for conditional text generation) and tokenizers for natural language processing tasks.
* `KeyBERT`: Library is employed to extract keywords from medical transcriptions in the given code snippet. Specifically, the KeyBERT class is used to create a model instance for keyword extraction.

-----


### OneVsRestClassifier
- Loads a medical transcription dataset from a CSV file, selecting only the 'transcription' and 'keywords' columns. Null values in 'transcription' and 'keywords' columns are dropped.
- The dataset is split into training and testing sets using train_test_split.
- TF-IDF vectorization is applied to convert text data into numerical features. Parameters like min_df, smooth_idf, tokenizer, ngram_range, and max_features are set for the TF-IDF vectorizer.
- Used multi-label classification model employing the One-vs-Rest strategy, utilizing the Stochastic Gradient Descent (SGD) classifier with logistic loss.
- Predictions are made on the test set using the trained model and Jaccard Index is calculated as an evaluation metric for multi-label classification.
- Got an accuracy of 85.47 while tested in test data.

-----

### Facebook BART Base Model
- Loads a medical transcription dataset from a CSV file, selecting only the 'transcription' and 'keywords' columns. Null values in 'transcription' and 'keywords' columns are dropped.
- Initializes a BART (Bidirectional and Auto-Regressive Transformers) model and tokenizer from the Hugging Face Transformers library.
- Defines a custom dataset class for handling medical transcription and keyword data.
- Splits the dataset into training and testing sets.
- Creates PyTorch data loaders for training and testing.
- Uses Adam optimizer for model training.
- The training loop iterates through epochs and batches, updating model weights.
- Defines a function for generating keywords using the trained BART model.

#### Reasoning: BART is chosen for keyword extraction due to its effectiveness in sequence-to-sequence tasks and text summarization and extraction.

-----

### KeyBERT Model
- Loads a medical transcription dataset from a CSV file, selecting only the 'transcription' and 'keywords' columns. Null values in 'transcription' and 'keywords' columns are dropped.
- Uses the KeyBERT model for keyword extraction from medical transcriptions.
- Iterates through the transcriptions, extracts keywords, and prints both the actual and model-extracted keywords for comparison.

#### Reasoning: KeyBERT: A keyword extraction model is chosen for its ability to extract keywords from text using BERT embeddings. It operates in an unsupervised manner, making it suitable for cases where labeled data may be limited. Applied to each medical transcription to extract the top 15 keywords. The actual keywords from the dataset are compared with the model-extracted keywords.

-----

### Code credit

Code credits for this code go to [Aman Khandelwal](https://github.com/wolfblunt)