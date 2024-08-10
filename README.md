# RojakLanguageSentimentAnalysis
This is a machine learning project focused on analysing and classifying sentiments in code-switched and code-mixed text, specifically targeting the unique linguistic characteristics found in Malaysian conversations.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [References](#references)

## Introduction

RojakLanguageSentimentAnalysis is a machine learning project designed to analyze and classify sentiments within code-switched and code-mixed text, specifically focusing on Malaysian linguistic patterns. This project tackles the unique challenges of multilingual sentiment analysis by employing both deep learning and traditional machine learning models, offering a comprehensive approach to understanding sentiments in a linguistically diverse context.

## Installation

To set up the project environment, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/Wei-RongRong2/RojakLanguageSentimentAnalysis
    ```
2. Navigate to the project directory:
    ```bash
    cd RojakLanguageSentimentAnalysis
    ```
3. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the clustering analysis, follow these steps:

1. Ensure you have Jupyter Notebook installed. If not, you can install it using:
    ```bash
    pip install notebook
    ```

2. Navigate to the project directory where the Jupyter Notebook is located:
    ```bash
    cd RojakLanguageSentimentAnalysis
    ```

3. Launch Jupyter Notebook:
    ```bash
    jupyter notebook
    ```

4. In the Jupyter Notebook interface, open the `RojakLanguageSentimentAnalysis.ipynb` file.

5. Run the cells in the notebook to execute the clustering analysis.

## Methodology

This project focused on sentiment analysis of Malaysian code-switched and code-mixed text, using data from Reddit and Hugging Face. Key steps included:

### Data Collection

- **Source:** A fusion of two datasets: a Reddit dataset from the Malaysia subreddit and a Twitter rojak dataset from [mesolitica](https://huggingface.co/mesolitica) on Hugging Face.
- **Reddit Dataset:** Derived from the Malaysia subreddit using the Reddit API, capturing diverse discussions within the Malaysian community. The [Malaya](https://malaya.readthedocs.io/en/stable/) library was used to identify and gather Rojak languages.
- **Twitter Rojak Dataset:** Sourced from Hugging Face's [language-detection-dataset](https://huggingface.co/datasets/mesolitica/language-detection-dataset/tree/main), focusing on Twitter Rojak records.

### Preprocessing

- **Data Cleaning:** Removed duplicates, converted emojis to text, expanded contractions, and handled reduplicated words. Noise, including URLs and usernames, was removed while retaining punctuation for segmentation.
- **Segmentation & Tokenization:** Used Malaya HuggingFace for sentence segmentation and NLTK for tokenization, enabling precise analysis of code-switched text.
- **Language Detection & Stemming:** Detected language at the word level with Malaya’s FastText; applied stemming/lemmatization based on language.
- **Normalization:** Replaced abbreviations and removed redundant or non-standard words using regex rules.
- **Named Entity Recognition (NER):** Applied NER using Malaya and SpaCy, with challenges in code-mixed text.
- **Data Splitting:** Divided the data into training, validation, and test sets (70-15-15 ratio).
- **Feature Extraction:** Used TF-IDF, PCA (95% variance), and Truncated SVD for dimensionality reduction.

### Model Training

1. **Multinomial Naive Bayes (MultinomialNB):** Utilized for text classification due to its efficiency with count-based features like word frequencies. Trained on TF-IDF features to capture text patterns.

2. **Support Vector Machine (SVM):** Chosen for its ability to handle high-dimensional text data and its versatility in finding the optimal hyperplane in the TF-IDF feature space. Also trained on TruncatedSVD-reduced features for enhanced speed and robustness.

3. **Long Short-Term Memory (LSTM):** A deep learning model used to capture sequential and long-range dependencies in text data. The LSTM-based neural network was designed to understand contextual flow and nuanced sentiment patterns.

## Results

### Model Evaluation Results

1. **Multinomial Naive Bayes (MNB):**
   - Improved accuracy, recall, and F1 score post-tuning, though precision slightly decreased, indicating more false positives.

2. **Support Vector Machine (SVM):**
   - No significant change after tuning, suggesting default parameters were optimal, showing stability in performance.

3. **Truncated SVM:**
   - Marginal changes in performance, indicating the model likely reached its peak with the given features.

4. **Long Short-Term Memory (LSTM):**
   - Significant improvements in all metrics post-tuning, highlighting its strength in capturing temporal dependencies and reducing overfitting.

### Best Model Configuration

- **Accuracy:** LSTM (65%) slightly outperformed SVM.
- **Precision vs. Recall Balance:** LSTM provides the best balance.
- **Complexity and Interpretability:** SVM and MNB are simpler to interpret and quicker to train, ideal for scenarios requiring interpretability or efficiency.

For a more detailed explanation of these steps and results, refer to the full report: [Report - Sentiment Analysis on Out-Of-Vocabulary (OOV) Malaysia Rojak Language.pdf](./Report%20-%20Sentiment%20Analysis%20on%20Out-Of-Vocabulary%20(OOV)%20Malaysia%20Rojak%20Language.pdf).

## Contributing

Contributions are welcome! Please fork this repository, make your changes in a new branch, and submit a pull request for review.

1. Fork the repo
2. Create a feature branch (`git checkout -b feature-name`)
3. Commit your changes (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin feature-name`)
5. Create a new Pull Request

## Acknowledgments

This project was developed in collaboration with [naruto sun](https://github.com/limjosun). We worked together on the sentiment analysis, model development, and project documentation.

## License

This project is part of an academic course and is intended for educational purposes only. It may contain references to copyrighted materials, and the use of such materials is strictly for academic use. Please consult your instructor or institution for guidance on sharing or distributing this work.

For more details, see the [LICENSE](./LICENSE) file.

## Contact

Created by [Wrrrrr](https://github.com/Wei-RongRong2) - feel free to contact me!  
For any inquiries, you can also reach out to [naruto sun](https://github.com/limjosun)

## References

- **Reddit Dataset:** [Reddit API](https://www.reddit.com/dev/api/)
- **Twitter Rojak Dataset:** [Hugging Face - Mesolitica](https://huggingface.co/mesolitica)
- **Natural Language Processing:** [Malaya Library Documentation](https://malaya.readthedocs.io/en/stable/)
- **Machine Learning Algorithms:** [Scikit-Learn Documentation](https://scikit-learn.org/stable/)
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1 Score
