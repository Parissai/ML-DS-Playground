# ML-DS-Playground

This repository is a collection of personal machine learning and data science projects created while learning and experimenting with various techniques. The projects included here cover a wide range of topics such as supervised and unsupervised learning, clustering, feature engineering, and more.


## Projects
Here is a list of some projects included in this repository:


1. **Supervised Learning**
    - [Classify Song Genres from Audio Data](classify_song_genres.ipynb)
        - **Techniques:** Exploratory Data Visualization, Feature Scaling, Feature Reduction, Cross-Validation, Logistic Regression, Decision Trees
        - **Tasks:** Classification of song genres based on audio features, model evaluation using classification reports.

2. **Unsupervised Learning**
    - [Clustering Antarctic Penguin Species](clustering_antarctic_penguin.ipynb):
        - **Techniques:** Data Manipulation with Pandas, Dummy Variable Creation, Elbow Method, K-Means Clustering, Feature Scaling
        - **Tasks:** Clustering penguin species based on features like flipper length and body mass.

3. **Feature Engineering**
    - [Exploring NYC Public School Test Result Scores](exploring_school_test_result.ipynb)
        - **Techniques:** Sorting and Subsetting, Grouped Summary Statistics
        - **Tasks:** Feature engineering to understand NYC public school test results.

4. **Deep Learning with Keras**
    - [Building an E-Commerce Clothing Classifier Model with Keras](e_commerce_clothing_classifier.ipynb)
        - **Techniques:** Neural Networks in Keras, Compiling and Training Models, CNNs, Image Classification
        - **Tasks:** Classification of clothing items in e-commerce datasets using convolutional neural networks.


5. **Natural Language Processing and Audio Analysis Techniques**
    - [Customer Support Calls](customer_support_call.ipynb)
        - **Techniques:** Speech Recognition, Audio Feature Extraction, Sentiment Analysis, Named Entity Recognition (NER), Text Similarity/Embedding
        - **Tasks:** Analyzing customer support calls to identify sentiment, entities, and key topics from audio.

6. **Data Engineering for E-commerce Orders and Demand Forecasting with PySpark**
    - [Cleaning an Orders Dataset with PySpark](cleaning_dataset_PySpark.ipynb)
        - **Techniques:** Data Cleaning, PySpark, Data Transformation
        - **Tasks:**
            - Cleaning and preprocessing an e-commerce orders dataset.
            - Removed orders placed between 12 am and 5 am.
            - Created a time_of_day column.
            - Filtered out products that are no longer sold.
            - Converted columns to lowercase and extracted relevant address data.
            - Exported cleaned data for use in demand forecasting.

    - [Building a Demand Forecasting Model](building_demand_forecasting_model.ipynb)
        - **Techniques:** Random Forest Regression, Feature Engineering with PySpark, Time-Series Forecasting
        - **Tasks:**
            - Forecasting sales and inventory needs for promotional planning.
            - Cleaned and aggregated sales data at daily intervals.
            - Built and trained a Random Forest model to predict future product sales.
            - Evaluated model performance using Mean Absolute Error (MAE).
            - Predicted sales for promotional week and ensured optimal inventory management.

7. **Recommendation Systems and Data Visualization**
    - [Comparing Cosmetics by Ingredients](ccomparing_cosmetics_by_ingredients.ipynb)
        - **Techniques:** Content-Based Filtering, Ingredient Tokenization, Document-Term Matrix (DTM), Dimensionality Reduction with t-SNE, Interactive Visualization with Bokeh
        - **Tasks:** Built a recommendation system for moisturizers targeting dry skin by analyzing ingredient similarities. Used t-SNE for dimensionality reduction and Bokeh for interactive product comparison visualization.


## Technologies Used

The projects in this repository are implemented using the following technologies:

- **Programming Language:** Python
- **Interactive Environment:** Jupyter Notebook
- **Libraries for Data Manipulation and Analysis:** Pandas, NumPy
- **Machine Learning Frameworks:** Scikit-Learn, TensorFlow/Keras (for deep learning projects), PySpark (for large-scale data processing and engineering)
- **Visualization Tools:** Matplotlib, Seaborn, Bokeh
- **NLP Libraries:** NLTK, SpaCy, SpeechRecognition (for speech-to-text)
- **Clustering and Feature Engineering Techniques:** K-Means Clustering, t-SNE, Document-Term Matrix (DTM)
- **Specialized Techniques:** Random Forest Regression for forecasting, Content-Based Filtering for recommendation systems