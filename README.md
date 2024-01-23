## Naive Bayes Classifier for Women's Clothing E-Commerce Reviews

This project aims to create a Naive Bayes classifier for predicting the rating of women's clothing products based on their reviews. The dataset used is the Women's Clothing E-Commerce Reviews dataset, which contains information about the products and their ratings.

### Dependencies

- Python 3.x
- NumPy
- Pandas
- Scikit-learn

## Files
* `Naive_Bayes_.ipynb` - Python code to load the clothing reviews dataset, preprocess the text data, train a Naive Bayes model, and evaluate the model performance.
* `Womens Clothing E-Commerce Reviews.csv` - Input dataset of womens' clothing reviews.

## Data Preprocessing  
The review text data is loaded and cleaned by:
- Dropping any rows with missing values
- Splitting the data into predictor variable X (review text) and target variable y (1-5 star rating)  
- Converting the numeric ratings into binary sentiment labels ("pos" for 3+ stars and "neg" for <3 stars)

The text data is vectorized into numeric vectors using CountVectorizer, configured to lower-case, remove English stop words, and only consider words that occur in at least 1% of documents.

## Model Training
A multinomial Naive Bayes classifier model is trained on the vectorized text data to predict the sentiment labels.

## Model Evaluation
The model accuracy is evaluated using:

- Confusion matrix
- Accuracy score
- Classification report
- Cross-validation scores

### Future Work

- Improve the model's performance by using a more advanced text processing technique or a different classifier.
- Analyze the impact of different features on the model's performance.
- Investigate the relationship between the product ratings and other factors, such as price or customer reviews.

### Accuracy

The accuracy of the model is 0.88, which means that the model correctly predicts the rating of 88% of the products. The confusion matrix is as follows:

```
array([[ 343,  280],
       [ 423, 4853]], dtype=int64)
```

### Classification Report

The classification report provides more detailed information about the model's performance:

```
              precision    recall  f1-score   support

         neg       0.45      0.55      0.49       623
         pos       0.95      0.92      0.93      5276

    accuracy                           0.88      5899
   macro avg       0.70      0.74      0.71      5899
weighted avg       0.89      0.88      0.89      5899
```

The overall accuracy of the model is 0.88.

### Cross-Validation

The k-fold cross-validation results are as follows:

```
Cross-validation scores:[0.87218591 0.88162672 0.87944808 0.87863372 0.87136628 0.8619186 0.88735465 0.87718023 0.88008721 0.86991279]
```

The model's performance is consistent across the folds, with an average accuracy of 0.879
