import numpy
import pandas as pd
import json
import random
import numpy as np

reviews = []
with open('reviews_Movies_and_TV_5.json') as json_file: 
    for rec in json_file:
        dic = json.loads(rec)
        reviews.append(dic)

# Use only the first 500,000 records for faster computation
random.seed(123)
random.shuffle(reviews)
reviews = reviews[:500000]

review = []
label = []

for rev in reviews:
    review.append(rev['reviewText'])
    label.append(rev['overall'])


review_df = pd.DataFrame({"review":review, "label":label})

print("Number of documents is %s"%len(review_df))
print("Number of labels is %s"%len(review_df.label.unique()))
print("Label distribution is as follows: ")
distribution = review_df.groupby('label')['review'].nunique().reset_index()
distribution['proportion'] = distribution['review'] / len(review_df)
print(distribution)
print("Average word length in a review is %s"%(np.mean([len(text.split(' ')) for text in review_df['review']])))