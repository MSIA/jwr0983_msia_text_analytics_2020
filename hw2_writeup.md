# HW-2

**Name:** Wang Jue

**NetID:** jwr0983

## Problem 1

For this problem I used 3 different models to evaluate the performance of different parameters for word2vec in gensim library. The first one uses CBOW with window size = 5, embedding size = 100 and minimum count = 3, the second using CBOW with window size = 10 and embedding size = 150 and minimum count = 3, while the third one uses skip-gram with window size = 10, embedding size = 100 and minimum count = 3.  

Computation run-time:
- CBOW with window size 5 and embedding size 100: 1.95 seconds
- CBOW with window size 10 and embedding size 150: 1.98 seconds
- Skip-gram with window size 5 and embedding size 100: 2.97 seconds

From the above results we see that skip-gram is generally slower than CBOW on this dataset. The reason is that skip-gram approach involves more calculations. For the same window size skip gram has to undergo significantly more backward propagations than CBOW, making it computationally more inefficient. 

We used the following 10 words to test whether the three models give comparable results in terms of similar words. The results are mostly similar, and as an example, for the word "god", the first model gives the top 5 similar words "existence, fact, belief, evidence, and manifestation", the second model gives "existence, fact, belief, evidence, manifestation", while the third model gives "satan, exist, believing, existance and exists". As the corpus is relatively small, we see that the embedding size of 100 is sufficient to give sensible results. 


## Problem 2 



### Relevant Code

- File1 can be found [here](<https://github.com/username/repo/file1)
- File2 can be found [here](<https://github.com/username/repo/file1)