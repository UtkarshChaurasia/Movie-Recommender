from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

text = ["London Paris London","Paris Paris London"]
cv = CountVectorizer() #To compute the cosine similarity. The output of this comes as a sparse_matrix.

count_matrix = cv.fit_transform(text)

#print count_matrix.toarray()
similarity_scores = cosine_similarity(count_matrix)

print similarity_scores
