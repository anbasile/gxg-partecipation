"""
Classifier library
"""

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer as tfidf
from sklearn.feature_extraction.text import CountVectorizer as vectorizer

ngram = Pipeline([('features', FeatureUnion([('wrd',
                                              tfidf(binary=False,
                                                    max_df=1.0,
                                                    min_df=2,
                                                    norm='l2',
                                                    sublinear_tf=True,
                                                    use_idf=True,
                                                    lowercase=True)),
                                             ('char',
                                              tfidf(analyzer='char',
                                                    ngram_range=(3, 6),
                                                    binary=False,
                                                    max_df=1.0,
                                                    min_df=2,
                                                    norm='l2',
                                                    sublinear_tf=True,
                                                    use_idf=True,
                                                    lowercase=True))])),
                  ('clf', LinearSVC())])
words= Pipeline([('features', FeatureUnion([('wrd',
                                              tfidf(binary=False,
                                                    max_df=1.0,
                                                    min_df=2,
                                                    norm='l2',
                                                    sublinear_tf=True,
                                                    use_idf=True,
                                                    lowercase=True))])),
                  ('clf', LinearSVC())])


chars= Pipeline([('features', FeatureUnion([('char',
                                              tfidf(analyzer='char',
                                                    ngram_range=(3, 6),
                                                    binary=False,
                                                    max_df=1.0,
                                                    min_df=2,
                                                    norm='l2',
                                                    sublinear_tf=True,
                                                    use_idf=True,
                                                    lowercase=True))])),
                  ('clf', LinearSVC())])


simple = Pipeline([('features', vectorizer(lowercase=False,
                                           token_pattern=r'\b\w+\b',
                                           ngram_range=(1,2))),
                   ('clf', LogisticRegression())])


def neural(): return (_ for _ in ()).throw(Exception('NotImplementedError'))


random = Pipeline([('features', tfidf()),
                   ('clf', DummyClassifier(strategy='uniform',
                                           random_state=42))])
