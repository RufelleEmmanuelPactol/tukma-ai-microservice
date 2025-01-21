import math
import re

import nltk
import numpy as np
from functools import lru_cache

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from ai.text_processing import *


class SimilarityScorer:
    """
    This is a multi-modal class that utilizes different similarity engines to calculate
    the similarity scores between keywords and a decently-sized text corpus.

    Currently, the scorer supports the following similarity engines:
    `ngram-product`: uses ngram-based similarity, where an average score is generated using ngrams where n in {1, 2, 3}.
                    this may be computationally intensive.


    """

    def __init__(self, engine='ngram-product', transformer='roberta-base-nli-stsb-mean-tokens', verbose=True):
        self.engine = engine
        self.verbose = verbose
        self.model = SentenceTransformer(transformer)
        self._write_log("Initializing SimilarityScorer")

    def _write_log(self, log):
        if self.verbose:
            print(log)



    def generate_ngrams(self, text, n):
        words = text.split()
        return [' '.join(words[i:i + n]) for i in range(len(words) - n + 1)]

    def cosine_sim(self, text1, text2):
        embedding1 = self.get_embedding(text1)
        embedding2 = self.get_embedding(text2)
        return cosine_similarity(embedding1, embedding2)[0][0]

    @lru_cache(maxsize=1280)
    def get_embedding(self, text):
        return self.model.encode([text])

    def transform_score(self, x):
        return np.sign(x) * abs(x ** 2)

    def calculate_relevance_scores(self, text, keywords):
        if self.engine == 'ngram-product':
            return self._calculate_relevance_scores_ngram(text, keywords)
        elif self.engine == 'sentence-chunk':
            return self._calculate_relevance_scores_sentence(text, keywords)
        else:
            raise ValueError("Invalid engine specified. Use 'ngram-product' or 'sentence-chunk'.")

    def _calculate_relevance_scores_ngram(self, text, keywords):
        preprocessed_text = text
        relevance_scores = {}

        for keyword in keywords:
            preprocessed_keyword = keyword

            all_scores = []
            max_score = 0
            best_ngram = ''
            for n in [3, 4, 5]:

                text_ngrams = self.generate_ngrams(preprocessed_text, n)

                for ngram in text_ngrams:
                    similarity = self.cosine_sim(ngram, preprocessed_keyword)
                    if np.isnan(similarity) or similarity is None:
                        continue
                    all_scores.append(similarity)

                    if similarity > max_score:
                        max_score = similarity
                        best_ngram = ngram

            transformed_scores = [self.transform_score(score) for score in all_scores]
            final_score = np.tanh((sum(transformed_scores) / np.sqrt(len(all_scores))) / 2) if len(
                all_scores) > 0 else 0

            relevance_scores[keyword] = {
                'similarity_score': final_score,
                'best_matching_ngram': best_ngram
            }

        print(relevance_scores)

        return relevance_scores


    def _calculate_relevance_scores_sentence(self, text, keywords):
        sentences = preprocess_text(text)
        sentences = chunk_text(sentences, max_chunk_size=24, overlap=8)
        print('[ilo]', sentences)



        print('TOKENIZED SENTENCES', sentences)
        print('LENGTH OF IT', len(sentences))
        relevance_scores = {}

        for keyword in keywords:
            relevance_unit = self.RelevanceUnit(self, keyword)
            for sentence in sentences:
                relevance_unit.transform_calculate_sentence(sentence)

            relevance_scores[relevance_unit.get_keyword()] = {
                'similarity_score': relevance_unit.aggregate_similarity(),
            }

        return relevance_scores

    class RelevanceUnit:

        def __init__(self, parent, keyword):
            self.parent = parent
            self.keyword = keyword
            self.scores = []
            self.sentences = []

        def transform_score(self, x):
            return np.sign(x) * abs(x ** 2)

        def pairwise_max(self, scores, sentences):
            i = 0
            for x in range(len(scores)):
                if scores[x] > scores[i]:
                    i = x
            return sentences[i]

        def augmented_cosine_loss(self, x, y, k):
            import torch



            cosine_sim =cosine_similarity(x.reshape(1, -1), y.reshape(1, -1))

            # Compute regular cosine distance
            cosine_dist = 1 - cosine_sim

            # Create a mask for pairs with different signs
            sign_mask = (np.sign(x) != np.sign(y))
            # Double the distance for pairs with different signs
            modified_dist = np.where(sign_mask, 4 * cosine_dist, cosine_dist)

            # Compute the loss as 1 - modified distance
            loss = 1 - modified_dist

            # Compute the mean loss
            mean_loss = loss.mean()

            return mean_loss



        def augmented_cosine_similarity(self, corpus_x, corpus_y, k):
            x_embed = self.parent.get_embedding(corpus_x)[0]
            y_embed = self.parent.get_embedding(corpus_y)[0]
            return self.relu(1 - self.augmented_cosine_loss(x_embed, y_embed, k))

        def relu(self, value):
            return max(0, value)


        def transform_calculate_sentence(self, sentence):
            self.sentences.append(sentence)
            score = self.augmented_cosine_similarity(sentence, self.keyword, 3)
            self.scores.append(score)


        def bind_scores(self, x):
            if x > 0.8:
                return 1
            if x < 0.45:
                return 0
            var  = x - 0.45
            ov_args = var / (0.8-0.45)
            return ov_args


        def aggregate_similarity(self):
            def sigmoid(x):
                return 1 / (1 + math.exp(-x))
            print(f'MAX RESULT[{self.keyword}]', self.pairwise_max(self.scores, self.sentences))
            return sigmoid(sum(self.scores) / np.sqrt(len(self.scores)) ** 2)

        def get_keyword(self):
            return self.keyword




