import os.path
from abc import ABC, abstractmethod

from sklearn.preprocessing import StandardScaler
#from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer
import torch
import copy
from torch.nn.functional import normalize
import nltk
from  .PyPlotHelper import plot_similarity_heatmap, plot_histogram
from .EmbeddingService import EmbeddingService, FastTextEmbeddingService, GloveEmbeddingService, SentenceTransformerEmbeddingService
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
import os
import pickle
import re

nltk.download('punkt')


def find_lowest_max_similarity(similarity_matrix, section_names1, section_names2, n=3):
    # Get the maximum similarity along the first dimension
    max_similarities, max_indices = torch.max(similarity_matrix, dim=1)

    # Get the top-n and bottom-n indices
    top_n_vals, top_n_indices = torch.topk(max_similarities, n, largest=True)
    bottom_n_vals, bottom_n_indices = torch.topk(max_similarities, n, largest=False)

    sections_doc1 = list(section_names1)
    sections_doc2 = list(section_names2)
    print(sections_doc2)

    # Retrieve the corresponding sections for top-n similarities
    top_n_sections = [(sections_doc1[i], sections_doc2[max_indices[i].item()], top_n_vals[j].item()) for j, i in
                      enumerate(top_n_indices)]

    # Retrieve the corresponding sections for bottom-n similarities
    bottom_n_sections = [(sections_doc1[i], sections_doc2[max_indices[i].item()], bottom_n_vals[j].item()) for j, i in
                         enumerate(bottom_n_indices)]

    return top_n_sections, bottom_n_sections


def get_three_lowest_similarities(similarity_vector, embeddings_second_2d):
    # Get the three smallest values and their indices
    min_vals, min_indices = torch.topk(similarity_vector, 3, largest=False)

    list_of_sections = list(embeddings_second_2d.keys())

    # Retrieve the corresponding sections and similarity values
    lowest_sections = [(list_of_sections[idx], min_vals[i].item()) for i, idx in enumerate(min_indices)]

    return lowest_sections


class Comparator(ABC):
    '''
    the class two compare two translated articles
    '''

    def __init__(self,
                 metric: str,
                 approach: str,
                 #tokenizer: BertTokenizer.from_pretrained('bert-base-uncased'),
                 model: EmbeddingService): #BertModel.from_pretrained("bert-base-uncased")):
        '''
        :param metric: defines the metric two compare two word embeddings
        :param tokenizer:      embedding tokenizer: BERT, GPT, etc.
        :param model:          embedding model: BERT, GPT, etc.
        '''
        self.metric = metric
        self.approach = approach
        #self.tokenizer = tokenizer
        self.model = model
        #self.model.eval()

    def get_bert_embeddings(self, sentence):
        '''tokens = self.tokenizer.tokenize(sentence)
        tokens = ['[CLS]'] + tokens + ['[SEP]']

        max_seq_length = 512
        tokens = tokens[:max_seq_length]

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # padding_length = max_seq_length - len(input_ids)
        # input_ids += [tokenizer.pad_token_id] * padding_length

        input_ids = torch.tensor([input_ids])
        with torch.no_grad():
            outputs = self.model(input_ids)
            embeddings = outputs.last_hidden_state[:, 0, :]
        return embeddings'''
        return torch.tensor(self.model.encode(sentence))

    @abstractmethod
    def get_similarity(self, first_translated_dict, second_translated_dict):
        pass

    def get_similarity_matrix(self, first_translated, second_translated):
        embeddings_first = [self.model.get_embeddings(subset1) for subset1 in
                            first_translated]
        embeddings_second = [self.model.get_embeddings(subset2) for subset2 in
                             second_translated]

        embeddings_first_2d = torch.squeeze(torch.stack(embeddings_first), dim=1)
        embeddings_second_2d = torch.squeeze(torch.stack(embeddings_second), dim=1)

        column_norms_first = torch.norm(embeddings_first_2d, dim=1)
        column_norms_second = torch.norm(embeddings_second_2d, dim=1)

        embeddings_first_2d_normalized = embeddings_first_2d.T / column_norms_first
        embeddings_second_2d_normalized = embeddings_second_2d.T / column_norms_second

        similarity_matrix = torch.matmul(embeddings_first_2d_normalized.T, embeddings_second_2d_normalized)

        return similarity_matrix

    def compute_the_word_vectors_for_words(self, content):
        content_words_dict = {key: re.findall(r'\b[a-zA-Z]+\b', value) for key, value in content.items()}
        result_cache = {
            key1: torch.stack(
                [torch.squeeze(embedding, dim=0) for s1 in subset1 if len(s1) > 2 and (embedding := self.model.get_embeddings(s1.lower())) is not None], dim=1)
            for key1, subset1 in content_words_dict.items() if len(subset1) > 2 and subset1}
        list_keys = list(content.keys())
       # name = 'cached_embeddings/' + self.model.__class__.__name__ + '/' + list_keys[0] + '-' + list_keys[1] + 'word-wise'
        name = os.path.join('cached_embeddings', self.model.__class__.__name__,
                            f"{list_keys[0]}-{list_keys[1]}word-wise")
        os.makedirs(name, exist_ok=True)
        with open(name + '.pkl', 'wb') as f:
            pickle.dump(result_cache, f)
        return result_cache

    def compute_the_word_vectors_for_sentences(self, content):
        #splitted_preprocessed_content = preprocess(content)
        result_cache = {
            key1: torch.stack([torch.squeeze(embedding, dim=0) for s1 in subset1 if (embedding := self.model.get_embeddings(s1)) is not None], dim=1)
            for key1, subset1 in content.items()}
        list_keys = list(result_cache.keys())
        # had to refactor it for platform independence
        #name = 'cached_embeddings/' + self.model.__class__.__name__ + '/' + list_keys[0] + '-' + list_keys[1] + 'sentence-wise'
        name = os.path.join('cached_embeddings', self.model.__class__.__name__,
                            f"{list_keys[0]}-{list_keys[1]}sentence-wise")
        os.makedirs(name, exist_ok=True)
        with open(name + '.pkl', 'wb') as f:
            print('store the file' + list_keys[0] + '-' + list_keys[1] + 'subset_to_subset' + '.pkl in cache')
            pickle.dump(result_cache, f)
        return result_cache

    def load_cached_embedding(self, content):
        if self.splitting == 'word-wise':
            return self.load_cached_embeddings_word_wise(content)
        else:
            return self.load_cached_embeddings_sentence_wise(content)

    def load_cached_embeddings_sentence_wise(self, content):
        splitted_preprocessed_content = preprocess(content)
        list_preprocessed_keys = list(splitted_preprocessed_content.keys())
        #name = 'cached_embeddings/' + self.model.__class__.__name__ + '/' + list_preprocessed_keys[0] + '-' + \
           #    list_preprocessed_keys[1] + 'sentence-wise'
        name = os.path.join('cached_embeddings', self.model.__class__.__name__,
                            f"{list_preprocessed_keys[0]}-{list_preprocessed_keys[1]}sentence-wise")
        os.makedirs(name, exist_ok=True)
        get_emb = lambda x: self.compute_the_word_vectors_for_sentences(splitted_preprocessed_content)
        if os.path.exists(name + '.pkl'):
            with open(name + '.pkl', 'rb') as f:
                print('caching the file' + list_preprocessed_keys[0] + '-' + list_preprocessed_keys[1] + 'subset_to_subset' + '.pkl')
                cached_dictionary = pickle.load(f)
            return cached_dictionary
        else:
            result_stored_content = get_emb(content)
            list_preprocessed_content = list(result_stored_content.keys())
            with open(name + '.pkl', 'rb') as f:
                print('caching the file' + list_preprocessed_content[0] + '-' + list_preprocessed_content[1] + 'subset_to_subset' + '.pkl')
                cached_dictionary = pickle.load(f)
            return cached_dictionary

    def load_cached_embeddings_word_wise(self, content):
        list_keys = list(content.keys())
      #  name = 'cached_embeddings/' + self.model.__class__.__name__ + '/' + list_keys[0] + '-' + list_keys[
         #   1] + 'word-wise'
        name = os.path.join('cached_embeddings', self.model.__class__.__name__,
                            f"{list_keys[0]}-{list_keys[1]}word-wise")
        os.makedirs(name, exist_ok=True)
        get_emb = lambda x: self.compute_the_word_vectors_for_words(content)
        if os.path.exists(name + '.pkl'):
            with open(name + '.pkl', 'rb') as f:
                print('caching the file' + list_keys[0] + '-' + list_keys[1] + 'subset_to_subset' + '.pkl')
                cached_dictionary = pickle.load(f)
            return cached_dictionary
        else:
            result_stored_content = get_emb(content)
            list_preprocessed_content = list(result_stored_content.keys())
            with open(name + '.pkl', 'rb') as f:
                print('caching the file' + list_preprocessed_content[0] + '-' + list_preprocessed_content[1] + 'subset_to_subset' + '.pkl')
                cached_dictionary = pickle.load(f)
            return cached_dictionary


'''def preprocess(content_arr):
    #sentences = [nltk.sent_tokenize(content) for content in content_arr]
    #sentences = [[sentence.strip() for sentence in inner_list if sentence.strip() != ""] for inner_list in sentences if len(inner_list) != 0]
    #result = [sublist + sentences[index + 1] if len(sublist) < 3 and index + 1 < len(sentences) else sublist for
    #          index, sublist in enumerate(sentences[:-1])]
    #result = result[:len(result) - 1]

    #return result'''


def preprocess(sentences_dict):
    result = {}
    keys_to_remove = []

    for key, content in sentences_dict.items():
        sentences = nltk.sent_tokenize(content)

        if len(sentences) < 3:
            if len(result) > 0:
                prev_key = list(result.keys())[-1]
                result[prev_key] += sentences
                keys_to_remove.append(key)
            else:
                result[key] = sentences
        else:
            result[key] = sentences

    # for key in keys_to_remove:
    #    del result[key]

    return result


class CCAComparator(Comparator):
    def __init__(self, metric, approach, model, topic_num, splitting, doPlots) -> None:
        super().__init__(metric=metric,
                         approach=approach,
                         #tokenizer=tokenizer,
                         model=model)
        self.topic_num = topic_num
        self.splitting = splitting
        self.doPlots = doPlots

    def get_similarity(self, first_translated, second_translated):
        if self.approach == 'article_to_subset':
            return self.get_CCA_score_article_to_subset(first_translated,
                                                        second_translated)
        elif self.approach == 'subsets_to_subsets':
            return self.get_CCA_score_pairwise(first_translated, second_translated)
        else:
            print("Unknown metric", self.metric)
            return torch.zeros(len(first_translated), len(second_translated)),

    def get_canonical_vectors(self, A, B, k):
        #covA = torch.cov(A)
        #covB = torch.cov(B)
        mean_a = torch.mean(A * 1.0, dim=0)
        mean_b = torch.mean(B * 1.0, dim=0)
        A_centered = A - mean_a
        B_centered = B - mean_b
        A_cov = (A_centered.T @ A_centered) / A.shape[0]
        B_cov = (B_centered.T @ B_centered) / B.shape[0]
        L_aa, V_aa = torch.linalg.eig(A_cov)
        V_aa_L_aa = torch.matmul(V_aa.real, torch.diag(1 / torch.sqrt(L_aa.real)))
        AA_inv_sqrt = torch.matmul(V_aa_L_aa, V_aa.real.T)
        print("V_aa_L_aa", V_aa_L_aa.shape, "   AA_inv_sqrt:", AA_inv_sqrt.shape)

        L_bb, V_bb = torch.linalg.eig(B_cov)
        V_bb_L_bb = torch.matmul(V_bb.real, torch.diag(1 / torch.sqrt(L_bb.real)))
        BB_inv_sqrt = torch.matmul(V_bb_L_bb, V_bb.real.T)

        M_1 = torch.matmul(AA_inv_sqrt * 1.0, A.T * 1.0)
        M_2 = torch.matmul(M_1, B * 1.0)
        M = torch.matmul(M_2, BB_inv_sqrt * 1.0)

        try:
            U, s, V = torch.linalg.svd(M, full_matrices=True)
        except Exception as e:
            print("An error occurred during SVD computation:", e)
            return None, None, None
        corr_a, corr_b = torch.matmul(AA_inv_sqrt, U[:, :k]), torch.matmul(BB_inv_sqrt, V.T[:, :k])
        return torch.matmul(A, corr_a), torch.matmul(B, corr_b), s

    def get_CCA_score_pairwise(self, first_translated, second_translated):
        if self.splitting == 'word-wise':
            return self.get_CCA_pairwise_wordwise(first_translated, second_translated)
        if self.splitting == 'sentence-wise':
            return self.get_CCA_pairwise_sentencewise(first_translated, second_translated)

    def get_CCA_pairwise_wordwise(self, first_translated, second_translated):
        if self.metric == 'pairwise-component-max':
            component_num = lambda x, y: min(x.shape[1], y.shape[1])
        else:
            component_num = lambda x, y: round(
                min(x.shape[1], y.shape[1]) * self.topic_num)  # topic_num should be between 0 and 1
        similarity_scores = {}

        gratest_hidden_topics_matrix = lambda tensor1, tensor2: tensor1 if tensor1.shape[1] > tensor2.shape[
            1] else tensor2

        return self.get_CCA_similarity_matrix(first_translated, second_translated, component_num)

    def get_CCA_similarity_matrix(self, first_translated, second_translated, component_num):
        embeddings_first_2d = self.load_cached_embedding(first_translated)

        embeddings_second_2d = self.load_cached_embedding(second_translated)
        similarity_scores = {}
        #for subtitle1, tensor1 in embeddings_first_2d.items():
        #    for subtitle2, tensor2 in embeddings_second_2d.items():
        #        similarity_scores[(subtitle1, subtitle2)] = (
        #            self.CCA_similarity_score(tensor1, tensor2, component_num(tensor1, tensor2)))

        similarity_matrix = torch.tensor([[self.CCA_similarity_score(tensor1, tensor2, component_num(tensor1, tensor2)) for
                                           subtitle2, tensor2 in embeddings_second_2d.items()] for subtitle1, tensor1 in
                                          embeddings_first_2d.items()])
        if self.doPlots:
            plot_similarity_heatmap(similarity_matrix, embeddings_second_2d.keys(), embeddings_first_2d.keys(), 'CCA')
        return find_lowest_max_similarity(similarity_matrix, embeddings_first_2d.keys(), embeddings_second_2d.keys())

    def get_CCA_pairwise_sentencewise(self, first_translated, second_translated):
        if self.metric == 'pairwise-component-max':
            component_num = lambda x, y: min(x.shape[1], y.shape[1])
        else:
            component_num = lambda x, y: round(
                min(x.shape[1], y.shape[1]) * self.topic_num)  # topic_num should be between 0 and 1
        similarity_scores = {}

        gratest_hidden_topics_matrix = lambda tensor1, tensor2: tensor1 if tensor1.shape[1] > tensor2.shape[
            1] else tensor2

        return self.get_CCA_similarity_matrix(first_translated, second_translated, component_num)

    def get_CCA_score_article_to_subset(self, first_translated, second_translated):
        if self.metric == 'pairwise-component-max':
            component_num = lambda x: x.shape[1]
        else:
            component_num = lambda x: round(x.shape[1] * self.topic_num)  # topic_num should be between 0 and 1
        similarity_scores = {}

        embeddings_first_2d = self.load_cached_embedding(first_translated)
        embeddings_first_2d_stacked = torch.cat(list(embeddings_first_2d.values()), dim=1)

        embeddings_second_2d = self.load_cached_embedding(second_translated)

        similarity_vector = torch.tensor(
            [self.CCA_similarity_score(embeddings_first_2d_stacked, tensor2, component_num(embeddings_first_2d_stacked)) for tensor2 in embeddings_second_2d.values()]
        )
        #min_val, min_ind = torch.min(similarity_vector, dim=0)
        #list_of_sections = list(embeddings_second_2d.keys())
        lowest_sections = get_three_lowest_similarities(similarity_vector, embeddings_second_2d)
        if self.doPlots:
            plot_histogram(similarity_vector, 'CCA')
        return lowest_sections

    def CCA_similarity_score(self, tensor1, tensor2, component_num):
        #print("tensor1.shape: " + str(tensor1.shape[0]) + ", " + str(tensor1.shape[1]) + " tensor2.shape: " + str(
        #    tensor2.shape[0]) + ", " + str(tensor2.shape[1]))
        #getmaxdim = lambda tensor: 768 if tensor1.shape[1] > 768 else tensor.shape[1]

        tensor1_np = self.reduce_dimensionality(tensor1) #tensor1.numpy()
        tensor2_np = self.reduce_dimensionality(tensor2) #tensor2.numpy()
        if component_num > min(tensor1_np.shape[1], tensor2_np.shape[1]):
            component_num = min(tensor1_np.shape[1], tensor2_np.shape[1])

        cca = CCA(n_components=component_num, tol=1e-03, max_iter=1000, scale=False)
        contentA = tensor1 if tensor1_np.shape[1] > tensor2_np.shape[1] else tensor2_np
        contentB = tensor2 if tensor2_np.shape[1] < tensor1_np.shape[1] else tensor1_np
        cca.fit(contentA, contentB)
        tensor1_c_numpy, tensor2_c_numpy = cca.transform(contentA, contentB)
        tensor1_c = torch.from_numpy(tensor1_c_numpy).float()
        tensor2_c = torch.from_numpy(tensor2_c_numpy).float()
        cca_similarity_score = self.computeCCASimilarity(tensor1, tensor2, tensor1_c, tensor2_c, component_num).clone().detach()
        #print("cca_similarity_score: " + str(cca_similarity_score))
        return cca_similarity_score

    def computeCCASimilarity(self, A, B, tensor1_c, tensor2_c, comonentNumber=15):
        A_to_A_c = self.totalSim(A, tensor1_c, tensor2_c, comonentNumber)
        B_to_B_c = self.totalSim(B, tensor1_c, tensor2_c, comonentNumber)
        return (2 * A_to_A_c * B_to_B_c) / (A_to_A_c + B_to_B_c)

    def totalSim(self, A, A_c, B_c, comonentNumber=15):
        return (torch.max(torch.tensor([self.sim(A[:, i], A_c, B_c, comonentNumber) for i in range(A.shape[1])])))# / A.shape[0]

    def cosSim(self, A_i, P_k):
        nom = (torch.dot(A_i, P_k))
        denom = (torch.linalg.norm(A_i) * torch.linalg.norm(P_k))
        res = torch.pow((nom / denom), 2)
        return res

    def sim(self, a_i, P, Q, comonentNumber=15):
        weights = torch.tensor([torch.dot(P[:, k], Q[:, k]) / (torch.linalg.norm(P[:, k] * torch.linalg.norm(Q[:, k]))) for k in range(comonentNumber)])
        sims = (torch.tensor([self.cosSim(a_i, P[:, k]) for k in range(comonentNumber)]))
        return torch.dot(sims, weights)

    def reduce_dimensionality(self, embeddings_first_np):
        if embeddings_first_np.shape[1] > embeddings_first_np.shape[0]:
            embeddings = embeddings_first_np.numpy()
            pca = PCA(n_components=embeddings_first_np.shape[0])
            return pca.fit_transform(embeddings)
        return embeddings_first_np.numpy()


class PCAComparator(Comparator):

    def __init__(self, metric, approach, model, topic_num, splitting, doPlots) -> None:
        '''Pass'''
        super().__init__(metric=metric,
                         approach=approach,
                         #tokenizer=tokenizer,
                         model=model)
        self.topic_num = topic_num
        self.splitting = splitting
        self.doPlots = doPlots

    def get_similarity(self, first_translated, second_translated):
        if self.approach == 'subsets_to_subsets':
            return self.get_pca_similarity_paarwise(first_translated, second_translated)
        elif self.approach == 'article_to_subset':
            return self.get_pca_article_to_subsets(first_translated, second_translated)
        elif self.approach == 'article_to_article':
            return self.get_similarity_scores_per_component(first_translated.values(),
                                                            second_translated.values()), self.get_similarity_matrix(
                first_translated.keys(), second_translated.keys())
        else:
            print("Unknown metric", self.metric)
            return torch.zeros(len(first_translated), len(second_translated)),

    def get_pca_article_to_subsets(self, first_translated, second_translated):
        if self.metric == 'pairwise-component-max':
            component_num = lambda x: x.shape[1]
        else:
            component_num = lambda x: round(x.shape[1] * self.topic_num)  # topic_num should be between 0 and 1

        splitted_preprocessed_first = preprocess(first_translated)
        splitted_preprocessed_second = preprocess(second_translated)

        embeddings_first_2d = self.load_cached_embedding(first_translated)
        embeddings_first_2d_stacked = torch.cat(list(embeddings_first_2d.values()), dim=1)

        embeddings_second_2d = self.load_cached_embedding(second_translated)

        similiarity_vector = torch.tensor(
            [self.compute_similarity_score(embeddings_first_2d_stacked, tensor1, component_num(embeddings_first_2d_stacked)) for tensor1 in
             embeddings_second_2d.values()])

        #min_val, min_ind = torch.min(similiarity_vector, dim=0)
        #list_of_sections = list(embeddings_second_2d.keys())
        lowest_sections = get_three_lowest_similarities(similiarity_vector, embeddings_second_2d)
        if self.doPlots:
            plot_histogram(similiarity_vector, 'PCA')
        return lowest_sections

    def get_pca_similarity_paarwise(self, first_translated, second_translated):

        embeddings_first_2d = self.load_cached_embedding(first_translated)

        embeddings_second_2d = self.load_cached_embedding(second_translated)

        if self.metric == 'pairwise-component-max':
            component_num = lambda x: x.shape[1]
        else:
            component_num = lambda x: round(x.shape[1] * self.topic_num)  # topic_num should be between 0 and 1
        similarity_scores = {}

        similarity_matrix = torch.tensor([[self.compute_similarity_score(tensor1, tensor2, component_num(tensor1)) for subtitle2, tensor2 in embeddings_second_2d.items()] for subtitle1, tensor1 in embeddings_first_2d.items()])
        if self.doPlots:
            plot_similarity_heatmap(similarity_matrix, embeddings_second_2d.keys(), embeddings_first_2d.keys(), 'PCA')
        return find_lowest_max_similarity(similarity_matrix, embeddings_first_2d.keys(), embeddings_second_2d.keys())
        # Iterate over both dictionaries and compute similarity scores
        #'''for subtitle1, tensor1 in embeddings_first_2d.items():
        #    for subtitle2, tensor2 in embeddings_second_2d.items():
        #        similarity_score = torch.tensor(self.compute_similarity_score(tensor1, tensor2, component_num(tensor1)))
        #        similarity_scores[(subtitle1, subtitle2)] = similarity_score.item()
        ##result = torch.stack([torch.tensor(self.compute_similarity_score(tensor1, tensor2, component_num(tensor1)))
        ##                      for tensor1 in embeddings_first_2d.values()
        ##                      for tensor2 in embeddings_second_2d.values()])
        ##(result.reshape(len(splitted_preprocessed_first.keys()), len(splitted_preprocessed_second.keys()))
        #return similarity_scores, self.get_similarity_matrix(embeddings_first_2d.keys(), embeddings_second_2d.keys())'''

    def get_similarity_scores_per_component(self, first_translated, second_translated):
        embeddings_first = [torch.squeeze(self.model.get_embeddings(subset1), dim=0) for subset1 in
                            first_translated]
        embeddings_second = [torch.squeeze(self.model.get_embeddings(subset2), dim=0) for subset2 in
                             second_translated]

        embeddings_first_2d = torch.stack(embeddings_first, dim=1)
        embeddings_second_2d = torch.stack(embeddings_second, dim=0)

        # similarity_scores = self.weighted_similarity(embeddings_second_2d, components.T, weights)

        return torch.tensor(
            [self.compute_similarity_score(embeddings_first_2d, embeddings_second_2d, component_num) for component_num
             in range(1, embeddings_first_2d.shape[1] + 1)])

    def compute_similarity_score(self, embeddings_first_2d, embeddings_second_2d, component_nums):
        components, weights = self.find_components(embeddings_first_2d, component_nums)
        # print(components.shape)
        embeddings_second_2d_normalized = normalize(1.0 * embeddings_second_2d, p=2.0, dim=0)
        components_normalized = normalize(1.0 * components, p=2.0,
                                          dim=0)  # actually not needed, since U from SVD  is already normalized

        components_relevance = torch.pow((torch.matmul(components_normalized.T, embeddings_second_2d_normalized)), 2)
        components_relevance_sums = torch.max(components_relevance, dim=1)

        #components_relevance_sums_averaged = components_relevance_sums / embeddings_second_2d.shape[0]
        return torch.dot(components_relevance_sums[0], weights)
        #return torch.dot(components_relevance_sums_averaged, weights)

    def find_components(self, translated_article_vectors, component_nums):
        component_num = min(component_nums, translated_article_vectors.shape[1])
        U, s, V = torch.linalg.svd(translated_article_vectors, full_matrices=True)
        U_selected = U[:, :component_num]
        components_copy = copy.deepcopy(U_selected)
        weights = self.compute_weights(s[:component_num])
        return components_copy, weights

    def compute_weights(self, singular_values):
        eigen_values = torch.square(singular_values)
        norm = torch.sum(eigen_values)
        weights = 1.0 * eigen_values / norm
        return weights

    def weighted_similarity(self, embeddings_second, components, weights):
        print('components:   ', components.shape)
        similarities = torch.tensor(
            [self.get_reconstruction_error(component, embeddings_second) for component in components])
        return torch.dot(similarities, weights)

    def get_reconstruction_error(self, component, embeddings_second):
        similarities = torch.square(torch.tensor(
            [self.cosine_similarity(component, embedding_second) for embedding_second in embeddings_second]))
        return torch.mean(similarities)

    def cosine_similarity(self, component, embedding_second):
        if (torch.norm(component) * torch.norm(embedding_second) == 0):
            return 0
        return torch.dot(component, embedding_second) / (torch.norm(component) * torch.norm(embedding_second))


class SimpleDistanceComparator(Comparator):

    def __init__(self, metric, approach, model,
                 splitting, doPlots) -> None:  #self, metric, approach, tokenizer, model, topic_num, splitting
        '''Pass'''
        super().__init__(metric=metric,
                         approach=approach,
                         #tokenizer=tokenizer,
                         model=model)
        self.splitting = splitting
        self.doPlots = doPlots

    def get_similarity(self, first_translated_dict, second_translated_dict):
        if self.approach == 'article_to_article':
            similarity_matrix = self.get_similarity_matrix(first_translated_dict.values(), second_translated_dict.values())
            return find_lowest_max_similarity(similarity_matrix, first_translated_dict.keys(), second_translated_dict.keys())
        elif self.approach == 'subsets_to_subsets':
            embeddings_first_2d, embeddings_second_2d, similarity_matrix = self.get_simple_distance_paarwise(first_translated_dict, second_translated_dict)
            print("dimension of the sim matrix: ", similarity_matrix.shape)
            if self.doPlots:
                plot_similarity_heatmap(similarity_matrix, embeddings_first_2d.keys(), embeddings_second_2d.keys(), 'simple-cosine-distance')
            return find_lowest_max_similarity(similarity_matrix,
                                                   embeddings_second_2d.keys(), embeddings_first_2d.keys())
        elif self.approach == 'article_to_subset':
            similarity_matrix = self.get_simple_distance_article_to_subset(first_translated_dict,
                                                                           second_translated_dict)
            #min_val, min_ind = torch.min(similarity_matrix, dim=0)
            #list_of_sections = list(second_translated_dict.keys())
            lowest_sections = get_three_lowest_similarities(similarity_matrix, second_translated_dict)
            return lowest_sections


    def get_simple_distance_article_to_subset(self, first_translated_dict, second_translated_dict):
        embeddings_first_2d = self.load_cached_embedding(first_translated_dict)
        embeddings_first_2d_stacked = torch.cat(list(embeddings_first_2d.values()), dim=1)
        embeddings_second_2d = self.load_cached_embedding(second_translated_dict)
        similiarity_vector = torch.tensor([self.get_similarity_matrix_with_metric(tensor1, embeddings_first_2d_stacked) for tensor1 in embeddings_second_2d.values()])
        if self.doPlots:
            plot_histogram(similiarity_vector, 'cosine-similarity')
        return similiarity_vector


    def get_simple_distance_paarwise(self, first_translated_dict, second_translated_dict):
        embeddings_first_2d = self.load_cached_embedding(first_translated_dict)
        embeddings_second_2d = self.load_cached_embedding(second_translated_dict)
        similarity_matrix = torch.tensor([[self.get_similarity_matrix_with_metric(tensor1, tensor2) for tensor1 in embeddings_first_2d.values()] for tensor2 in embeddings_second_2d.values()])
        return embeddings_first_2d, embeddings_second_2d, similarity_matrix

    def get_similarity_matrix_with_metric(self, tensor1, tensor2):
        column_norms_first = torch.norm(tensor1, dim=0)
        column_norms_second = torch.norm(tensor2, dim=0)

        embeddings_first_2d_normalized = tensor1 / column_norms_first
        embeddings_second_2d_normalized = tensor2 / column_norms_second

        similarity_matrix = torch.matmul(embeddings_first_2d_normalized.T, embeddings_second_2d_normalized)
        return torch.mean(similarity_matrix)