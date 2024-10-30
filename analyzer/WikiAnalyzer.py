import json

import torch

from .ArticleProcessor import ArticleProcessor
from .Comparator import Comparator
import numpy as np
import os


def store_translations(translation_function, base_lang, source_lang, content, json_file_path):
    content_translated = {key: translation_function(value) for key, value in content.items()}
    if not (base_lang == source_lang):
        if not os.path.exists(json_file_path):
            with open(json_file_path, 'w') as json_file:
                json.dump(content_translated, json_file, indent=4)
    return content_translated


class WikiAnalyzer(object):
    '''
    The class for analyzing the wikipedia articles using different methods and languages
    '''

    def __init__(self,
                 article_processor: ArticleProcessor,
                 comparator: Comparator) -> None:
        '''
        :param comparator:     choose the appropriate comparator: Base, PCA, WMD, etc.

        '''
        self.article_processor = article_processor
        self.comparator = comparator

    def analyze(self):
        first_content = self.article_processor.extract_sections(True)
        second_content = self.article_processor.extract_sections(False)

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        directory_path_first = os.path.join(project_root, "wiki_jsons", "translation" + self.article_processor.first_lang + "-" + self.article_processor.second_lang)
        directory_path_second = os.path.join(project_root, "wiki_jsons", "translation" + self.article_processor.second_lang + "-" + self.article_processor.first_lang)
        path_for_first_translation = os.path.join(project_root, "wiki_jsons",
                                                  "translation" + self.article_processor.first_lang + "-" + self.article_processor.second_lang,
                                                  self.article_processor.first_article + self.article_processor.first_lang + "-" + self.article_processor.second_lang + ".json")

        path_for_second_translation = os.path.join(project_root, "wiki_jsons",
                                                   "translation" + self.article_processor.second_lang + "-" + self.article_processor.first_lang,
                                                   self.article_processor.second_article + self.article_processor.second_lang + "-" + self.article_processor.first_lang + ".json")

        # Create the complete directory structure immediately
        os.makedirs(directory_path_first, exist_ok=True)
        os.makedirs(directory_path_second, exist_ok=True)

        translation_function_first = self.article_processor.get_translation_function(path_for_first_translation, True)
        translation_function_second = self.article_processor.get_translation_function(path_for_second_translation,
                                                                                      False)

        first_content_translated = store_translations(translation_function_first, self.article_processor.base_lang,
                                                      self.article_processor.first_lang, first_content,
                                                      path_for_first_translation)
        second_content_translated = store_translations(translation_function_second, self.article_processor.base_lang,
                                                       self.article_processor.second_lang, second_content,
                                                       path_for_second_translation)

        return self.comparator.get_similarity(first_content_translated, second_content_translated)

        # return similarity_matrix, weights

    def get_unpresent_content(self):
        similarity_matrix, weights = self.analyze()
        if (self.comparator.approach == 'subset_to_subset'):
            return
