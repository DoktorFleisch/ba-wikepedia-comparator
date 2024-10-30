#import wikipedia
import wikipediaapi
import os
import json
import re
from .Translator import Translator  # Corrected import statement
import nltk
from nltk.corpus import stopwords
import string

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('stopwords')


def split_string(string, length):
    return [string[i:i + length] for i in range(0, len(string), length)]

def filter_stop_words(text, lang):
    # Get the list of stop words from nltk
    stop_words = set(stopwords.words('english' if lang == 'en' else 'german')) if lang in ['en', 'de'] else set()

    #stop_words = set(stopwords.words('english'))

    # Tokenize the text into words and punctuation
    words = nltk.word_tokenize(text)

    # Filter out the stop words
    filtered_words = [word for word in words if word.lower() not in stop_words or word in string.punctuation]

    # Reconstruct the text
    filtered_text = " ".join(filtered_words)

    # Correct spacing around punctuation
    filtered_text = filtered_text.replace(" ,", ",").replace(" .", ".").replace(" !", "!").replace(" ?", "?")
    filtered_text = filtered_text.replace(" ;", ";").replace(" :", ":").replace(" '", "'")

    return filtered_text


def extract_sections(article_title, language):
    user_agent = 'ba-thesis-comperator (daniel.warkus@hhu.de)'

    # Get the project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Create the complete directory path
    directory_path = os.path.join(project_root, "wiki_jsons", "original_content", language)
    desired_file_path = os.path.join(directory_path, f"{article_title}.json")

    # Create the complete directory structure immediately
    os.makedirs(directory_path, exist_ok=True)

    stop_section_names = ['Einzelnachweise', 'Weiterführende Literatur', 'Notes', 'Quellen',
                         'Weblinks', 'Literatur', 'External links', 'Further reading', 'Sources',
                         'See also', 'References', 'Further reading']

    if os.path.exists(desired_file_path):
        with open(desired_file_path, "r") as json_file:
            loaded_data = json.load(json_file)
    else:
        wikipedia = wikipediaapi.Wikipedia(user_agent=user_agent, language=language)
        page = wikipedia.page(article_title)
        loaded_data = {}
        for section in page.sections:
            if section.text != '' and section.title not in stop_section_names:
                loaded_data[section.title] = filter_stop_words(section.text, "de")
                for subsection in section.sections:
                    if subsection.text != '' and subsection.title not in stop_section_names:
                        loaded_data[subsection.title] = filter_stop_words(subsection.text, "de")
        with open(desired_file_path, 'w') as json_file:
            json.dump(loaded_data, json_file, indent=4)
        # (section for section in re.split(r'\n\n==+ [^=]+ ==+\n', wikipedia.page(article_title).content))
        # return (value for value in loaded_data.values())
    return loaded_data


class ArticleProcessor:
    '''
    this class preprocess the article
    '''

    def __init__(self,
                 first_article: str,
                 second_article: str,
                 first_lang: str,
                 second_lang: str,
                 base_lang: str,
                 translator: Translator
                 ) -> None:
        '''
        :param first_article:  the Wikipedia name of the first artcle
        :param second_article: the Wikipedia name of the second artcle
        :param first_lang:     this parameter represents the language of the first Wikipedia article
        :param second_lang:    this parameter represents the language of the second Wikipedia article
        :param base_lang:      it specifies the language into which one of the articles will be translated for comparison purposes
        :param translator_tokenizer: translator tokenizer: FSMTTokenizer, MarianTokenizer, etc.
        :param translator_model: translator model: FSMTForConditionalGeneration, MarianMTModel, etc.
        '''
        self.first_article = first_article
        self.second_article = second_article
        self.first_lang = first_lang
        self.second_lang = second_lang
        self.base_lang = base_lang
        self.translator = translator

    def translate_or_return(self, content, base_lang, source, dest):
        if base_lang == source:
            return content
        if len(content) > 2000:
            splittedstr = [filter_stop_words(self.translator.translate_text(el), self.translator.dest_lang) for el in split_string(content, 2000)]
            return "".join(splittedstr)
        return filter_stop_words(self.translator.translate_text(content), self.translator.dest_lang)

    def extract_sections(self, isFirst):
        return extract_sections(self.first_article, self.first_lang) if isFirst else extract_sections(
            self.second_article, self.second_lang)

    def get_text_in_source_lang(self, content, isFirst, json_file):
        content_in_base_lang = self.translate_or_return(content, self.base_lang, source=self.first_lang,
                                 dest=self.second_lang) if isFirst else self.translate_or_return(content,
                                                                                                 self.base_lang,
                                                                                                 source=self.second_lang,
                                                                                                 dest=self.first_lang)
        return content_in_base_lang


    def get_translation_function(self, json_file, isFirst):
        if isFirst and self.base_lang == self.first_lang:
            return lambda subset: subset
        if not isFirst and self.base_lang == self.second_lang:
            return lambda subset: subset
        if os.path.exists(json_file):
            with open(json_file, 'r') as f:
                translations = json.load(f)
                return lambda subset: translations.get(subset, self.get_text_in_source_lang(subset, isFirst, json_file))
        else:
            return lambda subset: self.get_text_in_source_lang(subset, isFirst, json_file)