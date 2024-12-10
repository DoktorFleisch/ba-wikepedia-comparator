import wikipediaapi
from flask import request, jsonify
from analyzer.Translator import FaceBookTranslatorProvider
from analyzer.Comparator import PCAComparator
from analyzer.WikiAnalyzer import WikiAnalyzer
from analyzer.EmbeddingService import SentenceTransformerEmbeddingService
from analyzer.ArticleProcessor import ArticleProcessor

user_agent = 'ba-thesis-comperator (daniel.warkus@hhu.de)'
wiki_api_de = wikipediaapi.Wikipedia(user_agent=user_agent, language='de')
wiki_api_en = wikipediaapi.Wikipedia(user_agent=user_agent, language='en')

translator_de_en = FaceBookTranslatorProvider("de", "en")
translator_en_de = FaceBookTranslatorProvider("en", "de")
sentence_transformer_service = SentenceTransformerEmbeddingService('sentence-transformers/all-mpnet-base-v2')


def get_wiki(language):
    if language == 'de':
        return wiki_api_de
    elif language == 'en':
        return wiki_api_en
    else:
        return None

def get_section_content(article_name, section_names):
    page = wiki_api_en.page(article_name)
    content = []
    for section_name in section_names:
        section = page.section_by_title(section_name)
        if section:
            content.append(section.text)
        else:
            content.append(None)
    return content


def compare_articles(article1, article2):
    article_processor = ArticleProcessor(article1, article2, 'de', 'en',
                                         'en', translator_de_en)
    comparator = PCAComparator('pairwise', 'article_to_subset', sentence_transformer_service, 0.5, 'sentence-wise',
                               False)
    wiki_analyzer = WikiAnalyzer(article_processor, comparator)

    results = wiki_analyzer.analyze()

    section_results = []

    for section in results:
        section_results.append(section[0])

    return section_results


def get_title_en(article_name):
    title = wiki_api_de.page(article_name)
    en_title = title.langlinks['en'].title
    return en_title


def get_article_de(article_name):
    page = wiki_api_de.page(article_name)
    if page.exists():
        return {'title': page.title, 'summary': page.text}
    else:
        return None


def get_article_object_en(article_name):
    page = wiki_api_en.page(article_name)
    if page.exists():
        return page
    else:
        return None


def fetch_article():
    article_name = request.form['articleName']
    page_content = get_article_de(article_name)
    if page_content:
        return jsonify(page_content)
    else:
        return jsonify({'error': 'Article not found'}), 404


def fetch_en_title():
    article_name = request.form['articleName']
    en_title = get_title_en(article_name)
    return jsonify({'en_title': en_title})
