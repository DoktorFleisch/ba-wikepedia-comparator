from flask import Flask, render_template, request, jsonify
import Controlling as Ctrl
import GraphComponent
import json

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/fetch_article', methods=['POST'])
def fetch_article():
    return Ctrl.fetch_article()

# could be removed later, if not needed again
@app.route('/get_en_title', methods=['POST'])
def get_en_title():
    return Ctrl.fetch_en_title()


@app.route('/compare_articles', methods=['POST'])
def compare_articles_route():
    article_name = request.form['articleName']
    recommendations = request.form['recommendations']
    selected_recommendations = json.loads(recommendations)

    comparison_results = {}
    for recommendation in selected_recommendations:
        result = Ctrl.compare_articles(article_name, recommendation)
        comparison_results[recommendation] = result

    return jsonify({'comparison_results': comparison_results})

# Old, single handed comparison

# @app.route('/compare_articles', methods=['POST'])
# def compare_articles_route():
#     article_name = request.form['articleName']
#     en_title = request.form['enTitle']
#     comparison_results = Ctrl.compare_articles(article_name, en_title)
#
#     return jsonify({'comparison_results': comparison_results})

# Not used?
@app.route('/get_section_content', methods=['POST'])
def get_section_content():
    article_name = request.form['articleName']
    section_names = request.form['sectionNames'].split(',')
    content = Ctrl.get_section_content(article_name, section_names)

    return jsonify({'content': content})


@app.route('/get_recommendations', methods=['POST'])
def get_recommendations():
    article_name = request.form['articleName']
    article_en_name = Ctrl.get_title_en(article_name)
    wiki = Ctrl.get_wiki('en')
    article = Ctrl.get_article_object_en(article_en_name)
    graph = GraphComponent.make_graph_optimized(article, wiki)
    recommendations = GraphComponent.pagerank(graph)

    return jsonify({'recommendations': recommendations})


if __name__ == '__main__':
    app.run()
