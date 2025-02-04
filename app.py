from flask import Flask, render_template, request, jsonify
import Wikipedia as Wp
import GraphComponent
import json

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/fetch_article', methods=['POST'])
def fetch_article():
    return Wp.fetch_article()

# could be removed later, if not needed again
@app.route('/get_en_title', methods=['POST'])
def get_en_title():
    return Wp.fetch_en_title()


@app.route('/translate_section/<string:recommendation>/<string:result>/<string:articleName>', methods=['GET'])
def translate_section(recommendation, result, articleName):
    section_title = result.split(",")[0]
    section_title_de = result.split(",")[1]

    # Abschnittstext abrufen
    section_text = Wp.get_section_content_de(articleName, section_title_de)

    # Abschnittstext übersetzen
    translation = Wp.translate_section(recommendation, section_title)

    # Template rendern und Variablen übergeben
    return render_template('translate_section.html', section_text=section_text, translation=translation)

@app.route('/compare_articles', methods=['POST'])
def compare_articles_route():
    article_name = request.form['articleName']
    recommendations = request.form['recommendations']
    selected_recommendations = json.loads(recommendations)

    comparison_results = {}
    for recommendation in selected_recommendations:
        # this is a list of english section names
        results = Wp.compare_articles(article_name, recommendation)
        # this is a tuple with (recommended section, score)
        section_results = Wp.compare_sections(article_name, recommendation, results)  # Pass results as section_names

        # Prepare the formatted results
        formatted_results = []
        for i, result in enumerate(results):
            if i < len(section_results):  # Ensure no index out of range
                recommended_section, _, score = section_results[i]
                formatted_results.append(f"{result}, {recommended_section}, {score}")
            else:
                formatted_results.append(f"{result} No matching section found")

        comparison_results[recommendation] = formatted_results

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
    content = Wp.get_sections_content(article_name, section_names)

    return jsonify({'content': content})


@app.route('/get_recommendations', methods=['POST'])
def get_recommendations():
    article_name = request.form['articleName']
    article_en_name = Wp.get_title_en(article_name)
    wiki = Wp.get_wiki('en')
    article = Wp.get_article_object_en(article_en_name)
    graph = GraphComponent.make_graph_optimized(article, wiki)
    recommendations = GraphComponent.pagerank(graph)

    return jsonify({'recommendations': recommendations})


if __name__ == '__main__':
    app.run()
