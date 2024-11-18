from flask import Flask, render_template, request, jsonify
import Controlling as ctrl

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/fetch_article', methods=['POST'])
def fetch_article():
    return ctrl.fetch_article()

@app.route('/get_en_title', methods=['POST'])
def get_en_title():
    return ctrl.fetch_en_title()

@app.route('/compare_articles', methods=['POST'])
def compare_articles_route():
    article_name = request.form['articleName']
    en_title = request.form['enTitle']
    comparison_results = ctrl.compare_articles(article_name, en_title)

    return jsonify({'comparison_results': comparison_results})

@app.route('/get_section_content', methods=['POST'])
def get_section_content():
    article_name = request.form['articleName']
    section_names = request.form['sectionNames'].split(',')
    content = ctrl.get_section_content(article_name, section_names)

    return jsonify({'content': content})

if __name__ == '__main__':
    app.run()