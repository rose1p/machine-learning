from flask import Flask, render_template
from scoerRouter import score

app = Flask(__name__)
app.register_blueprint(score, url_prefix='/score')

@app.route('/')
def index():
    return render_template('index.html')

if __name__== '__main__':
    app.run(port=5000, debug=True)