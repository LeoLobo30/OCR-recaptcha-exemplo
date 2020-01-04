import flask
from flask_cors import CORS
import get_chars, solver

app = flask.Flask(__name__)
CORS(app)

@app.route('/')
def start():
    return flask.render_template('index.html')

@app.route('/imageSend', methods=['GET'])
def solverImage():
    nameImgSolver = flask.request.args.get('imgName')

    get_chars.getChars(nameImgSolver)

    solverlist = solver.solver()

    listToString = ''.join(map(str, solverlist))
    
    return listToString