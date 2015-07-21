from flask import Flask, Response
import json
from random import randint
from kmeans import get_heatmap_data
try:
    from flask.ext.cors import CORS, cross_origin  # The typical way to import flask-cors
except ImportError:
    # Path hack allows examples to be run without installation.
    import os
    parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.sys.path.insert(0, parentdir)

    from flask.ext.cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)

@app.route('/dummy', methods=['GET'])
@cross_origin() # allow all origins all methods.
def dummy():
    maxX = 420
    minX = 22
    maxY = 220
    minY = 60

    data = []
    # Manually generate data randomly
    # for _ in range(50):
    #     xval = randint(minX, maxX)
    #     yval = randint(minY, maxY)
    #     valval = randint(0, 100)
    #     point = {'x': xval, 'y': yval, 'value': valval}
    #     data.append(point)

    # Pull heatmap data from kmeans
    data = get_heatmap_data()


    res = Response(json.dumps(data), mimetype='application/json')
    res.headers['Access-Control-Allow-Credentials'] = 'true'
    return res

if __name__ == '__main__':
    app.run(debug=True, port=1234)
