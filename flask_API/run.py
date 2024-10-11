from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/index')
def index():
    return "Hello, World!"

@app.route('/home', methods=['POST'])
def home():

    age1 = request.json['age1']
    if not age1:
        return jsonify({
            'status': 'error',
            'message': 'age1 is required'
        })

    print(age1)
    res = jsonify({
        'status': 'success'
    })
    return res



if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)

