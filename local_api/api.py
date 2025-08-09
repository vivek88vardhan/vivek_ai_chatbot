from flask import Flask, jsonify

app = Flask(__name__)

@app.route("/data")
def get_local_data():
    return jsonify({"info": "Local weather is sunny and 75°F"})

if __name__ == "__main__":
    app.run(port=5000)
