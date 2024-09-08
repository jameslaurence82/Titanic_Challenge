
from flask import Flask, request

# Initialize the Flask app
app = Flask(__name__)

# Define an API endpoint that accepts POST requests
@app.route('/api', methods=['POST'])
def say_hello():
    # Parse incoming JSON data
    data = request.get_json(force=True)
    # Extract 'name' from the JSON and return a greeting
    name = data['name']
    return f'hello {name}'

# Run the app on port 10001 with debug mode enabled
if __name__ == '__main__':
    app.run(port=10001, debug=True)
