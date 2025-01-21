from flask import Flask
from server.api import app as resume_service

app = Flask(__name__)
app.register_blueprint(resume_service)
print("App init...")
@app.route('/')
def hello_world():
    return 'Hello World!'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005)