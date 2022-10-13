import imp
from flask import Flask,render_template,request
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])

def predict():
    if request.method == 'POST':
        preg = request.form['preg']
        glu= request.form['glu']
        bp = request.form['bp']
        skin = request.form['skin']
        ins = request.form['ins']
        bmi = request.form['bmi']
        dpdf = request.form['dpdf']
        age = request.form['age']

        data = [[float(preg),float(glu),float(bp),float(skin),float(ins),float(bmi),float(dpdf),float(age)]]
        lr = pickle.load(open('diabetes.pkl','rb'))
        prediction = lr.predict(data)[0]
        
    return render_template('index.html',prediction=prediction)
if __name__=="__main__":
    app.run(debug=True)
