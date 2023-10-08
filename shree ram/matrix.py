# from flask import Flask , render_template, request
# import pickle
# import numpy as np

# model = pickle.load(open("C:\\Users\\Dell\\complete web development\\shree ram\\model.pkl", "rb"))

# app = Flask(__name__)

# @app.route('/')
# def index():
#     return render_template('matrix.html')

# @app.route('/services')
# def services():
#     return render_template('matrix_services01.html')

# @app.route('/after', methods=['POST'])
# def after():
#     data1 = request.form['Age']
#     data2 = request.form['Gender']
#     data3 = request.form['Stream']
#     data4 = request.form['Internships']
#     data5 = request.form['CGPA']
#     data6 = request.form['Back Logs']
#     data8 = request.form['Hostel']
#     arr = np.array([[data1, data2, data3, data4, data5, data6, data8]])
#     pred = model.predict(arr)
#     return render_template('after.html', data=pred)

# @app.route('/about')
# def about():
#     return render_template('matrix_about_us.html')

# if __name__=="__main__":
#     app.run(debug=True) 


from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd  # Import pandas for one-hot encoding

model = pickle.load(open("C:\\Users\\Dell\\complete web development\\shree ram\\model.pkl", "rb"))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('matrix.html')


@app.route('/services')
def services():
    return render_template('matrix_services02.html')

# ... Other routes ...

@app.route('/after', methods=['POST'])
def after():
    data1 = float(request.form['Age'])
    data02 = str(request.form['Gender'])
    if (data02=='male'):
        data2=1

    if (data02=='female'):
        data2=0
    # data2 = float(request.form['Gender'])  # Get the gender as a string
    data03 = str(request.form['Stream'])
    if (data03=='Electronics And Communication'):
        data3=1

    if (data03=='Computer Science'):
        data3=2

    if (data03=='Information Technology'):
        data3=3

    if (data03=='Mechanical'):
        data3=4

    if (data03=='Electrical'):
        data3=5

    if (data03=='Civil'):
        data3=6
    # data3 = float(request.form['Stream'])
    data4 = float(request.form['Internships'])
    data5 = float(request.form['CGPA'])
    data6 = float(request.form['Back Logs'])
    data8 = float(request.form['Hostel'])
    
    # # Perform one-hot encoding for 'Gender'
    # gender_encoded = pd.get_dummies(data2_gender, prefix='Gender')
    # stream_encoded = pd.get_dummies(data3, prefix='Stream')

    # # Create an input array with the one-hot encoded gender
    arr = np.array([[data1,data2,data3, data4, data5, data6, data8]])
    # arr = np.hstack((arr, gender_encoded))
    # arr = np.hstack((arr, stream_encoded))  # Append one-hot encoded gender

    pred = model.predict(arr)
    return render_template('after.html', data=pred)

@app.route('/about')
def about():
    return render_template('matrix_about_us.html')

@app.route('/output02')
def output02():
    return render_template('matrix_output02.html')

@app.route('/output01')
def output01():
    return render_template('matrix_output01.html')

# ... Other routes ...

if __name__ == "__main__":
    app.run(debug=True)
