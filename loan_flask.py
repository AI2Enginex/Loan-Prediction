
import pickle 
from flask import render_template
from keras.models import load_model
from flask import Flask, request , flash


class Parameters:

    def __init__(self):

        self.sclar = pickle.load(open('./scaler.pkl','rb'))
        self.ann_model = load_model("./loan_prediction.h5")
        self.encoded_vectors = {'Yes' : 1 , "No" : 0 , 'Married' : 1 , "Un Married" : 0 ,
                                "Male" : 1 , "Female" : 0,"0" : [0,0,0] , "1" : [1,0,0] , "2" : [0,1,0] , "3+" : [0,0,1],
                                "Graduate" : 1 , "Not Graduate" : 0 , "Self Employed" : 1 , "Service Based" : 0,
                                "Rural" : [0,0] , "Urban" : [0,1] , "Semiurban" : [1,0]}
        

class FlaskApp(Parameters):

    def __init__(self):

        super().__init__()
        self.app = Flask(__name__ , template_folder = 'templates')
        self.app.secret_key = 'batman'

    def home(self):

        return render_template("index.html")
    
    def predict(self):

        try:
            applicant_income = request.form['applicantincome']
            co_applicant_income = request.form['coapplicantincome']
            loan_amount = request.form['loanamount']
            loan_amount_term = request.form['lat']
            credit_history = request.form['credithistory']
            married = request.form['maratialstatus']
            gender = request.form['gender']
            dependents = request.form['dependents']
            education = request.form['education']
            self_employee_status = request.form['empstatus']
            property_status = request.form['property']
            
            input_arr = self.sclar.transform([[int(applicant_income) , int(co_applicant_income) , int(loan_amount) ,int(loan_amount_term), 
                                           self.encoded_vectors[credit_history],self.encoded_vectors[gender] , 
                                           self.encoded_vectors[married],self.encoded_vectors[education] , 
                                           self.encoded_vectors[self_employee_status]] + self.encoded_vectors[dependents] + self.encoded_vectors[property_status]])
            
            
            result = self.ann_model.predict(input_arr)
            if result < 0.45:
              result = "chances of getting loan is very low"
            elif result >= 0.45 and result <= 0.74:
              result = "chances are high"
            elif result > 0.75:
                result = "chances are very high"
            return(render_template("index.html",result = result))
        
        except Exception:
            
            flash(category="error",message="fill the form correctly")
            return render_template("index.html")

    def run_app(self):

        self.app.add_url_rule('/', methods=['GET','POST'] , view_func=self.home)
        self.app.add_url_rule('/predict', methods=['GET','POST'] , view_func=self.predict)
        self.app.run(debug=True)



if __name__ == '__main__':

    fa = FlaskApp()
    fa.run_app()