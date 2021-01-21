from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates

import uvicorn
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

app = FastAPI()
templates = Jinja2Templates(directory="templates/")
model = pickle.load(open('random_forest_model.pkl' , 'rb'))

@app.get('/')
def index(request:Request):
    return templates.TemplateResponse('index.html',context={'request': request})
# def index():
#     return render('index.html')

standard_to = StandardScaler()
@app.post('/predict')
def predict(request:Request , Year: int = Form(...),
            Present_Price:float = Form(...) ,  Kms_Driven:int = Form(...) ,
            Owner:int = Form(...) ,Fuel_Type_Petrol = Form(...) ,
            Seller_Type_Individual = Form(...) , Transmission_Mannual = Form(...),
            ):
    Fuel_Type_Diesel=0

    if (Fuel_Type_Petrol == 'Petrol'):
        Fuel_Type_Petrol = 1
        Fuel_Type_Diesel = 0
    else:
        Fuel_Type_Petrol = 0
        Fuel_Type_Diesel = 1

    Year = 2020 - Year

    if (Seller_Type_Individual == 'Individual'):
        Seller_Type_Individual = 1
    else:
        Seller_Type_Individual = 0

    if (Transmission_Mannual == 'Mannual'):
        Transmission_Mannual = 1
    else:
        Transmission_Mannual = 0
    # print(Year)
    # print(Present_Price)
    Kms_Driven2 = np.log(Kms_Driven)
    # print(Owner , Fuel_Type_Petrol , Seller_Type_Individual , Transmission_Mannual , Kms_Driven2)

    prediction = model.predict([[Present_Price,Kms_Driven2,Owner,Year, Fuel_Type_Diesel, Fuel_Type_Petrol, Seller_Type_Individual, Transmission_Mannual]])
    output = round(prediction[0], 2)
    # json_compatible_item_data = jsonable_encoder(output)
    # return JSONResponse(content=json_compatible_item_data)

    if output<0:
        return templates.TemplateResponse('index.html',
                                          context={'request': request,
                                                   'prediction_text': "Sorry you cannot sell this car"})
    else:
        return templates.TemplateResponse('index.html',
                                          context={'request': request,
                                                   'prediction_text': "You Can Sell The Car at {} lakhs".format(output)})
if __name__ == '__main__':
    uvicorn.run(app , host='127.0.0.1' , port=8080)