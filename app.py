from flask import Flask,request,render_template,jsonify
from src.pipeline.prediction_pipeline import CustomData,PredictPipeline
import numpy as np

application=Flask(__name__)

app=application


@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])

def predict_datapoint():
    if request.method=='GET':
        return render_template('form.html')
    
    else:
        restaurant_latitude = request.form.get('Restaurant_latitude'),
        if isinstance(restaurant_latitude, tuple):
            restaurant_latitude = restaurant_latitude[0]
            restaurant_latitude = float(restaurant_latitude)

        restaurant_longitude = request.form.get('Restaurant_longitude'),
        if isinstance(restaurant_longitude, tuple):
            restaurant_longitude = restaurant_longitude[0]
            restaurant_longitude = float(restaurant_longitude)

        delivery_location_latitude = request.form.get('Delivery_location_latitude'),
        if isinstance(delivery_location_latitude, tuple):
            delivery_location_latitude = delivery_location_latitude[0]
            delivery_location_latitude = float(delivery_location_latitude)

        delivery_location_longitude = request.form.get('Delivery_location_longitude'),
        if isinstance(delivery_location_longitude, tuple):
            delivery_location_longitude = delivery_location_longitude[0]
            delivery_location_longitude = float(delivery_location_longitude)

        ordered_date = request.form.get('ordered_date'),
        if isinstance(ordered_date, tuple):
            ordered_date = ordered_date[0]
            ordered_date = str(ordered_date)

        data=CustomData(
            age=int(request.form.get('age')),
            ratting = float(request.form.get('ratting')),
            restaurant_latitude = restaurant_latitude,
            restaurant_longitude = restaurant_longitude,
            delivery_location_latitude = delivery_location_latitude,
            delivery_location_longitude = delivery_location_longitude,
            weather = request.form.get('weather'),
            traffic = request.form.get('traffic'),
            vehicle_condition = int(request.form.get('vehicle_condition')),
            order_type = request.form.get('order_type'),
            type_of_vehicle = request.form.get('type_of_vehicle'),
            multiple_deliver = int(request.form.get('multiple_deliver')),
            festival = request.form.get('festival'),
            city = request.form.get('city'),
            ordered_date = ordered_date,
            time_orderd = str(request.form.get('time_orderd')),
            time_order_picked = str(request.form.get('time_order_picked')),
        )

        final_new_data=data.get_data_as_dataframe()
        predict_pipeline=PredictPipeline()
        pred=predict_pipeline.predict(final_new_data)

        results=np.round(pred[0],2)

        return render_template('results.html',final_result=results)


if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True)

