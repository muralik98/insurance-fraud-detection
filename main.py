from flask import Flask, request, render_template, Response, send_file
import os
from flask_cors import CORS
import PredictPipeline
import TrainPipeline

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)

@app.route("/", methods=['GET'])
def home():
    return render_template('index3.html')

@app.route("/predict", methods=['POST'])
def predictRouteClient():
    try:
        data_path = None
        if request.is_json:
            data_path = request.json.get('filepath')
        elif 'filepath' in request.form:
            data_path = request.form.get('filepath')
        else:
            return Response("Error: 'filepath' is required!", status=400)

        if data_path:
            print(f"Received filepath: {data_path}")  # Debugging line
            pipe = PredictPipeline.PredictPipeline(data_path)
            output_filepath = pipe.predict_pipe()
            #output_file = './artifacts/FinalPredictedOutput/final_predicted_output.csv'
            return send_file(output_filepath, as_attachment=True)
        else:
            return Response("Error: 'filepath' is not provided!", status=400)

    except Exception as e:
        return Response(f"Error Occurred! {e}", status=500)

@app.route("/train", methods=['POST'])
def trainRouteClient():
    try:
        data_path = None
        if request.is_json:
            data_path = request.json.get('folderPath')
        elif 'folderPath' in request.form:
            data_path = request.form.get('folderPath')
        else:
            return Response("Error: 'folderPath' is required!", status=400)

        if data_path:
            print(f"Received folderPath: {data_path}")  # Debugging line
            pipe = TrainPipeline.TrainingPipeline(data_path)
            pipe.train_pipe()
            return Response("Training successful!!")
        else:
            return Response("Error: 'folderPath' is not provided!", status=400)

    except Exception as e:
        return Response(f"Error Occurred! {e}", status=500)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5001))
    app.run(port=port, debug=True)
