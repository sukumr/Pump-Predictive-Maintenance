import pandas as pd
from joblib import load
import io
from flask import Flask, render_template, request, session, send_file
from flask_session import Session
from tempfile import mkdtemp
import os

app = Flask(__name__)

# Ensure templates are auto-reloaded
app.config["TEMPLATES_AUTO_RELOAD"] = True

# Ensure responses aren't cached
@app.after_request
def after_request(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"
    return response


# Configure session to use filesystem (instead of signed cookies)
app.config["SESSION_FILE_DIR"] = mkdtemp()
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# sensors
FINAL_SENSORS = [
    "sensor_00",
    "sensor_04",
    "sensor_06",
    "sensor_07",
    "sensor_08",
    "sensor_09",
    "sensor_10",
    "sensor_11",
    "sensor_12",
]

# loading the minimax scaler
SCALER = load("model/minimax_scaler.joblib")
# loading the trained model
MODEL = load("model/random_forest.joblib")

# home page
@app.route("/")
def index():
    return render_template("index.html")


# handle data page
@app.route("/load_data", methods=["GET"])
def load_data():
    return render_template("load_data.html")


# carry out single point prediction
@app.route("/predict", methods=["POST"])
def predict():
    data_dict = request.form.to_dict()

    count = 0

    # to check if all the inputs are empty
    for v in data_dict.values():
        # check for valid inputs
        if v == "":
            count += 1
        elif v.isnumeric():
            continue
        else:
            return render_template(
            "load_data.html", type="danger", message="Please enter a valid input"
        )

    if count == 9:
        return render_template(
            "load_data.html", type="danger", message="All the values are missing!"
        )

    df = pd.DataFrame([data_dict.values()], columns=data_dict.keys())

    # replacing empty fields with -1
    df = df.replace("", -1)

    # minimax scaling
    df = SCALER.transform(df)
    # prediction
    y = MODEL.predict(df)
    # getting probability values
    prob = MODEL.predict_proba(df)
    status = int(y[0])
    prob = prob[0][status]

    # if class is 0 (Normal)
    if y[0] == 0:
        return render_template(
            "load_data.html",
            type="success",
            message=f"Pump will be okay with probability {round(prob, 3)}",
        )
    # if class is 1 (Broken)
    else:
        return render_template(
            "load_data.html",
            type="danger",
            message=f"Pump is going to fail with probability {round(prob, 3)}",
        )


# carry out multi point prediction
@app.route("/multi_predict", methods=["POST"])
def multi_predict():

    file = request.files["test_file"]

    # check if a file is uploaded
    if file:
        # check file extention
        if file.filename.endswith("csv"):
            # create a path
            file_path = os.path.join("test", file.filename)
            # save the file
            file.save(file_path)
            try:
                # parse the file
                data = pd.read_csv(file_path)
            # to handle parser error
            except pd.errors.ParserError:
                # delete the file
                os.remove(file_path)
                # show the warning message
                return render_template(
                    "load_data.html",
                    type="danger",
                    message=f"Please load a csv file",
                )
            data_columns = list(data.columns)

            # to check if the file contains the required sensors
            for sensor in FINAL_SENSORS:
                if sensor not in data_columns:
                    os.remove(file_path)
                    return render_template(
                        "load_data.html",
                        type="danger",
                        message=f"File is missing below specified sensors",
                    )

            data_df = dict()

            # fill missing values with -1
            for sensor in FINAL_SENSORS:
                data[sensor].fillna(-1, inplace=True)
                data_df[sensor] = data[sensor]

            data_df = pd.DataFrame(data_df)

            # normalizing the data
            X = SCALER.transform(data_df)
            # prediction
            status = MODEL.predict(X)
            # getting probability values
            probability = MODEL.predict_proba(X)

            status_list = []
            probability_list = []

            # creating list of all the predictions
            for pred, pro in zip(status, probability):
                y = int(pred)
                p = pro[y]
                status_list.append(y)
                probability_list.append(p)

            data_df["prediction"] = status_list
            data_df["probability"] = probability_list

            # writing a csv file
            session["data_df"] = data_df.to_csv(index=False, header=True, sep=";")
            os.remove(file_path)
            # if the file contains single data point
            if data_df.shape[0] == 1:
                if y == 0:
                    # if the prediction = 0
                    return render_template(
                        "load_data.html",
                        type="success",
                        message=f"Pump will be okay with probability {round(p, 3)}",
                    )
                else:
                    # if the prediction = 1
                    return render_template(
                        "load_data.html",
                        type="danger",
                        message=f"Pump is going to fail with probability {round(p, 3)}",
                    )
            else:
                # if multipoint prediction
                return render_template(
                    "load_data.html",
                    type="primary",
                    message=f"Done, Ready for download.",
                )
        else:
            # if no file is uploaded
            return render_template(
                "load_data.html",
                type="danger",
                message=f"Please upload a csv file",
            )
    else:
        # if file extension is not csv
        return render_template(
            "load_data.html", type="danger", message=f"Select a file for prediction."
        )


# to download a prediction results
@app.route("/download", methods=["POST"])
def download():
    if "data_df" in session:
        # https://stackoverflow.com/questions/62823361/download-dataframe-as-csv-from-flask
        csv = session["data_df"]
        # Create a string buffer
        buf_str = io.StringIO(csv)
        # Create a bytes buffer from the string buffer
        buf_byt = io.BytesIO(buf_str.read().encode("utf-8"))
        # clear session
        session.clear()
        # Return the CSV data as an attachment
        return send_file(
            buf_byt,
            mimetype="text/csv",
            as_attachment=True,
            attachment_filename="prediction.csv",
        )
    else:
        # if downloding without prediction
        return render_template(
            "load_data.html",
            type="danger",
            message=f"Select a file and carry out prediction.",
        )


if __name__ == "__main__":
    # app.run(debug=True)
    app.run(host="0.0.0.0", port=5000)
