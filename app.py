import os
from flask import Flask, render_template, request
import torch
from models.simple3dcnn import Simple3DCNN
from utils import preprocess_video
from torchvision import transforms
# from models.simple3dcnn import Simple3DCNN

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load classes and model
classes = sorted(os.listdir("dataset/UCF-101"))  # Update if needed
model = Simple3DCNN(num_classes=len(classes))
model.load_state_dict(torch.load("saved_models/model.pth", map_location=torch.device('cpu')))
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
])

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["video"]
        if file:
            path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(path)
            tensor = preprocess_video(path, transform)
            with torch.no_grad():
                output = model(tensor)
                pred = output.argmax(1).item()
                label = classes[pred]
            return render_template("result.html", label=label, video=file.filename)
    return render_template("index.html")

@app.route("/result")
def result():
    return render_template("result.html")

if __name__ == "__main__":
    app.run(debug=True)


model = Simple3DCNN(num_classes=len(classes))
model.load_state_dict(torch.load("saved_models/model.pth"))
model.eval()  # Set the model to evaluation mode
