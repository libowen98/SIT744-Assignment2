import os
import readline
from flask import Flask, request, redirect, url_for, send_from_directory, render_template
from werkzeug.utils import secure_filename
import numpy as np
from model.models.densenet import DenseNet
import torch, cv2

ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'png'])
UPLOAD_FOLDER = 'uploads'

# load resnet model

resnet_model = DenseNet(growthRate=12,
                        depth=100,
                        reduction=0.5,
                        bottleneck=True,
                        nClasses=4)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
resnet_model.to(device)
resnet_model.eval()

trained_model_path = 'model/ckpt/checkpoint_01/bs128_lr1e-06_epoch50_bestacc0.7324324250221252_sgd.pt'
if os.path.exists(trained_model_path):
    resnet_model.load_state_dict(
        torch.load(trained_model_path, map_location='cpu'))
else:
    print('You do not load model !')
    exit()

# load label map

with open('model/data/label.map', 'r', encoding='utf8') as fi_label:
    id2label = {idx: label for idx, label in enumerate(fi_label.readlines())}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

import torchvision.transforms as transforms


def inference(file_path):
    img = cv2.imread(file_path)
    img = cv2.resize(img, (32, 32))
    mean = [117.40523108152614, 129.08895341687355, 138.8146819924039]
    std = [71.07845902847068, 67.67320379657173, 67.58323701160265]

    local_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean, std)])

    img = local_transform(img).unsqueeze(0).to(device)

    img_out = resnet_model(img)
    _, preds = img_out.max(0)
    preds = preds.item()
    return preds


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            output = inference(file_path)
            output = f'Category is【{id2label[output]}】'

            return render_template('home.html',
                                   label=output,
                                   imagesource=file_path,
                                   mode_code=1)
    return render_template('home.html',
                           label='',
                           imagesource='static/imgs/example.jpg',
                           mode_code=0)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, threaded=False)
