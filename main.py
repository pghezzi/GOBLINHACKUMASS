from flask import Flask, request, render_template
from inference import main as lipmain
from argparse import Namespace
import time

app = Flask(__name__)

# Set the upload folder
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    args = {'face': 'input/dictator_orig.mp4', 'audio': 'input/dictator_audio_extracted.wav', 'outfile': f'results/{time.time()}', 'static': False, 'fps': 25.0, 'pads': [0, 10, 0, 0], 'face_det_batch_size': 16, 'wav2lip_batch_size': 128, 'resize_factor': 1, 'crop': [0, -1, 0, -1], 'box': [-1, -1, -1, -1], 'rotate': False, 'nosmooth': False, 'img_size': 96, 'checkpoint_path': 'checkpoints/wav2lip_gan.pth'}
    lipmain(Namespace(**args))
    return "WORKING"

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    
    file = request.files['file']
    
    if file.filename == '':
        return 'No selected file'
    
    if file and allowed_file(file.filename):
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
        return 'File uploaded successfully'
    else:
        return 'File type not allowed'

if __name__ == '__main__':
    app.run(debug=True)
