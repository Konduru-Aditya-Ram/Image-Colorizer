# from flask import Flask, request, redirect, url_for, render_template, jsonify
# import os
# from PIL import Image, ImageOps
# import io
# import base64
# import tensorflow as tf
# from keras.preprocessing.image import img_to_array, load_img
# from skimage.transform import resize
# from skimage.io import imsave, imshow
# import numpy as np
# import cv2 as cv
# import matplotlib.pyplot as plt
# from skimage.color import rgb2lab, lab2rgb



# app = Flask(__name__)


# @app.route('/')
# def index():
#     """Render the upload form (optional)."""
#     return render_template('index.html')


# def mse(y_true, y_pred):
#     return tf.reduce_mean(tf.square(y_pred - y_true))

# @app.route('/upload' , methods=['POST'])
# def col():
#     bwimg=request.files['file-input']
#     image = Image.open(bwimg)

   

#     model = tf.keras.models.load_model(r'C:\Users\Kondu\project(iitSoC)\models\Image colorizer.keras',
#                                     custom_objects={'mse': mse},
#                                     compile=True)

#     img1_color=[]
#     # image_path=r"C:\Users\Kondu\OneDrive\Desktop\dataset(color images-2)\temp\prabhas-3.jpg"
#     # ori=img_to_array(load_img(image_path))
#     ori=img_to_array(image)
#     ori_cv=image
#     # ori_cv=cv.imread(image_path)

#     img1 = resize(ori ,(256,256))
#     img1_color.append(img1)

#     img1_color = np.array(img1_color, dtype=float)
#     img1_color = rgb2lab(1.0/255*img1_color)[:,:,:,0]
#     img1_color = img1_color.reshape(img1_color.shape+(1,))

#     output1 = model.predict(img1_color)
#     output1 = output1*128

#     result = np.zeros((256, 256, 3))
#     result[:,:,0] = img1_color[0][:,:,0]
#     result[:,:,1:] = output1[0]

#     result = np.zeros((256, 256, 3))
#     result[:,:,0] = img1_color[0][:,:,0]
#     result[:,:,1:] = output1[0]
#     result=lab2rgb(result)
#     result=resize(result ,ori_cv.shape)



#     img_io = io.BytesIO()
#     result.save(img_io, format='PNG')
#     img_io.seek(0)
#     img_data = base64.b64encode(img_io.getvalue()).decode('utf-8')
#     # Send the modified image to the template
#     return render_template('colorized.html', img_data=img_data)

# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, request, render_template,send_file
import io
import base64
import tensorflow as tf
from keras.preprocessing.image import img_to_array
from skimage.transform import resize
from skimage.color import rgb2lab, lab2rgb
from PIL import Image
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    """Render the upload form."""
    # return render_template('index.html')
    return send_file('index.html')

def mse(y_true, y_pred):
    """Custom Mean Squared Error function."""
    return tf.reduce_mean(tf.square(y_pred - y_true))

@app.route('/upload', methods=['POST'])
def col():
    """Handle file upload, colorize the image, and send it to the template."""
    if 'file-input' not in request.files:
        return "No file part in the request", 400

    bwimg = request.files['file-input']
    if bwimg.filename == '':
        return "No file selected", 400

    image = Image.open(bwimg).convert('RGB')

 
    model = tf.keras.models.load_model(
       r'\static\styles\Image colorizer.keras',
        custom_objects={'mse': mse},
        compile=True
    )

    ori = img_to_array(image)
    img1_color = [resize(ori, (256, 256), anti_aliasing=True)]
    img1_color = np.array(img1_color, dtype=float)
    img1_color = rgb2lab(1.0 / 255 * img1_color)[:, :, :, 0]
    img1_color = img1_color.reshape(img1_color.shape + (1,))

   
    output1 = model.predict(img1_color)
    output1 = output1 * 128

   
    result = np.zeros((256, 256, 3))
    result[:, :, 0] = img1_color[0][:, :, 0]
    result[:, :, 1:] = output1[0]
    result_rgb = lab2rgb(result)

    
    result_resized = resize(result_rgb, ori.shape[:2], anti_aliasing=True)

    
    result_pil = Image.fromarray((result_resized * 255).astype('uint8'))
    img_io = io.BytesIO()
    result_pil.save(img_io, format='PNG')
    img_io.seek(0)

   
    img_data = base64.b64encode(img_io.getvalue()).decode('utf-8')
    image.save(img_io,format='PNG')
    img_io.seek(0)
    bwimg_data=base64.b64encode(img_io.getvalue()).decode('utf-8')
   
    return render_template('colorized.html', img_data=img_data,bwimg_data=bwimg_data)

if __name__ == '__main__':
    app.run(debug=True)
