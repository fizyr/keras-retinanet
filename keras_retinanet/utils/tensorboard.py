# -*- coding: utf-8 -*-
import tensorflow as tf
import io
from PIL import Image
import numpy as np

def TensorboardImage(writer, image, number=0, step=0):
    """  Display images on tensorboard
    # Arguments
        writer : buffer to write the image
        images : image OPENCV format.  we cast in uint8 because need for tensorboard buffer format.
        
    """
    

    height, width, c = image.shape
    
        
    # temp = images[i,:,:,:]*255
    temp = np.copy(image)
    #if channel == 1:
        #temp = np.expand_dims(temp, axis=-1)
        
    temp = temp.astype('uint8')
    
        
    if c == 3:
        output = io.BytesIO()
        temp = temp[:, :, ::-1] #BGR to RGB
        temp = Image.fromarray(temp)
        temp.save(output, format='JPEG')
        image_string = output.getvalue()
        output.close()
        img = tf.Summary.Image(height=height,
                             width=width,
                             colorspace=c,
                             encoded_image_string=image_string)
            #img = sess.run(tf.summary.image(name='image'+str(i), tensor=np.expand_dims(images[i,:,:,:], axis=0), max_outputs=1))
    
        summary =  tf.Summary(value=[tf.Summary.Value(tag='image_'+str(number), image=img )])
        writer.add_summary(summary, step)