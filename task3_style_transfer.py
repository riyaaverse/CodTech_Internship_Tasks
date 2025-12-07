# Task 3: Neural Style Transfer
# Instructions: Implement a Neural Style Transfer model to apply artistic styles to photographs.
# Libraries: TensorFlow, TensorFlow Hub, OpenCV

import tensorflow_hub as hub
import tensorflow as tf
import cv2
import numpy as np

def load_image(img_path):
    """Loads an image and prepares it for the AI model."""
    # Read the file
    img = tf.io.read_file(img_path)
    # Decode raw data into an image
    img = tf.image.decode_image(img, channels=3)
    # Convert image data to float32 format (required for the model)
    img = tf.image.convert_image_dtype(img, tf.float32)
    # Add a "batch" dimension (models expect a batch of images, not just one)
    img = img[tf.newaxis, :]
    return img

def tensor_to_image(tensor):
    """Converts the AI's output back into a saveable image file."""
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        tensor = tensor[0]
    return tensor

if __name__ == "__main__":
    print("Loading the AI model from Google TensorFlow Hub... (this usually takes 1-2 minutes)")
    # We use a pre-trained model specifically made for style transfer
    model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

    try:
        print("Loading images...")
        # Ensure these files exist in your folder!
        content_image = load_image('content.jpg')
        style_image = load_image('style.jpg')

        print("Stylizing image... (Applying the artistic style to your photo)")
        # The model takes two inputs: content and style
        stylized_image = model(tf.constant(content_image), tf.constant(style_image))[0]

        # Save the result
        output_filename = 'generated_art.jpg'
        # Convert RGB to BGR for OpenCV saving
        final_image = cv2.cvtColor(tensor_to_image(stylized_image), cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_filename, final_image)
        
        print(f"\nSuccess! The styled image has been saved as '{output_filename}'.")
        print("Go check your folder to see the result!")

    except Exception as e:
        print(f"\nError: {e}")
        print("Please make sure you have 'content.jpg' and 'style.jpg' in this folder.")