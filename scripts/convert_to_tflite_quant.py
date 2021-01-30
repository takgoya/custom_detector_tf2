import tensorflow as tf
import argparse

# Define model and output directory arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Folder that the saved model is located in',
                    default='tflite_models/ssd_mobilenet_v2_fpnlite_320_320/saved_model')
parser.add_argument('--output', help='Folder that the tflite model will be written to',
                    default='tflite_models/ssd_mobilenet_v2_fpnlite_320_320')
parser.add_argument('--images', help='Folder that the images dataset is located in',
                    default='images/test')
args = parser.parse_args()


# A generator that provides a representative dataset
def representative_dataset():
  dataset_list = tf.data.Dataset.list_files(args.images + '/*jpg')
  for i in range(830):
    image = next(iter(dataset_list))
    image = tf.io.read_file(image)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [320, 320])
    image = tf.cast(image / 255., tf.float32)
    image = tf.expand_dims(image, 0)
    yield [image]

converter = tf.lite.TFLiteConverter.from_saved_model(args.model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset

#converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

converter.allow_custom_ops = True
converter.experimental_new_converter = True

tflite_quant_model = converter.convert()

output = args.output + '/model_quant.tflite'
with tf.io.gfile.GFile(output, 'wb') as f:
  f.write(tflite_quant_model)