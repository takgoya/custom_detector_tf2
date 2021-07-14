# custom_detector_tf2
Customized Object Detector from TensorFlow 2 Object Detection API
TF2 model trained with custom dataset (transfer learning applied)

- TF2 object detection API --> **TF2_object_detection.ipynb**

- Modelos TFLite --> **/tflite_models**

- Modelos TF --> **/models**
  
- Ejecutar modelos TFLite

  - detecciones sobre cámara en vivo --> **tflite_object_detection_camera.py**
  - detecciones sobre archivo de vídeo --> **tflite_object_detection_video.py**
  - detecciones sobre una imágen --> **tflite_object_detection_image.py**

- Ejecutar modelos TF
  - detecciones sobre cámara en vivo --> **object_detection_camera.py**
  - detecciones sobre archivo de vídeo --> **object_detection_video.py**
  - detecciones sobre una imágen --> **object_detection_image.py**

- Ficheros de configuración
  - tflite --> **conf_tflite.json**
  - tf -> **conf.json**

- Reconocimiento de matrículas --> **ocr_plate_recognition.py**

- Gestión base de datos --> **/db**

- Gestión GPS --> **/gps**

- Gestión botón en Raspberry PI para activar/parar detección --> **/button**
 
- Directorio con imáges/vídeos/capturas procesados --> **/media**
