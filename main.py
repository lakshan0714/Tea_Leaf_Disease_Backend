from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import io
from Predict import predict
import uvicorn
import tensorflow as tf

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React app will run on port 3000
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your trained model (replace with your actual model path)
model = tf.keras.models.load_model('./models/Model_v1.h5')

@app.get("/")
async def root():
    return {"message": "Breast Cancer Detection API"}



@app.post("/predict")
async def make_predict(file: UploadFile = File(...)):

    # Read the image file
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")  # Ensure the image is in RGB format
    image = image.resize((224, 224))
    prediction=predict(model,image)
    
    # # Preprocess the image
    # image = image.resize((224, 224))  # Adjust size based on your model's requirements
    # image_array = np.array(image)
    # image_array = image_array / 255.0  # Normalize
    # image_array = np.expand_dims(image_array, axis=0)
    
    # Make prediction (uncomment when model is loaded)
    # prediction = model.predict(image_array)
    # result = "Cancer" if prediction[0][0] > 0.5 else "No Cancer"
    
    # For now, return a dummy prediction

    if prediction==0:
         result = 'Blister_Blide'  # Replace with actual prediction
    elif prediction==1:
        result='Brown_Blight'
    else:
        result = 'healthy'
    return {"prediction": result}



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 