from keras.models import load_model
from utils import get_face_embedding, compare_faces

# Load the pre-trained FaceNet model
model = load_model("facenet_keras.h5")

# Get embeddings for two images
embedding1 = get_face_embedding(model, "face1.jpg")
embedding2 = get_face_embedding(model, "face2.jpg")

# Compare the two faces
distance = compare_faces(embedding1, embedding2)

print(f"Euclidean Distance: {distance}")
