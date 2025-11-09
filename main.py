from src.data_cleaning import clean_data
from src.train_model import train_model
from src.predict import predict_new_data

clean_data()
model, metrics = train_model()
preds, best = predict_new_data()

print("\n✅ Pipeline executed successfully!")
print("Best Product Recommendation:\n", best)
