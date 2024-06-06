from joblib import load

# Đường dẫn đến file chứa model
file_path = 'best_svm_model.joblib'

# Tải model từ file
model = load(file_path)

if model is not None:
    print("Load model thành công")
else:
    print("Không thể load được model")
