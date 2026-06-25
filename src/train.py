import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import json
import joblib
import os
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

load_dotenv()

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
mlflow.set_registry_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
PROJECT = os.getenv("PROJECT")

EVAL_THRESHOLD = 0.70


def train(
    params: dict,
    data_path: str = "data/train_phase1.csv",
    eval_path: str = "data/eval.csv",
) -> float:
    """
    Huấn luyện mô hình và ghi nhận kết quả vào MLflow.

    Tham số:
        params: dict chứa các siêu tham số cho RandomForestClassifier
        data_path: đường dẫn đến file dữ liệu huấn luyện
        eval_path: đường dẫn đến file dữ liệu đánh giá

    Trả về:
        accuracy (float): độ chính xác trên tập đánh giá
    """
    # TODO 1.5.1: Đọc dữ liệu huấn luyện và đánh giá từ CSV vào DataFrame.
    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    # TODO 1.5.2: Tách đặc trưng (X) và nhãn (y) cho cả hai tập.
    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    # TODO 1.5.3: Bắt đầu một MLflow run để theo dõi thí nghiệm.
    with mlflow.start_run():
        # TODO 1.5.4: Ghi nhận các siêu tham số vào MLflow.
        mlflow.log_params(params)

        # TODO 1.5.5: Khởi tạo và huấn luyện mô hình RandomForestClassifier.
        model = RandomForestClassifier(**params, random_state=42)
        model.fit(X_train, y_train)

        # TODO 1.5.6: Dự đoán và tính accuracy, f1_score trên tập đánh giá.
        preds = model.predict(X_eval)
        acc = accuracy_score(y_eval, preds)
        f1 = f1_score(y_eval, preds, average="weighted")

        # TODO 1.5.7: Ghi nhận các chỉ số vào MLflow.
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        # TODO 1.5.8: Log mô hình đã huấn luyện vào MLflow artifact.
        mlflow.sklearn.log_model(model, "model")

        # TODO 1.5.9: In kết quả ra màn hình để theo dõi.
        print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")

        # TODO 1.5.10: Lưu metrics ra outputs/metrics.json cho GitHub Actions.
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/metrics.json", "w") as f:
            json.dump({"accuracy": acc, "f1_score": f1}, f)

        # TODO 1.5.11: Lưu mô hình ra models/model.pkl để upload lên GCS.
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")

    # TODO 1.5.12: Trả về accuracy cho hàm gọi.
    return acc


if __name__ == "__main__":
    # Đọc siêu tham số từ params.yaml và gọi hàm train()
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
