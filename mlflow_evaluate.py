from typing import Any

import mlflow
import mlflow.pyfunc
import pandas as pd  # type: ignore
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

eval_data = pd.DataFrame(
    {
        "inputs": [
            "What is MLflow?",
            "What is Spark?",
        ],
        "ground_truth": [
            "MLflow is an open-source platform for managing the end-to-end machine learning "
            "lifecycle. It was developed by Databricks, a company that specializes in big data and "
            "machine learning solutions. MLflow is designed to address the challenges that data "
            "scientists and machine learning engineers face when developing, training, and deploying "
            "machine learning models.",
            "Apache Spark is an open-source, distributed computing system designed for big data "
            "processing and analytics. It was developed in response to limitations of the Hadoop "
            "MapReduce computing model, offering improvements in speed and ease of use. Spark "
            "provides libraries for various tasks such as data ingestion, processing, and analysis "
            "through its components like Spark SQL for structured data, Spark Streaming for "
            "real-time data processing, and MLlib for machine learning tasks",
        ],
    }
)

# MLflowサーバーのURIを設定
mlflow.set_tracking_uri("http://localhost:5001")
mlflow.set_experiment("my-genai-experiment")

# 既存のアクティブなrunがあれば終了
if mlflow.active_run():
    mlflow.end_run()


# mlflow.pyfunc.log_model にわたすために必要な関数を実装する
class ChatOpenAIWrapper(mlflow.pyfunc.PythonModel):  # type: ignore
    def __init__(self):
        self.system_prompt = "Answer the following question in two sentences"

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,  # type: ignore
        model_input: pd.DataFrame,
    ) -> list[str | list[str | dict[Any, Any]]]:
        # contextからモデル設定を取得することも可能（今回は使用しない）
        # model_config = context.artifacts if context else {

        # ここで使いたいモデルを指定する
        llm = ChatOpenAI(
            base_url="http://localhost:1234/v1",
            api_key=None,
            temperature=0.7,
            name="google/gemma-3-12b",
        )

        # 結果はまとめて返却する
        predictions = []
        for question in model_input["inputs"]:
            response = llm.invoke(
                [
                    SystemMessage(content=self.system_prompt),
                    HumanMessage(content=question),
                ]
            )
            predictions.append(response.content)

        return predictions


with mlflow.start_run() as run:
    # 入力例を作成
    input_example = pd.DataFrame({"inputs": ["What is MLflow?"]})

    # モデルのsignatureを推論するための予測を実行
    model_instance = ChatOpenAIWrapper()

    # カスタムモデルをMLflowモデルとしてログ（signatureとinput_exampleを指定）
    logged_model_info = mlflow.pyfunc.log_model(
        name="model",
        python_model=model_instance,
        input_example=input_example,
    )

    # 事前定義された question-answering metrics を使って評価する
    results = mlflow.evaluate(
        logged_model_info.model_uri,
        eval_data,
        targets="ground_truth",
        model_type="question-answering",
    )
    print(f"See aggregated evaluation results below: \n{results.metrics}")

    # `results.tables` でデータ毎の結果を取得できる
    eval_table = results.tables["eval_results_table"]
    print(f"See evaluation table below: \n{eval_table}")
