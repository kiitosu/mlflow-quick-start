from typing import Any

import mlflow
import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# 評価データセットを定義します
eval_data = pd.DataFrame(
    {
        "inputs": [
            "What is MLflow?",
            "What is Spark?",
        ],
        "ground_truth": [
            "MLflow is an open-source platform for managing the end-to-end machine learning (ML) lifecycle. It was developed by Databricks, a company that specializes in big data and machine learning solutions. MLflow is designed to address the challenges that data scientists and machine learning engineers face when developing, training, and deploying machine learning models.",
            "Apache Spark is an open-source, distributed computing system designed for big data processing and analytics. It was developed in response to limitations of the Hadoop MapReduce computing model, offering improvements in speed and ease of use. Spark provides libraries for various tasks such as data ingestion, processing, and analysis through its components like Spark SQL for structured data, Spark Streaming for real-time data processing, and MLlib for machine learning tasks",
        ],
    }
)


# モデルを評価するための関数を定義します
# 生成AIの出力を返します
def openai_qa(inputs: pd.DataFrame) -> list[str | list[str | dict[Any, Any]]]:
    predictions = []
    system_prompt = "Please answer the following question in formal language."

    # eval_dtaの内 inputs を取り出して、invokeします
    llm = ChatOpenAI(
        base_url="http://localhost:1234/v1",
        api_key=None,
        temperature=0.7,
        name="google/gemma-3-12b",
    )
    for question in inputs["inputs"]:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=question),
        ]
        output = llm.invoke(messages).content
        predictions.append(output)

    return predictions


# 既存のアクティブなrunがあれば終了
if mlflow.active_run():
    mlflow.end_run()

# 評価を実行します
with mlflow.start_run():
    results = mlflow.evaluate(
        model=openai_qa,  # 上で定義した生成AIを実行する関数
        data=eval_data,  # 評価データセット
        targets="ground_truth",  # 評価データセットの正解ラベル
        model_type="question-answering",  # モデルのタイプを指定します
    )

print(results.metrics)
