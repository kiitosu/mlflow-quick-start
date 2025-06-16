import mlflow

##################
# 専門性の評価例を定義
##################
professionalism_example_score_2 = mlflow.metrics.genai.EvaluationExample(
    input="What is MLflow?",
    output=(
        "MLflow is like your friendly neighborhood toolkit for managing your machine learning projects. It helps "
        "you track experiments, package your code and models, and collaborate with your team, making the whole ML "
        "workflow smoother. It's like your Swiss Army knife for machine learning!"
    ),
    score=2,
    justification=(
        "The response is written in a casual tone. It uses contractions, filler words such as 'like', and "
        "exclamation points, which make it sound less professional. "
    ),
)
professionalism_example_score_4 = mlflow.metrics.genai.EvaluationExample(
    input="What is MLflow?",
    output=(
        "MLflow is an open-source platform for managing the end-to-end machine learning (ML) lifecycle. It was "
        "developed by Databricks, a company that specializes in big data and machine learning solutions. MLflow is "
        "designed to address the challenges that data scientists and machine learning engineers face when "
        "developing, training, and deploying machine learning models.",
    ),
    score=4,
    justification=("The response is written in a formal language and a neutral tone. "),
)


#########################
# 専門性の評価メトリクスを定義
#########################
def professionalism(
    model: str,
):  # answer_similarity 等既存のメトリクスに合わせてmodelを引数にしました
    return mlflow.metrics.genai.make_genai_metric(
        name="professionalism",
        definition=(
            "Professionalism refers to the use of a formal, respectful, and appropriate style of communication that is "
            "tailored to the context and audience. It often involves avoiding overly casual language, slang, or "
            "colloquialisms, and instead using clear, concise, and respectful language."
        ),
        grading_prompt=(
            "Professionalism: If the answer is written using a professional tone, below are the details for different scores: "
            "- Score 0: Language is extremely casual, informal, and may include slang or colloquialisms. Not suitable for "
            "professional contexts."
            "- Score 1: Language is casual but generally respectful and avoids strong informality or slang. Acceptable in "
            "some informal professional settings."
            "- Score 2: Language is overall formal but still have casual words/phrases. Borderline for professional contexts."
            "- Score 3: Language is balanced and avoids extreme informality or formality. Suitable for most professional contexts. "
            "- Score 4: Language is noticeably formal, respectful, and avoids casual elements. Appropriate for formal "
            "business or academic settings. "
        ),
        # ここで評価例を渡しています
        examples=[professionalism_example_score_2, professionalism_example_score_4],
        model=model,
        parameters={"temperature": 0.0},
        aggregations=["mean", "variance"],
        greater_is_better=True,
    )
