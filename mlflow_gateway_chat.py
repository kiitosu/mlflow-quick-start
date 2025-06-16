from mlflow.deployments import get_deploy_client

client = get_deploy_client("http://localhost:5002")
name = "chat"
data = dict(
    messages=[
        # システムに「あなたはハリー・ポッターのソーティングハットです。」と伝える
        {"role": "system", "content": "You are the sorting hat from harry potter."},
        # ユーザーに「私は勇敢で、努力家で、賢く、裏切り者です。」と伝える
        {
            "role": "user",
            "content": "I am brave, hard-working, wise, and backstabbing.",
        },
        # ユーザーに「私はどのハリーポッターの家に属する可能性が高いですか？」と伝える
        {
            "role": "user",
            "content": "Which harry potter house am I most likely to belong to?",
        },
    ],
    n=3,
    temperature=0.5,
)

response = client.predict(endpoint=name, inputs=data)
print(response)
