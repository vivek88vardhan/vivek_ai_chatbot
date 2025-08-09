from prefect import flow, task

@task
def process_input(text):
    return text.upper()

@flow
def chatbot_flow(user_input):
    result = process_input(user_input)
    print("Processed:", result)
