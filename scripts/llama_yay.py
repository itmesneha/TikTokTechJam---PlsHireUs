from llama_cpp import Llama

# Load the model
llm = Llama(model_path="./models/7B/ggml-model.bin")

# Call the model
output = llm(
    "Q: Name the planets in the solar system? A: ",
    max_tokens=32,
    stop=["Q:", "\n"],
    echo=True
)

print(output)
