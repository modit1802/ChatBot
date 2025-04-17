import requests

# Test with a simple query
response = requests.post(
    "http://localhost:8000/search",
    json={"query": "what is the sign for A"}
)
print(response.json())
