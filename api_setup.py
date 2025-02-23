api_key = input("Enter your FRED API key: ")

with open(".env", "w") as f:
    f.write(f"FRED_API_KEY={api_key}\n")

print("API key saved successfully!")
