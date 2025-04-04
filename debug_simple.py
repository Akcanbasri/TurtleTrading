print("Debug starting...")

with open(".env", "r") as f:
    content = f.read()
    print("Content of .env file:")
    print(repr(content))

print("Debug finished.")
