from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

# Check loaded variables - focus on the problematic ones
print("DC_LENGTH_ENTER:", repr(os.getenv("DC_LENGTH_ENTER")))
print("DC_LENGTH_EXIT:", repr(os.getenv("DC_LENGTH_EXIT")))
print("ATR_LENGTH:", repr(os.getenv("ATR_LENGTH")))
print("RISK_PER_TRADE:", repr(os.getenv("RISK_PER_TRADE")))

# Print the raw content of the .env file
print("\nRaw content of .env file:")
with open(".env", "r") as f:
    for i, line in enumerate(f):
        # Strip newlines but keep other whitespace for debugging
        print(f"{i}: {repr(line.rstrip())}")
