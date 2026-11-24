import os
import json
import secrets
import hashlib
import getpass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUTH_FILE = os.path.join(BASE_DIR, "auth_config.json")

if not os.path.exists(AUTH_FILE):
    print("Error: auth_config.json not found.")
    raise SystemExit(1)

with open(AUTH_FILE, "r", encoding="utf-8") as file:
    config = json.load(file)

users = config.get("users", {})

print("\n========== CYBER-11 PASSWORD RESET ==========")
print("\nExisting accounts:")

for username in users:
    print("-", username)

username = input("\nEnter the username to reset: ").strip()

if username not in users:
    print("Error: Username does not exist.")
    raise SystemExit(1)

new_password = getpass.getpass("Enter new password: ")
confirm_password = getpass.getpass("Confirm new password: ")

if len(new_password) < 8:
    print("Error: Password must contain at least 8 characters.")
    raise SystemExit(1)

if new_password != confirm_password:
    print("Error: Passwords do not match.")
    raise SystemExit(1)

salt = secrets.token_hex(16)

password_hash = hashlib.pbkdf2_hmac(
    "sha256",
    new_password.encode("utf-8"),
    salt.encode("utf-8"),
    200000
).hex()

users[username]["password_hash"] = f"{salt}:{password_hash}"

config["users"] = users

with open(AUTH_FILE, "w", encoding="utf-8") as file:
    json.dump(config, file, indent=4)

print("\nPassword successfully reset!")
print("Your account role has been preserved.")