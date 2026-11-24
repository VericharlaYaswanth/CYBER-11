import hashlib
import os
import getpass


def hash_password(password):
    salt = os.urandom(16)

    hashed = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode(),
        salt,
        200000
    )

    return salt.hex() + ":" + hashed.hex()


if __name__ == "__main__":
    password = getpass.getpass("Enter password: ")
    confirm = getpass.getpass("Confirm password: ")

    if password != confirm:
        print("Passwords do not match!")
    else:
        print("\nPassword hash:")
        print(hash_password(password))