import hashlib
from termcolor import colored

def hash_password(password, algorithm="sha256"):
    """Hash a password using the selected algorithm."""
    if algorithm == "sha256":
        return hashlib.sha256(password.encode()).hexdigest()
    elif algorithm == "md5":
        return hashlib.md5(password.encode()).hexdigest()
    else:
        raise ValueError("Unsupported hashing algorithm")

def dictionary_attack(hash_to_crack, dictionary_file, algorithm="sha256"):
    """Perform a dictionary attack to crack a password."""
    with open(dictionary_file, "r") as file:
        for line in file:
            word = line.strip()
            if hash_password(word, algorithm) == hash_to_crack:
                return word  # Password found
    return None  # Password not found

def analyze_password_policy(password):
    """Analyze the password based on basic policy rules."""
    rules = [
        (len(password) >= 8, "Password should be at least 8 characters long."),
        (any(c.isupper() for c in password), "Password should include an uppercase letter."),
        (any(c.islower() for c in password), "Password should include a lowercase letter."),
        (any(c.isdigit() for c in password), "Password should include a number."),
        (any(c in "!@#$%^&*()-_=+[]{}" for c in password), "Password should include a special character."),
    ]
    return [rule for passed, rule in rules if not passed]

def check_password(password, dictionary_file="common_passwords.txt", algorithm="sha256"):
    hashed_password = hash_password(password, algorithm)
    print(f"Hashed password ({algorithm}): {hashed_password}")

    cracked = dictionary_attack(hashed_password, dictionary_file, algorithm)
    if cracked:
        print(colored(f"Password found in dictionary: {cracked}", "red"))
        return False

    policy_violations = analyze_password_policy(password)
    if policy_violations:
        print(colored("Policy violations:", "yellow"))
        for rule in policy_violations:
            print(colored(f"- {rule}", "yellow"))
        return False
    else:
        print(colored("Password meets all policies.", "green"))
        return True

def main():
    import getpass
    password = getpass.getpass("Enter password to check: ")
    if check_password(password):
        print(colored("Password is strong and not found in the dictionary.", "green"))
    else:
        print(colored("Password is weak or found in the dictionary.", "red"))

if __name__ == "__main__":
    main()
