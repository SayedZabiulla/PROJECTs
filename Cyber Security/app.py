from flask import Flask, request, render_template_string
import hashlib

app = Flask(__name__)

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

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        password = request.form['password']
        hashed_password = hash_password(password)
        cracked = dictionary_attack(hashed_password, "common_passwords.txt")

        policy_violations = analyze_password_policy(password)

        return render_template_string('''
            <!DOCTYPE html>
            <html>
            <head>
                <title>Password Strength Analyzer</title>
                <link rel="stylesheet" type="text/css" href="/static/style.css">
            </head>
            <body>
                <div class="container">
                    <h2>Password Check Result</h2>
                    {% if cracked %}
                        <p class="result found">Password found in dictionary: {{cracked}}</p>
                    {% else %}
                        <p class="result not-found">Password not found in dictionary.</p>
                    {% endif %}
                    {% if policy_violations %}
                        <p class="violations">Policy Violations:</p>
                        <ul class="violations-list">
                            {% for violation in policy_violations %}
                                <li>{{violation}}</li>
                            {% endfor %}
                        </ul>
                    {% else %}
                        <p class="compliance">Password meets all policies.</p>
                    {% endif %}
                    <a class="retry-button" href="/">Try another password</a>
                </div>
            </body>
            </html>
        ''', cracked=cracked, policy_violations=policy_violations)

    return '''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Password Strength Analyzer</title>
            <link rel="stylesheet" type="text/css" href="/static/style.css">
        </head>
        <body>
            <div class="container">
                <h2>Password Strength Analyzer</h2>
                <form method="post">
                    <label for="password">Password:</label>
                    <input type="password" name="password" id="password">
                    <input type="submit" value="Check Password">
                </form>
            </div>
        </body>
        </html>
    '''

if __name__ == '__main__':
    app.run(debug=True)
