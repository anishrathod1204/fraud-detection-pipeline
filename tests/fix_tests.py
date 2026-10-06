import re

with open("tests/test_features.py", "r") as f:
    content = f.read()

# Add isFraud and isFlaggedFraud where missing before the closing brace of the dict
content = re.sub(r'("newbalanceDest": \d+\.\d+,)\n\s*\}', r'\1\n            "isFraud": 0,\n            "isFlaggedFraud": 0,\n        }', content)

with open("tests/test_features.py", "w") as f:
    f.write(content)
