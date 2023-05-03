import re


string = "ba12456B789C"
digits = re.findall(r'\d+(?=[A-Za-z])', string)
print(digits)  # Output: ['123', '456', '789']