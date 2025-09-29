import re
import sys

def clean_log_file(log_file_path):
    with open(log_file_path, 'r') as file:
        lines = file.readlines()

    with open(log_file_path, 'w') as file:
        for line in lines:
            # Remove lines containing progress percentages from 1% to 99%
            if re.search(r"\[\s*(?:\d{1,2})%\]", line) and not re.search(r"\[\s*100%\]", line):
                continue
            file.write(line)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python clean_log.py <log_file_path>")
    else:
        log_file_path = sys.argv[1]
        clean_log_file(log_file_path)