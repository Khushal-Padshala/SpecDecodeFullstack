"""
Comprehensive, diverse prompt dataset for training high-acceptance draft models.
Spans programming, reasoning, math, QA, structured extraction, and conversational English.
"""

EXPANDED_PROMPTS = [
    # Code & Syntax patterns (Python, SQL, JS, Shell)
    {"domain": "code", "prompt": "Write a Python function to check if a string is a palindrome ignoring case and non-alphanumeric characters."},
    {"domain": "code", "prompt": "Write a Python function to compute the Fibonacci sequence iteratively up to n elements."},
    {"domain": "code", "prompt": "Implement a binary search algorithm in Python that returns the index of target in a sorted list."},
    {"domain": "code", "prompt": "Write a Python function to reverse a singly linked list in-place."},
    {"domain": "code", "prompt": "Write a Python function to find the maximum sum contiguous subarray using Kadane's algorithm."},
    {"domain": "code", "prompt": "Write a SQL query to select all employees with a salary greater than 70000 ordered by department ID."},
    {"domain": "code", "prompt": "Write a Python function to read a CSV file, calculate column averages, and return a dictionary."},
    {"domain": "code", "prompt": "Write a Python decorator that measures and prints the execution time of any function."},
    {"domain": "code", "prompt": "Write a Python function to validate whether an email address format is valid using regular expressions."},
    {"domain": "code", "prompt": "Implement a quicksort algorithm in Python and explain its average time complexity."},
    {"domain": "code", "prompt": "Write a Python function to check if two given strings are anagrams of each other."},
    {"domain": "code", "prompt": "Write a Python class for a stack data structure with push, pop, peek, and is_empty methods."},
    {"domain": "code", "prompt": "Write a Python function to flatten a nested list of arbitrary depth into a single list."},
    {"domain": "code", "prompt": "Write a Python script to send an HTTP GET request to a JSON endpoint and handle connection errors."},
    {"domain": "code", "prompt": "Write a SQL query using GROUP BY and HAVING to find departments with more than 5 employees."},

    # Algorithmic reasoning & logic
    {"domain": "reasoning", "prompt": "Explain step-by-step why binary search has a time complexity of O(log n)."},
    {"domain": "reasoning", "prompt": "Explain the difference between breadth-first search (BFS) and depth-first search (DFS)."},
    {"domain": "reasoning", "prompt": "What is the difference between concurrency and parallelism in computer science?"},
    {"domain": "reasoning", "prompt": "Explain why hash table lookups are average O(1) time complexity and when they degrade to O(n)."},
    {"domain": "reasoning", "prompt": "Explain the concept of dynamic programming and how memoization avoids redundant calculations."},
    {"domain": "reasoning", "prompt": "What is the difference between a process and a thread in modern operating systems?"},
    {"domain": "reasoning", "prompt": "Explain how speculative decoding accelerates LLM generation using draft and target models."},
    {"domain": "reasoning", "prompt": "Explain how public-key cryptography and RSA encryption work at a high level."},

    # General Knowledge, Summarization & Structured Q&A
    {"domain": "qa", "prompt": "What are the three primary colors and how do they combine to form secondary colors?"},
    {"domain": "qa", "prompt": "Explain the water cycle step by step including evaporation, condensation, and precipitation."},
    {"domain": "qa", "prompt": "Summarize the three main advantages of cloud computing for modern businesses."},
    {"domain": "qa", "prompt": "Explain the difference between renewable and non-renewable energy sources with examples."},
    {"domain": "qa", "prompt": "What causes the four seasons on Earth? Explain axial tilt and orbital revolution."},
    {"domain": "qa", "prompt": "List 5 essential practices for maintaining personal cybersecurity online."},
    {"domain": "qa", "prompt": "Explain the concept of inflation in economics and two primary factors that cause it."}
]

def get_training_prompts():
    return EXPANDED_PROMPTS

if __name__ == "__main__":
    prompts = get_training_prompts()
    print(f"Total training prompts: {len(prompts)}")
