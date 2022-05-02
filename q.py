from model import print_time, tokenizer, model, predict


context = '''# Create a function named 'is_prime' that test a number wether a prime number

def
'''

with print_time('predict'):
    print(context + predict(tokenizer, model, context, max_length=1024))
    print()
