from model import print_time, tokenizer, model, predict


context = '''def read_and_sort(txt_path):
    """Read numbers from a txt file, each number in a line, return sorted array by desc"""
    '''

with print_time('predict'):
    print(predict(tokenizer, model, context))
    print()

context = '''def fun(arr):
    """
    find the top 3 biggest values in arr
    """
    '''

with print_time('predict'):
    print(predict(tokenizer, model, context))
    print()

context = '''def fun(arr):
    """
    return the second smallest number in arr
    """
    '''

with print_time('predict'):
    print(predict(tokenizer, model, context))
    print()

context = '''def query(db_path, sql):
    """
    connect sqlite3 database where path is db_path
    execute sql and return the results
    """
    '''

with print_time('predict'):
    print(predict(tokenizer, model, context))
    print()

