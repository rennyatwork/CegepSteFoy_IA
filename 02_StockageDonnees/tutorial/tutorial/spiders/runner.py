## https://github.com/Microsoft/vscode-python/issues/63
import os
from scrapy.cmdline import execute

#dir = os.getcwd()
#spider = "file://" +dir + "/quotes_spyder.py"
print('abs dirname: ', os.path.dirname(os.path.abspath(__file__)))
##https://note.nkmk.me/en/python-script-file-path/
spider = os.path.dirname(os.path.abspath(__file__)) + "/quotes_spyder.py"
#execute(['scrapy','runspider', 'fullly qualified path to myspider.py file'])
execute(['scrapy','runspider', spider])