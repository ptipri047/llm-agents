import requests
from libs.mylib import BaseClass, SubClass

print("hello")
a = 5.0
b = 2
print(f"my val is: {a}")

"""
    this function blabla
"""


def myfunction(myinput):
    print(f"the val blabla {myinput}")
    return 5


myfunction(5.0)
myfunction("abd")

    
myinstance = BaseClass('hello')
myinstance.mymethod(10)

myinstance = SubClass('testsubclass')
myinstance.mymethod(20)
myinstance.submethod()
myinstance.mycall(url='http://www.google.fr')

'''
  this is an example of using type
  this is a keyword
'''
a=True
print(type(a))
print(type(myinstance))

# we are seing array
arr = [1,2,3,6]   # ['srt','try']
print(arr)

'''
   for loop
'''
for i in arr:
    print(i)

b = range(3)
for i in range(5):
    print(f'my i: {i}')
    
c = ['po','pi','ru','rt']
print(type(c))
length  = len(c)  
print(f'len:{length}')  

for st in c:
    print(f'my str: {st}')
    
for i, st in enumerate(c):
    print(f'my str {i}: {st}')
    

mystr = 'my house is big'
arr = mystr.split()
for i,elem in enumerate(arr):
    print(f'my str {i}: {elem}')

'''string manipulation'''
mystr = 'my house is big'
print(type(mystr))

length = len(mystr)
print(f'le: {length}')

ind = mystr.index('i')
print(# The variable `ind` is storing the index of the first occurrence of the character 'i' in the
# string `mystr`.
ind)

mystr = mystr.replace('big','small')
print(mystr)

creditcardnb = 'rty 1058 45'
import re
pat = re.compile(pattern='^[a-z]{3} [0-9]{4}.*')

if (pat.match(creditcardnb)):
    print('match')
else:
   print('no match')       


''' 
if


if <cond:bool>:
   #code

if <cond:bool>:
   #code
else:
  # code
        
if <cond:bool>:
   #code
elif <cond:bool>   
  # code
else:
  # code

'''

mystr = mystr.replace('big','small')
print(mystr)

#creditcardnb = 'rty 1058 45'
creditcardnb=''
import re
pat = re.compile(pattern='^[a-z]{3} [0-9]{4}.*')
match = pat.match(creditcardnb)

if len(creditcardnb) == 0:
    print('you should enter your card nb')
elif not match:
    print('you entered invalid format')
else:    
    print('thank you')

'''import json
import requests
from .conda.Lib.threading import enumerate

response = requests.get("https://randomuser.me/api")

status_code = response.status_code

# cookies
cookies = response.cookies
pop = cookies.get_policy()
ne = pop.netscape
rfc = pop.rfc2965


# get payload
data = response.json()
print(type(data))
print(data)
first = data["results"][0]["name"]["first"]

# first = data['results']['name']
'''