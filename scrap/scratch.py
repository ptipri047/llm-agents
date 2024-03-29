x=5
y=7


def mymethod(a=None,b=None,c=None):
    print(a)
    print(b)
    

func=mymethod    
varname=['x', 'y']
ar = [eval(z) for z in varname]

#ar = {'a':eval(varname)}
mymethod(*ar)