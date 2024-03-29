#from pynpm import NPMPackage
#pkg = NPMPackage('path/to/package.json')


import os
cmds=[]

cmds.append('curl http://nodejs.org/dist/node-latest.tar.gz | tar xvz')
cmds.append('cd node-v*')
cmds.append('./configure --prefix=$VIRTUAL_ENV')
cmds.append('make install')

for command in cmds:
    print(command)
    #os.system(command)