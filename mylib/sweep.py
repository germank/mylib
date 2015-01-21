import glob
import os

def lst(args):
    return glob.glob(os.path.join(args, '*'))


def variable_name(dir_name):
    '''A string is a variable if it is enclosed in angle brackets.
    Eg: This is a variable named var1: <var1>'''
    if len(dir_name)<3:
        return False
    if dir_name[0] == '<' and dir_name[-1] == '>':
        return dir_name[1:-1]
    else:
        return False
            
def find_first_variable(splitted_path):
    #find variable name and position
    for i, dir_name in enumerate(splitted_path):
        var_name = variable_name(dir_name)
        if var_name:
            return i, var_name
    return None, None

def split_path(template_path):
    splitted_path = template_path.split('/')
    if splitted_path[0] == '':
        splitted_path = splitted_path[1:]
    return splitted_path
    
def point(template_path, match):
    '''Uses a dictionary like the one returned by sweep
    to fix the variables in the template_path
    Example:
    /home/user/project/x/<var_a>/<var_b>
    {'var_a': 'foo',
    'var_b': = 'bar
    }
    returns /home/user/project/foo/bar
    '''
    if isinstance(template_path, basestring):
        splitted_path = split_path(template_path)
    else:
        splitted_path = template_path
    ret = []
    for dir_name in splitted_path:
        var_name = variable_name(dir_name)
        if var_name:
            ret.append(match[var_name])
        else:
            ret.append(dir_name)
    return os.path.join(*ret)


def sweep(template_path):
    '''Returns a dictionary instantiating variables like those in:
    /home/user/project/x/<var_a>/<var_b>
    with all subdirectories in the template_path'''
    if isinstance(template_path, basestring):
        splitted_path = split_path(template_path)
    else:
        splitted_path = template_path
    #get some variable in the path to replace
    var_i, var_name = find_first_variable(splitted_path)
    if var_i:
        #replace variable with every matching file/directory 
        for subdir in lst(os.path.join('/', *splitted_path[:var_i])):
            subdir_name = os.path.basename(subdir)
            new_template_path = splitted_path[:var_i] + [subdir_name] + \
                splitted_path[var_i+1:]
            # continue recursively
            for x in sweep(new_template_path):
                x[var_name+ ':path'] = subdir
                x[var_name] = subdir_name
                yield x
    else:
        #base case: no variables, nothing to report
        yield {':path': os.path.join('/', *splitted_path)}
