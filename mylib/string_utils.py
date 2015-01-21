def eat(s1, s2, reverse=False):
    '''Removes s2 from the beginning of s1, or from the end if reverse=True
    If s1 does not begin (or ends) with s2, it leaves the string unchanged'''
    l = len(s2)
    if reverse:
        rest, bite = s1[:-l], s1[-l:]
    else:
        rest, bite = s1[l:], s1[:l]
    if bite == s2:
        return rest
    else:
        return s1