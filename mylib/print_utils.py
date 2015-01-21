from IPython.core.display import HTML
import operator
def to_html_table(d, columns=None, rows=None):
    '''takes a dictionary of row=>column=>value and returns an HTML table.
    It's important that the columns are always the same'''
    cols = columns if columns else d.itervalues().next().keys()
    ret = "<table>"
    ret += "<tr>"
    ret += "<th></th>"
    for c in cols:
        ret += "<th>{0}</th>".format(c)
    ret += "</tr>"
    if not rows:
        rows = sorted(d.keys())
    for r,res_r in map(lambda k: (k,d[k]), rows):
        ret += "<tr>"
        ret += "<td>{0}</td>".format(r)
        for k,v in map(lambda k: (k,res_r[k]), cols):
            ret += "<td>"
            ret += "{0:.2f}".format(v)
            ret += "</td>"
        ret += "</tr>"
    ret += "</table>"
    return HTML(ret)

def latex_sanitize(s):
    s = s.replace("_", "$\_$")
    return s
    
def to_latex_table(d, columns=None, rows=None, caption=None, label=None, fmt=str):
    if len(d) == 0:
        return
    default_non_cell = "-"
    cols = columns if columns else \
        list(sorted(set.union(*[set(v.keys()) for v in d.itervalues()])))
    ret = "\\begin{table}[tb]" + "\n"
    ret += "\\begin{small}" + "\n"
    ret += "\\tabcolsep=0.11cm" + "\n"
    ret += "\\begin{{tabular}}{{@{{}}{0}@{{}}}}".format("c"*(len(cols)+1)) + "\n"
    ret += "\\hline" + "\n"
    ret += '&'+"&".join([latex_sanitize(str(c)) for c in cols]) + r"\\" + "\n"
    ret += "\\hline" + "\n"
    if not rows:
        rows = sorted(d.keys())
    for r,res_r in map(lambda k: (k,d[k]), rows):
        r = latex_sanitize(str(r))
        ret += "&".join([r] + [fmt(v) for c,v in 
                               map(lambda c: (c,res_r[c] if c in res_r 
                                                        else default_non_cell), 
                                   cols)])
        ret += r"\\" + "\n"
    
    ret += "\\hline" + "\n"
    ret += "\\end{tabular}" + "\n"
    if caption:
        ret += "\\caption{{{0}}}".format(latex_sanitize(caption))
    if label:
        ret += "\\label{{{0}}}".format(label)
    ret += "\\end{small}" + "\n"
    ret += "\\end{table}" + "\n"
    return ret
    