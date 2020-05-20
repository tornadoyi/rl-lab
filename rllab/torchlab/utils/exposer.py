

def expose(src, dst, override=False, filter=None):
    if filter is None: filter = lambda _: False
    for k in dir(src):
        v = getattr(src, k)
        if (not override and k in dst) or filter(k): continue
        dst[k] = v