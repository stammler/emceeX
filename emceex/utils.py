import shutil

def print_meter(desc, val, thr, tot, cl=None, cr=None, fmt="{:6.2f}"):
    Nw = shutil.get_terminal_size().columns
    pre = desc + " "
    suf = " " + fmt.format(val)
    Nbar = Nw - len(pre) - len(suf)
    Nleft = int(Nbar * thr / tot)
    Nright = Nbar-Nleft
    bar_left = ""
    if Nleft:
        bar_left = _get_bar(val/thr, Nleft, color=cl)
    bar_right = ""
    if Nright:
        bar_right = _get_bar((val-thr)/(tot-thr), Nright, color=cr)
    bar = pre + bar_left + bar_right + suf
    print(bar)
    
def _get_bar(f, N, color=None):
    colors = {
        "black": "\033[30m",
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "purple": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[38m",
        "reset": "\033[0m"
    }
    f = max(0, min(1, f))
    charset = u" " + u''.join(map(chr, range(0x258F, 0x2587, -1)))
    nsyms = len(charset)-1
    bar_length, frac_bar_length = divmod(int(f * N * nsyms), nsyms)
    bar = charset[-1] * bar_length
    if bar_length < N:
        bar = bar + charset[frac_bar_length] + charset[0] * (N - bar_length - 1)
    if color:
        bar = colors[color.lower()] + bar + colors["reset"]
    return bar