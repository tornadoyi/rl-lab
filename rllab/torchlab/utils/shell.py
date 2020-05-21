import subprocess

def run(cmd, shell=True, timeout=None):
    with subprocess.Popen(cmd, shell=shell, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as p:
        p.wait(timeout)
        if p.returncode == 0: return p.stdout.read().decode()
        else: raise Exception('Command run error\n{}'.format(p.stderr.read().decode()))
