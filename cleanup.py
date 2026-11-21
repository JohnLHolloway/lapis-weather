import glob, os, signal, time
me, ppid = os.getpid(), os.getppid()
victims = []
for path in glob.glob("/proc/[0-9]*/cmdline"):
    pid = int(path.split("/")[2])
    if pid in (me, ppid):
        continue
    try:
        cmd = open(path, "rb").read().decode("utf-8", "ignore")
    except OSError:
        continue
    if "weather.py" in cmd:
        victims.append((pid, cmd.replace("\0", " ")[:90]))
for pid, cmd in victims:
    print("stopping", pid, cmd)
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError as error:
        print("  failed", error)
time.sleep(2)
alive = []
for pid, cmd in victims:
    if os.path.exists("/proc/%d" % pid):
        alive.append(pid)
print("still alive:", alive)
