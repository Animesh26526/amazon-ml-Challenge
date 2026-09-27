import subprocess
import re

out = subprocess.check_output("tasklist", shell=True).decode('utf-8', errors='ignore')
lines = out.strip().split('\n')[3:]
procs = []
for line in lines:
    parts = line.strip().split()
    if len(parts) >= 5:
        name = parts[0]
        pid = parts[1]
        mem = parts[-2].replace(',', '')
        if mem.isdigit():
            procs.append((name, pid, int(mem)))

procs.sort(key=lambda x: x[2], reverse=True)
print("Top 10 memory processes:")
for name, pid, mem in procs[:10]:
    print(f"{name:<25} PID {pid:<8} {mem/1024:.1f} MB")
