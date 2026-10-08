import subprocess, time, torch
def log(m): print(f"[{time.strftime('%H:%M:%S')}] [diag3] {m}", flush=True)
log(subprocess.run(["nvidia-smi", "topo", "-m"], capture_output=True, text=True).stdout)
n = torch.cuda.device_count()
for i in range(n):
    log(f"cuda:{i} peers: " + str([j for j in range(n) if j != i and torch.cuda.can_device_access_peer(i, j)]))
x = torch.randn(1, 1, 5120, device="cuda:0", dtype=torch.float32)
for dst in range(1, n):
    for sz, name in ((5120, "20KB"), (5120 * 600, "12MB")):
        t = torch.randn(sz, device="cuda:0")
        t.to(f"cuda:{dst}"); torch.cuda.synchronize(0); torch.cuda.synchronize(dst)
        s = time.time()
        for _ in range(20):
            t.to(f"cuda:{dst}")
        torch.cuda.synchronize(0); torch.cuda.synchronize(dst)
        log(f"hop cuda:0 -> cuda:{dst} {name}: {(time.time()-s)/20*1e3:.2f} ms")

# host-staged hop (avoids the GPU<->GPU peer path)
def host_hop(t, dst):
    return t.cpu().to(f"cuda:{dst}")
for dst in range(1, n):
    for sz, name in ((5120, "20KB"), (5120 * 600, "12MB")):
        t = torch.randn(sz, device="cuda:0")
        host_hop(t, dst); torch.cuda.synchronize(0); torch.cuda.synchronize(dst)
        s = time.time()
        for _ in range(20):
            host_hop(t, dst)
        torch.cuda.synchronize(0); torch.cuda.synchronize(dst)
        log(f"HOST-STAGED hop cuda:0 -> cuda:{dst} {name}: {(time.time()-s)/20*1e3:.2f} ms")
# plain same-device small op latency for reference
a = torch.randn(5120, device="cuda:0"); torch.cuda.synchronize()
s = time.time()
for _ in range(200): a = a * 1.0001
torch.cuda.synchronize(); log(f"same-device 200 tiny kernels: {(time.time()-s)*1e3:.2f} ms")
