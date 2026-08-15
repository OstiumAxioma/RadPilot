import subprocess
import sys

res = subprocess.run([sys.executable, "backend/test_backend.py"], capture_output=True, text=True, encoding="utf-8", errors="ignore")
with open("test_out.log", "w", encoding="utf-8") as f:
    f.write("=== STDOUT ===\n")
    f.write(res.stdout or "")
    f.write("\n=== STDERR ===\n")
    f.write(res.stderr or "")

print("Test complete. Return code:", res.returncode)
