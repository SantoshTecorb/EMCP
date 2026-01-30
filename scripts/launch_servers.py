import subprocess
import time
import sys
import os

def launch_server(name, script_path, port):
    print(f"🚀 Starting {name} server on port {port}...")
    # Add src to PYTHONPATH
    env = os.environ.copy()
    project_root = os.getcwd()
    env["PYTHONPATH"] = f"{project_root}:{os.path.join(project_root, 'src')}:{env.get('PYTHONPATH', '')}"
    
    # Run the server as a separate process
    # We use a small helper wrapper to instantiate and run the server class
    cmd = [
        sys.executable,
        "-c",
        f"from servers.{name}_server import {name.capitalize()}MCPServer; import asyncio; s = {name.capitalize()}MCPServer(); s.run()"
    ]
    
    process = subprocess.Popen(
        cmd,
        cwd=project_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env
    )
    return process

if __name__ == "__main__":
    servers = [
        ("documentation", 8001),
        ("tickets", 8002),
        ("runbooks", 8003)
    ]
    
    processes = []
    try:
        for name, port in servers:
            p = launch_server(name, f"src/servers/{name}_server.py", port)
            processes.append((name, p))
            time.sleep(2) # Give it a moment to start
            
        print("\n✅ All servers launched! Press Ctrl+C to stop all.")
        
        while True:
            for name, p in processes:
                if p.poll() is not None:
                    out, err = p.communicate()
                    print(f"❌ Server {name} stopped unexpectedly!")
                    print(f"Error: {err}")
                    sys.exit(1)
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\n🛑 Shutting down servers...")
        for name, p in processes:
            p.terminate()
        print("Done.")
