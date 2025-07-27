import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import backend_rag as main

bind = "0.0.0.0:5000"
workers = 1
worker_class = "sync"
timeout = 300

def post_fork(server, worker):
    server.log.info(f"Worker {worker.pid} initializing backend components...")
    main.initialize_backend_components()
    server.log.info(f"Worker {worker.pid} finished initialization.")