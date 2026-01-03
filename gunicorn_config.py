import multiprocessing

# Gunicorn configuration file
bind = "0.0.0.0:5000"
backlog = 2048

# Workers
# Recommended: 2 * cpu_cores + 1
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = 'sync'  # 'gevent' is better for IO-bound, but 'sync' is safer if not using gevent monkeypatching
worker_connections = 1000
timeout = 30
keepalive = 2

# Logging
accesslog = '-'
errorlog = '-'
loglevel = 'info'

# Process Naming
proc_name = 'plansphere_app'
