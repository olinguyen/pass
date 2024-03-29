import os
from IPython.lib import passwd

c = c  # pylint:disable=undefined-variable
c.NotebookApp.ip = '*'
c.IPKernelApp.pylab = 'inline'  # in-line figure when using Matplotlib
c.NotebookApp.port = int(os.getenv('PORT', 8888))
# Allow to run Jupyter from root user inside Docker container
c.NotebookApp.allow_root = True
c.NotebookApp.open_browser = False

# sets a password if PASSWORD is set in the environment
if 'PASSWORD' in os.environ:
    password = os.environ['PASSWORD']
    if password:
        c.NotebookApp.password = passwd(password)
    else:
        c.NotebookApp.password = ''
        c.NotebookApp.token = ''
    del os.environ['PASSWORD']
