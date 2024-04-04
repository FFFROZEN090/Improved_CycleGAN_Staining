FROM nvcr.io/nvidia/pytorch:21.02-py3

RUN set -ex

# Install dependencies
RUN pip install visdom dominate

# Copy requirements and install
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install -r /tmp/requirements.txt

# Setup Jupyter Notebook (Optional: Configure for your needs)
RUN pip install jupyterlab
RUN jupyter notebook --generate-config --allow-root
RUN echo "c.NotebookApp.ip = '0.0.0.0'" >> ~/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.allow_root = True" >> ~/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.open_browser = False" >> ~/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.token = ''" >> ~/.jupyter/jupyter_notebook_config.py

# Expose port for Jupyter Notebook
EXPOSE 8888