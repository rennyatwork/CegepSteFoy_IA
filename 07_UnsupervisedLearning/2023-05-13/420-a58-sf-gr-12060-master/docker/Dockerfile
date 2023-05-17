FROM python:3.11-bullseye

RUN pip install --upgrade pip

COPY ./requirements.txt /
RUN pip install -r requirements.txt

RUN mkdir /notebooks
RUN mkdir /root/.jupyter
COPY jupyter_notebook_config.py /root/.jupyter

EXPOSE 8888

CMD jupyter notebook /notebooks
