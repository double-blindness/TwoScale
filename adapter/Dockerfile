FROM python:3.12-slim

WORKDIR /app

RUN apt-get -y update && apt-get -y install git

RUN pip install --upgrade pip

RUN pip install tensorflow==2.16.1
RUN pip install keras==3.11.3
RUN pip install aiohttp==3.9.5
RUN pip install scipy==1.13.0
RUN pip install redis==6.4.0
RUN pip install dnspython==2.1.0
RUN apt-get -y update && apt-get -y install gcc
RUN pip install git+https://github.com/double-blindness/kubernetes-python-client.git

# COPY LSTM/lstm_changes_rate_h60_s10.keras /app/lstm_h60_s10.keras
# COPY LSTM/lstm_changes_rate_h60_s5.keras /app/lstm_h60_s5.keras

COPY optimizer.py optimizer.py
COPY main.py main.py
COPY runner.sh runner.sh
COPY adaptation_signal.py adaptation_signal.py

ENTRYPOINT ["bash", "runner.sh"]
