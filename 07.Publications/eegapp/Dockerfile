FROM python:3.6
COPY . /app
WORKDIR /app
RUN pip3 install -r requirements.txt
RUN pip3 install torch==1.3.1+cpu torchvision==0.4.2+cpu -f https://download.pytorch.org/whl/torch_stable.html
ENTRYPOINT ["python"]
CMD ["app.py"]