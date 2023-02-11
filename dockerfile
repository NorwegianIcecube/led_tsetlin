FROM python:3.8-bullseye
COPY ./ /app/led_tsetlin
WORKDIR /app/led_tsetlin
RUN pip install git+https://github.com/cair/tmu.git 
ENTRYPOINT [ "bin/bash" ]
CMD ["cd /app/led_tsetlin"]