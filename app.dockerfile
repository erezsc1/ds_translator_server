FROM translation_image_base

WORKDIR /usr/src
COPY . .

EXPOSE 80

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "80"]