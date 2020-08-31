
# MarianMT Service
Machine Translation RESTful API service. Based on Helsinki-NLP repository, and supports the following languages:
- arb <--> eng
- arb --> fr
- arb <--> heb
- arb <--> rus
- arb <--> spa
- arb <--> tur
- de <--> heb
- ell --> arb
- eng --> heb
- es --> heb
- fi <--> heb
- fr --> heb
- fra --> arb
- heb <--> sv
- ita --> arb
- jpn --> arb


All models are stored in /trained_models/
New models can be saved in that directory, and update translator_config.json accordingly.
## Building & Running 
### Docker
#### build: 
build base docker image (libraries, environment and dependencies) on online network
```bash
sudo docker build -f base.dockerfile -t translation_image_base .
```
build application docker image on offline environment
```bash
sudo docker build -f app.dockerfile -t translation_image_app .
```
clean temp containers
```bash
sudo docker system prune
```

#### run:
with gpu:
```bash
nvidia-docker run -p 8000:80 translation_image_app
```
without gpu:
```bash
docker run -p 8000:80 translation_image_app
```
access via ```localhost:8000/docs```

### Uvicorn
```uvicorn api:app --reload --host 0.0.0.0 --port 8000```


## Usage
using the translator service can be done by installing the dedicated client class, or accessing the RESTful API directly.
### Client Class
for client class installations, see: gitlab_translator_client.git
```python
from translation_client import TranslatorClient


if __name__ == '__main__':
    service_url = "http://0.0.0.0:8000"
    tc = TranslatorClient("heb", "arb", service_url)
    query1 = "שלום לכם, ילדים וילדות"
    query2 = [
        "שלום לכם, ילדים וילדות",
        "אני יובל המבולבל"
    ]
    tc.translate(query1)  # مرحباً أيها الأطفال والبنات
    tc.translate(query2)  # ['مرحباً أيها الأطفال والبنات', 'أنا يوبيل مشوّش']

```

### API calls
```python
import requests

URL = r"http://127.0.0.1:8000/translate_list/"

if __name__ == '__main__':
    query = [
        "שלום, אחמד",
        "מה שלומך היום?"
    ]
    src_lang = "heb"
    tgt_lang = "arb"
    request = {
        "source_lang": src_lang,
        "target_lang": tgt_lang,
        "data_list": query,
        "content-type":"application/json"
    }
    response = requests.get(URL, params=request).json()
```


##TODO
- wrap AugmenText so it will work with the API
- add specialized tokens to augmentations part
- crawl over huggingface.co and download all available models
- fix the >>tok<< in some languages that discard the first token