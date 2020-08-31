import os
import json
import time
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelWithLMHead
from uvicorn.config import logger as api_logger

'''
    timing decorator
'''
def timing(f):
    def wrap(*args, **kwargs):
        time1 = time.time()
        ret = f(*args, **kwargs)
        time2 = time.time()
        api_logger.info('{:s} function took {:.3f} ms'.format(f.__name__, (time2-time1)*1000.0))
        return ret
    return wrap


SPECIAL_CASE_TGT_MAP = {
    "he": "heb",
    "ara": "arb",
    "ar": "arb",
    "en": "eng",
}

class Translator():
    def __init__(
            self,
            source_lang="heb",
            target_lang="arb",
            special_tokens=None
    ):
        '''
        :param source_lang: source langauge to be translated from ["heb","arb","eng"]
        :param target_lang: target langauge to be translated to ["heb","arb", "eng"]
        :param max_cache_entries: maximum entrie
        '''

        self.source_lang = SPECIAL_CASE_TGT_MAP.get(source_lang, source_lang)
        self.target_lang = SPECIAL_CASE_TGT_MAP.get(target_lang, target_lang)
        self.special_tokens = special_tokens

        with open("translator_config.json", "r") as fp:
            languages_dict = json.load(fp)
        try:
            pretrained_model = languages_dict[self.source_lang][self.target_lang]["model_name"]
            special_token = languages_dict[self.source_lang][self.target_lang]["special_tok"]
        except:
            raise NotImplementedError(f"translation {self.source_lang}->{self.target_lang} not available")

        trained_model_path = os.path.join("trained_models", pretrained_model)
        self.special_tok = special_token

        # loading models
        self.tokenizer = AutoTokenizer.from_pretrained(trained_model_path)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        api_logger.info(f"inference device: {self.device}")

        self.trained_model_path = trained_model_path
        self.model : AutoModelWithLMHead = AutoModelWithLMHead.from_pretrained(self.trained_model_path)

        if self.special_tokens is not None:
            spec_dict = {"additional_special_tokens": self.special_tokens}
            num_added = self.tokenizer.add_special_tokens(spec_dict)
            api_logger.info(f"number tokens added: {num_added}")
            self.model.resize_token_embeddings(len(self.tokenizer))  # adding new tokens

        self.model.to(self.device)
        self.max_sequence_len = 512
        self.translation_df = pd.DataFrame(columns=["src_text", "tgt_text"])
        self.translation_cache = {}

    def _get_special_token(self):
        return f">>{self.special_tok}<< ";

    def preprocess(self, x : str):
        '''
        this method can be overriden to define specific preprocess behavior
        :param x: string to be preprocessed for translation
        :return: processed string
        '''
        cur_x = self._get_special_token() + x
        cur_x = cur_x.replace(".", " </s> ")
        return cur_x

    def to(self, device):
        '''
        mode model to device (cuda/cpu)
        :param device:
        :return:
        '''
        self.device = device
        self.model.to(device)


    def clear_cache(self):
        del self.translation_cache
        self.translation_cache = {}


    @timing
    def translate(self, x):
        '''
        translation function
        :param x : [str, [str], pd.Series(str)] - data to be translated:
        :return: returns df : pd.DataFrame with original and translated text
        '''
        cur_translation_df = pd.DataFrame(columns=["src_text", "tgt_text"])
        if type(x) == list:
            cur_translation_df["src_text"] = x
        elif type(x) == str:
            cur_translation_df["src_text"] = [x]
        elif type(x) == pd.Series:
            cur_translation_df["src_text"] = x
        api_logger.info(cur_translation_df)

        no_translations = []
        after_translations = []
        if len(cur_translation_df) > 0:
            for src_text_itr in cur_translation_df["src_text"]:
                translation_itr = self.translation_cache.get(src_text_itr, None)
                if translation_itr is None:
                    # need to translate
                    no_translations.append(src_text_itr)
        # translated new
        if len(no_translations) > 0:
            processed_text_list_no_translation = [self.preprocess(x) for x in no_translations]
            input_dict = self.tokenizer.prepare_translation_batch(processed_text_list_no_translation)

            # TODO fix the special tokens feature
            # special_tokens_attention_mask = ~input_dict["input_ids"] >= torch.Tensor(
            #     self.tokenizer.additional_special_tokens_ids).min()

            input_dict.to(self.device)
            translated = self.model.generate(**input_dict)
            tgt_text = [self.tokenizer.decode(t, skip_special_tokens=True) for t in translated]
            for src_text, translated_txt in zip(no_translations, tgt_text):
                self.translation_cache[src_text] = translated_txt

        # get translations from cache
        for query_text in cur_translation_df["src_text"].tolist():
            after_translations.append(self.translation_cache[query_text])
        cur_translation_df["tgt_text"] = after_translations
        return cur_translation_df

    def free_model_mem(self):
        del self.model
        self.model = None
        torch.cuda.empty_cache()
        api_logger.info(f"freed model {self.trained_model_path} from gpu memory.")

    def load_model_to_gpu(self):
        if self.model is None:
            torch.cuda.empty_cache()
            self.model: AutoModelWithLMHead = AutoModelWithLMHead.from_pretrained(self.trained_model_path)
            api_logger.info(f"loaded model {self.trained_model_path} to gpu memory.")



