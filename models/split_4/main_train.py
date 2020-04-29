import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np # linear algebra
import pandas as pd
import math

from transformers import BertForSequenceClassification, BertTokenizer, BertConfig
from transformers import RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig
from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizer, XLMRobertaConfig
from transformers import XLNetForSequenceClassification, XLNetTokenizer, XLNetConfig
from transformers import XLMForSequenceClassification, XLMTokenizer, XLMConfig
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, DistilBertConfig
from fastai.text import *


from transformers import AdamW
from functools import partial


############################################### MODEL ###########################################################
MODEL_CLASSES = {
    'bert': (BertForSequenceClassification, BertTokenizer, BertConfig),
    'xlnet': (XLNetForSequenceClassification, XLNetTokenizer, XLNetConfig),
    'xlm': (XLMForSequenceClassification, XLMTokenizer, XLMConfig),
    'roberta': (RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig),
    'distilbert': (DistilBertForSequenceClassification, DistilBertTokenizer, DistilBertConfig),
    'xlm-roberta': (XLMRobertaForSequenceClassification, XLMRobertaTokenizer, XLMRobertaConfig)
}


class TransformersBaseTokenizer(BaseTokenizer):
    """Wrapper around PreTrainedTokenizer to be compatible with fast.ai"""

    def __init__(self, pretrained_tokenizer, model_type, **kwargs):
        self._pretrained_tokenizer = pretrained_tokenizer
        self.max_seq_len = pretrained_tokenizer.max_len
        self.model_type = model_type

    def __call__(self, *args, **kwargs):
        return self

    def tokenizer(self, t: str) -> List[str]:
        """Limits the maximum sequence length and add the spesial tokens"""
        CLS = self._pretrained_tokenizer.cls_token
        SEP = self._pretrained_tokenizer.sep_token
        if self.model_type in ['roberta', 'xlm-roberta']:
            tokens = self._pretrained_tokenizer.tokenize(t, add_prefix_space=True)[:self.max_seq_len - 2]
            tokens = [CLS] + tokens + [SEP]
        else:
            tokens = self._pretrained_tokenizer.tokenize(t)[:self.max_seq_len - 2]
            if self.model_type in ['xlnet']:
                tokens = tokens + [SEP] + [CLS]
            else:
                tokens = [CLS] + tokens + [SEP]
        return tokens



class TransformersVocab(Vocab):
    def __init__(self, tokenizer):
        super(TransformersVocab, self).__init__(itos=[])
        self.tokenizer = tokenizer

    def numericalize(self, t: Collection[str]) -> List[int]:
        "Convert a list of tokens `t` to their ids."
        return self.tokenizer.convert_tokens_to_ids(t)
        # return self.tokenizer.encode(t)

    def textify(self, nums: Collection[int], sep=' ') -> List[str]:
        "Convert a list of `nums` to their tokens."
        nums = np.array(nums).tolist()
        return sep.join(
            self.tokenizer.convert_ids_to_tokens(nums)) if sep is not None else self.tokenizer.convert_ids_to_tokens(
            nums)

    def __getstate__(self):
        return {'itos': self.itos, 'tokenizer': self.tokenizer}

    def __setstate__(self, state: dict):
        self.itos = state['itos']
        self.tokenizer = state['tokenizer']
        self.stoi = collections.defaultdict(int, {v: k for k, v in enumerate(self.itos)})


class CustomTransformerModel(nn.Module):

    def __init__(self, transformer_model):
        super(CustomTransformerModel, self).__init__()
        self.transformer = transformer_model

    def forward(self, input_ids):
        # Return only the logits from the transfomer
        logits = self.transformer(input_ids)[0]
        return logits


if __name__ == '__main__':

    # choose GPU device #0
    torch.cuda.set_device(0)

    # Seed to split the data
    SEED = 66111134

    ############################################### Load data###########################################################
    root_path = '../../data/'

    train = pd.read_csv(root_path + 'Train.csv')

    # train = train.drop(['agreement'], axis=1)
    train.columns = ['id', 'comment_text', 'label', 'agreement']

    agreement = np.unique(np.asarray(train['agreement']), return_counts=True)
    values = agreement[0][1]

    # Load only if agreement != 0.333
    train_df = []
    for i in range(len(train)):
        if math.isnan(train['label'][i]) == False:
            if train['agreement'][i] == 1. or train['agreement'][i] == values:
                if train['label'][i] != 1.0 and train['label'][i] != -1.0 and train['label'][i] != 0.0:
                    train['label'][i] = float(round(train['label'][i], 0))
                    train_df.append([train['id'][i], train['comment_text'][i], train['label'][i]])
                else:
                    train_df.append([train['id'][i], train['comment_text'][i], train['label'][i]])

    train_df = pd.DataFrame(train_df)
    train_df.columns = ['id', 'comment_text', 'label']

    ############################################### Choose model ###########################################################

    model_type = 'roberta'
    model_class, tokenizer_class, config_class = MODEL_CLASSES[model_type]

    model_name = 'roberta-base'
    transformer_tokenizer = tokenizer_class.from_pretrained(model_name)
    transformer_base_tokenizer = TransformersBaseTokenizer(pretrained_tokenizer=transformer_tokenizer,
                                                           model_type=model_type)
    fastai_tokenizer = Tokenizer(tok_func=transformer_base_tokenizer, pre_rules=[], post_rules=[])

    transformer_vocab = TransformersVocab(tokenizer=transformer_tokenizer)
    numericalize_processor = NumericalizeProcessor(vocab=transformer_vocab)

    tokenize_processor = TokenizeProcessor(tokenizer=fastai_tokenizer,
                                           include_bos=False,
                                           include_eos=False)

    transformer_processor = [tokenize_processor, numericalize_processor]


    # Choose a suitable batch size
    bs = 8
    pad_first = bool(model_type in ['xlnet'])
    pad_idx = transformer_tokenizer.pad_token_id

    # Split into 90%-10% with seed 666
    databunch = (TextList.from_df(train_df, cols='comment_text', processor=transformer_processor)
                 .split_by_rand_pct(0.1, seed=SEED)
                 .label_from_df(cols='label')
                 .databunch(bs=bs, pad_first=pad_first, pad_idx=pad_idx))

    print(databunch)

    # Load model with 3 clases (we modify last layer afterwards)
    config = config_class.from_pretrained(model_name)
    config.num_labels = 3

    transformer_model = model_class.from_pretrained(model_name, config=config)
    custom_transformer_model = CustomTransformerModel(transformer_model=transformer_model)


    # Define learner
    CustomAdamW = partial(AdamW, correct_bias=False)
    learner = Learner(databunch, custom_transformer_model, opt_func=CustomAdamW, metrics=[rmse])

    # Set directory to save models
    # Make sure that you have the permission to write, otherwise it fails to save the model and permissions need to be changed by sudo chmod 777
    os.makedirs('models/')
    learner.model_dir = 'models/'

    # Change last layer
    learner.model.transformer.classifier.out_proj = torch.nn.Linear(in_features=768, out_features=1, bias=True).cuda(0)

    # create layer groups
    list_layers = [learner.model.transformer.roberta.embeddings,
                   learner.model.transformer.roberta.encoder.layer[0],
                   learner.model.transformer.roberta.encoder.layer[1],
                   learner.model.transformer.roberta.encoder.layer[2],
                   learner.model.transformer.roberta.encoder.layer[3],
                   learner.model.transformer.roberta.encoder.layer[4],
                   learner.model.transformer.roberta.encoder.layer[5],
                   learner.model.transformer.roberta.encoder.layer[6],
                   learner.model.transformer.roberta.encoder.layer[7],
                   learner.model.transformer.roberta.encoder.layer[8],
                   learner.model.transformer.roberta.encoder.layer[9],
                   learner.model.transformer.roberta.encoder.layer[10],
                   learner.model.transformer.roberta.encoder.layer[11],
                   learner.model.transformer.roberta.pooler,
                   learner.model.transformer.classifier]

    learner.split(list_layers)

    # Train last layer first
    learner.freeze_to(-1)
    lr = 1e-3
    learner.fit_one_cycle(5, slice(lr), moms=(0.8, 0.7))
    learner.save('stage_1')
    # Train the whole network with lr/10.
    learner.unfreeze()
    learner.fit_one_cycle(4, slice(lr/10.), moms=(0.8, 0.7))
    learner.save('stage_final')



