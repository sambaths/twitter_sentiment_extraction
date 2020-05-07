import config
import torch
import numpy as np
import pandas as pd

class TweetDataset:
    def __init__(self, tweet, sentiment, selected_text):
        self.tweet = tweet
        self.sentiment = sentiment
        self.selected_text = selected_text
        self.max_len = config.MAX_LEN
        self.tokenizer = config.TOKENIZER

    def __len__(self):
        return len(self.tweet)

    def __getitem__(self, item):
        tweet = ' '.join(str(self.tweet[item]).lower().split())
        selected_text = ' '.join(str(self.selected_text[item]).lower().split())
        sentiment = self.sentiment[item]
        len_sel_text = len(selected_text)
        idx0 = -1
        idx1 = -1

        for ind in (i for i,e in enumerate(tweet) if e == selected_text[0]):
            if tweet[ind: ind + len_sel_text] == selected_text:
                idx0 = ind
                idx1 = ind + len_sel_text - 1
                break
        
        char_targets = [0] * len(tweet)
        if idx0 != -1 and idx1 != -1:
            for j in range(idx0, idx1):
                if tweet[j] != " ":
                    char_targets[j] = 1
        
        tok_tweet = self.tokenizer.encode(tweet)
        tok_tweet_ids = tok_tweet.ids
        tok_tweet_tokens = tok_tweet.tokens
        tok_tweet_offsets = tok_tweet.offsets

        targets = []
        for j, (offset1, offset2) in enumerate(tok_tweet_offsets):
            if sum(char_targets[offset1:offset2]) > 0:
                targets.append(j)
    
        # targets_start = targets[0]
        # targets_end = targets[-1]
        # targets_start = [0] * config.MAX_LEN
        # targets_end = [0] * config.MAX_LEN
        # non_zero = np.nonzero(targets)[0]
        # print(non_zero)
        # if len(non_zero) > 0:
        #     targets_start[non_zero[0]] = 1
        #     targets_end[non_zero[-1]] = 1

        sentiment_id = {'positive' : self.tokenizer.encode('positive').ids,
            'neutral' : self.tokenizer.encode('neutral').ids,
            'negative' : self.tokenizer.encode('negative').ids }     

        input_ids =  [0] + sentiment_id[sentiment] + [2] + [2] + tok_tweet_ids + [2]
        token_type_ids = [0] * len(input_ids)
        mask = [1] * len(token_type_ids)
        tweet_offsets = [(0, 0)]*4 + tok_tweet.offsets + [(0, 0)]

        targets_start = [0] * config.MAX_LEN
        targets_end = [0] * config.MAX_LEN
        non_zero = np.nonzero(targets)[0]
        if len(non_zero) > 0:
            targets_start[non_zero[0]+4] = 1
            targets_end[non_zero[-1]+4] = 1
        # targets_start += 4
        # targets_end += 4

        padding_length = self.max_len - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + ([1] * padding_length)
            mask = mask + ([0] * padding_length)
            token_type_ids = token_type_ids + ([0] * padding_length)
            tweet_offsets = tweet_offsets + ([(0, 0)] * padding_length)
    
        return {
            'ids': torch.tensor(input_ids, dtype = torch.long),
            'mask': torch.tensor(mask, dtype = torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype = torch.long),
            'targets': torch.tensor(targets, dtype = torch.long),
            'targets_start': torch.tensor(np.argmax(targets_start), dtype = torch.long),
            'targets_end': torch.tensor(np.argmax(targets_end), dtype = torch.long),
            'padding_len': torch.tensor(padding_length, dtype = torch.long),
            'tweet_tokens': ' '.join(tok_tweet_tokens),
            'orig_tweet': self.tweet[item],
            'orig_sentiment': self.sentiment[item],
            'orig_selected':self.selected_text[item],
            'offsets' : torch.tensor(tweet_offsets, dtype=torch.long)
        }


# if __name__=='__main__':
#     dfx = pd.read_csv(config.TRAINING_FILE, nrows=1).dropna().reset_index(drop=True)
#     train_dataset=TweetDataset(
#         tweet=dfx.text.values,
#         sentiment=dfx.sentiment.values,
#         selected_text= dfx.selected_text.values
#     )
#     print(train_dataset[0])

