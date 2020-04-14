import config
import torch

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
        tweet = ' '.join(str(self.tweet[item]).split())
        selected_text = ' '.join(str(self.selected_text[item]).split())

        len_sel_text = len(selected_text)
        idx0 = -1
        idx1 = -1

        for ind in (i for i,e in enumerate(tweet) if e == selected_text[0])):
            if tweet[ind: ind + len_sel_text] == selected_text:
                idx0 = ind
                idx1 = ind + len_sel_text - 1
                break
        
        char_targets = [0] * len(tweet)
        if idx0 != -1 and idx1 != -1:
            for j in range(idx0, idx+1):
                if tweet[j] != " ":
                    char_targets[j] = 1
        
        tok_tweet = self.tokenizer.encode(tweet)
        tok_tweet_tokens = tok_tweet.tokens
        tok_tweet_ids = tok_tweet.ids
        tok_tweet_offsets = tok_tweet.offsets[1:-1]

        targets = [0] * (len(tok_tweet_tokens) - 2)
        for j, (offset1, offset2) in enumerate(tok_tweet_tokens):
            if sum(char_targets[offset1:offset2]) > 0:
                targets[j] = 1

